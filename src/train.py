"""Train five imbalance-aware classifiers on an embargoed temporal split."""
import json
import os
import time

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import FEATURE_COLUMNS, HORIZON_HOURS, MODEL_DIR, PROCESSED_PATH, REPORT_DIR, SEED, TARGET, ensure_directories

MODEL_NAMES = ["logistic_regression", "random_forest", "hist_gradient_boosting", "xgboost", "lightgbm"]


def temporal_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Split timestamps 70/15/15; purge 24h before validation and test.

    Labels look forward 24h, so the purge keeps their outcome windows from
    crossing either boundary. Percentages describe windows before purging.
    """
    result = data.copy()
    result["timestamp"] = pd.to_datetime(result.timestamp, errors="raise", utc=True)
    result = result.sort_values(["timestamp", "server_id"], kind="stable")
    times = result.timestamp.drop_duplicates().reset_index(drop=True)
    if len(times) < 168:
        raise ValueError("At least 168 unique hourly timestamps are needed for three temporal windows.")
    validation_start = times.iloc[len(times) * 70 // 100]
    test_start = times.iloc[len(times) * 85 // 100]
    gap = pd.Timedelta(hours=HORIZON_HOURS)
    train = result[result.timestamp < validation_start - gap].copy()
    validation = result[(result.timestamp >= validation_start) & (result.timestamp < test_start - gap)].copy()
    test = result[result.timestamp >= test_start].copy()
    metadata = {"horizon_hours": HORIZON_HOURS, "fractions_before_purge": [0.70, 0.15, 0.15],
                "purged_rows": len(result) - len(train) - len(validation) - len(test)}
    for name, subset in [("training", train), ("validation", validation), ("test", test)]:
        if subset.empty:
            raise ValueError(f"The {name} window is empty after the 24-hour purge. Generate more days.")
        metadata[name] = {
            "start": subset.timestamp.min().isoformat(), "end": subset.timestamp.max().isoformat(),
            "rows": len(subset), "positive_rate": float(subset[TARGET].mean()),
        }
    return train, validation, test, metadata


def require_both_classes(data: pd.DataFrame, name: str) -> None:
    """Fail clearly when a window cannot support rare-event evaluation."""
    if set(data[TARGET].unique()) != {0, 1}:
        raise ValueError(f"The {name} window must contain both classes. Generate more servers/days.")


def training_class_weight(labels) -> float:
    """Caller supplies TRAIN labels only; never infer weights from other windows."""
    labels = pd.Series(labels)
    if labels.isna().any() or set(labels.unique()) != {0, 1}:
        raise ValueError("Training labels must contain both binary classes without missing values.")
    return float(labels.eq(0).sum() / labels.eq(1).sum())


def build_models(training_labels) -> dict[str, Pipeline]:
    """Use train-fitted imputers and class weights, without synthetic resampling."""
    weight = training_class_weight(training_labels)
    workers = int(os.environ.get("SENTINELDC_N_JOBS", "4"))
    if workers == 0 or workers < -1:
        raise ValueError("SENTINELDC_N_JOBS must be -1 or a positive integer.")
    estimators = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=600, random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=14, min_samples_leaf=8,
            class_weight="balanced", max_samples=0.65, n_jobs=workers, random_state=SEED,
        ),
        # Disable the estimator's random early-stopping validation split.
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=120, max_leaf_nodes=15,
            l2_regularization=2.0, early_stopping=False, random_state=SEED,
        ),
    }
    estimators["xgboost"] = XGBClassifier(
        n_estimators=250, max_depth=5, learning_rate=0.05, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.1,
        tree_method="hist", device="cpu", max_bin=128, scale_pos_weight=weight,
        n_jobs=workers, random_state=SEED, eval_metric="logloss",
    )
    estimators["lightgbm"] = LGBMClassifier(
        n_estimators=250, num_leaves=31, max_depth=6, learning_rate=0.05,
        min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=5.0, reg_alpha=0.1, max_bin=127, scale_pos_weight=weight,
        n_jobs=workers, random_state=SEED, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )
    models = {}
    for name, estimator in estimators.items():
        steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
        if name == "logistic_regression":
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", estimator))
        models[name] = Pipeline(steps)
    return models


def main() -> dict[str, Pipeline]:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError("Preprocess data first: python -m src.preprocess")
    ensure_directories()
    data = pd.read_csv(PROCESSED_PATH)
    train, _, _, split = temporal_split(data)
    require_both_classes(train, "training")
    for name in ["training", "validation", "test"]:
        info = split[name]
        print(f"{name.title()}: {info['start']} to {info['end']} | "
              f"{info['rows']:,} rows | Positive failures: {info['positive_rate']:.2%}")
    print(f"Purged rows across both boundaries: {split['purged_rows']:,}")
    models = build_models(train[TARGET])
    training_records = []
    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        started = time.perf_counter()
        model.fit(train[FEATURE_COLUMNS], train[TARGET])
        training_records.append({"model": name, "fit_seconds": time.perf_counter() - started})
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2) + "\n")
    (REPORT_DIR / "split_metadata.json").write_text(json.dumps(split, indent=2) + "\n")
    pd.DataFrame(training_records).to_csv(REPORT_DIR / "training_runtime.csv", index=False)
    (MODEL_DIR / "training_class_weights.json").write_text(json.dumps({
        "source_split": "training", "negative_samples": int(train[TARGET].eq(0).sum()),
        "positive_samples": int(train[TARGET].eq(1).sum()),
        "scale_pos_weight": training_class_weight(train[TARGET]),
    }, indent=2) + "\n")
    return models


if __name__ == "__main__":
    main()
