"""Small model smoke tests and training-only imputation checks."""
import json

import joblib
import numpy as np
import pytest

from src.config import FEATURE_COLUMNS, TARGET
from src.evaluate import select_best_model
from src.generate_data import generate_telemetry
from src.predict import EXAMPLE_RECORD, predict_record, risk_level
from src.preprocess import preprocess_data
from src.train import build_models, temporal_split


def test_models_fit_and_predict_without_imputation_leakage(tmp_path):
    data = preprocess_data(generate_telemetry(40, 30, 42))
    train, _, test, _ = temporal_split(data)
    x_train, x_test = train[FEATURE_COLUMNS].copy(), test[FEATURE_COLUMNS].copy()
    x_train.loc[x_train.index[:5], "temperature_c"] = np.nan
    x_test["temperature_c"] = 119.0
    models = build_models(train[TARGET])
    models["random_forest"].set_params(classifier__n_estimators=10)
    models["hist_gradient_boosting"].set_params(classifier__max_iter=10)
    for name in ["xgboost", "lightgbm"]:
        models[name].set_params(classifier__n_estimators=10)
    for model in models.values():
        model.fit(x_train, train[TARGET])
        scores = model.predict_proba(x_test)[:, 1]
        assert np.isfinite(scores).all()
        assert ((scores >= 0) & (scores <= 1)).all()
        position = FEATURE_COLUMNS.index("temperature_c")
        assert model.named_steps["imputer"].statistics_[position] == x_train.temperature_c.median()
    path = tmp_path / "best_failure_model.joblib"
    joblib.dump(models["logistic_regression"], path)
    (tmp_path / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS))
    (tmp_path / "decision_threshold.json").write_text(json.dumps({"selected_threshold": 0.35}))
    result = predict_record(EXAMPLE_RECORD, path)
    assert result["decision_threshold"] == 0.35
    assert result["predicted_failure_within_24h"] == (result["model_risk_score"] >= 0.35)
    assert 0 <= result["model_risk_score"] <= 1
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    incomplete = {**EXAMPLE_RECORD, "temperature_c": None}
    assert np.isfinite(predict_record(incomplete, path)["model_risk_score"])


@pytest.mark.parametrize("score,expected", [(0, "LOW"), (0.2, "MEDIUM"), (0.5, "HIGH"), (0.8, "CRITICAL"), (1, "CRITICAL")])
def test_risk_boundaries(score, expected):
    assert risk_level(score) == expected


def test_predict_rejects_bad_record():
    with pytest.raises(ValueError, match="Missing telemetry fields"):
        predict_record({})
    with pytest.raises(ValueError, match="valid range"):
        predict_record({**EXAMPLE_RECORD, "cpu_utilization": 101})


def test_selection_prioritizes_ap_with_recall_tiebreak():
    import pandas as pd
    metrics = pd.DataFrame([
        {"model": "a", "pr_auc": 0.8, "recall": 0.5, "precision": 0.6, "f1": 0.6},
        {"model": "b", "pr_auc": 0.797, "recall": 0.7, "precision": 0.6, "f1": 0.65},
        {"model": "c", "pr_auc": 0.6, "recall": 0.99, "precision": 0.6, "f1": 0.65},
    ])
    assert select_best_model(metrics) == "a"
