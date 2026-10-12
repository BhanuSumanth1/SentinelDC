"""Select using validation only, then evaluate one frozen model on final test."""
import json
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve, fbeta_score, brier_score_loss,
)

from src.config import FEATURE_COLUMNS, FIGURE_DIR, MODEL_DIR, PROCESSED_PATH, REPORT_DIR, TARGET, ensure_directories
from src.train import MODEL_NAMES, require_both_classes, temporal_split
from src.event_evaluation import print_event_summary
from src.calibration import CalibratedFailureModel, calibration_windows
from src.alert_policy import AlertPolicy, apply_alert_policy, evaluate_operational_alerts
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

THRESHOLDS = np.round(np.arange(0.01, 0.9901, 0.001), 3)
SELECTION_RULE = "Highest validation PR-AUC (average precision); exact ties: F2 at 0.5, then model name."
THRESHOLD_RULE = "Highest policy-validation F2 among recall >= 0.80 and precision >= 0.20; otherwise highest F2 overall. Ties: precision, recall, higher threshold."
EMERGENCY_RULE = "Maximum of 0.50, policy-validation calibrated probability 99.9th percentile, and selected threshold + 0.05; capped at 0.999999."


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def classification_metrics(y_true, scores, threshold: float) -> dict:
    """Compute metrics from one cached score vector without refitting."""
    predicted = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "f2": float(fbeta_score(y_true, predicted, beta=2, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, predicted)), "threshold": float(threshold),
        "true_positive": int(tp), "false_positive": int(fp),
        "true_negative": int(tn), "false_negative": int(fn),
    }


def select_best_model(metrics: pd.DataFrame) -> str:
    """Only validation AP determines the winner except exact ties."""
    tie_metric = "f2" if "f2" in metrics else "f1"
    return str(metrics.sort_values(["pr_auc", tie_metric, "model"],
                                   ascending=[False, False, True]).iloc[0]["model"])


def analyze_thresholds(y_true, scores) -> tuple[pd.DataFrame, float]:
    """Efficient exact counts for a fine, fixed validation-only threshold grid."""
    labels, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    order = np.argsort(scores)
    ordered = scores[order]
    positives = np.r_[0, np.cumsum(labels[order])]
    positions = np.searchsorted(ordered, THRESHOLDS, side="left")
    tp = positives[-1] - positives[positions]
    fp = len(labels) - positions - tp
    fn = positives[-1] - tp
    precision = np.divide(tp, tp + fp, out=np.zeros(len(tp), dtype=float), where=tp + fp > 0)
    recall = tp / positives[-1]
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(tp), dtype=float), where=2 * tp + fp + fn > 0)
    f2 = np.divide(5 * tp, 5 * tp + fp + 4 * fn, out=np.zeros(len(tp), dtype=float), where=5 * tp + fp + 4 * fn > 0)
    results = pd.DataFrame({"threshold": THRESHOLDS, "precision": precision,
        "recall": recall, "f1": f1, "f2": f2, "false_positive": fp.astype(int),
        "false_negative": fn.astype(int), "true_positive": tp.astype(int),
        "false_alerts_per_1000_healthy_observations": 1000 * fp / (len(labels) - positives[-1])})
    results["preferred_constraints_met"] = (results.recall >= 0.80) & (results.precision >= 0.20)
    eligible = results[results.preferred_constraints_met]
    if eligible.empty:
        eligible = results
    best = eligible.sort_values(["f2", "precision", "recall", "threshold"],
                                ascending=[False, False, False, False]).iloc[0]
    selected = float(best.threshold)
    results["selected"] = results.threshold.eq(selected)
    return results, selected


def select_emergency_threshold(probabilities, threshold):
    return float(min(0.999999, max(0.50, np.quantile(probabilities, 0.999), threshold + 0.05)))


def window_metadata(frame):
    timestamps = pd.to_datetime(frame.timestamp, utc=True)
    return {"start": timestamps.min().isoformat(), "end": timestamps.max().isoformat(),
            "rows": len(frame), "positive_rows": int(frame[TARGET].sum())}


def calibration_report(labels, raw, calibrated, split):
    return [{"split": split, "score_type": name, "brier_score": float(brier_score_loss(labels, values))}
            for name, values in [("uncalibrated_model_score", raw), ("calibrated_failure_probability", calibrated)]]


def save_reliability_plot(labels, raw, calibrated, split, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for name, values in [("Uncalibrated", raw), ("Calibrated", calibrated)]:
        fraction, mean = calibration_curve(labels, values, n_bins=10, strategy="quantile")
        ax.plot(mean, fraction, marker="o", label=name)
    ax.set(xlabel="Mean predicted probability / score", ylabel="Observed failure frequency",
           title=f"{split}: reliability (10 quantile bins)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURE_DIR / filename, dpi=150); plt.close(fig)


def select_on_validation(validation: pd.DataFrame) -> dict:
    """No test argument: compare frozen TRAIN models, calibrate, select policy."""
    ensure_directories()
    require_both_classes(validation, "validation")
    validation = validation.reset_index(drop=True)
    calibration, policy_validation = calibration_windows(validation)
    rows, scores = [], {}
    # Load one model at a time to keep memory bounded.
    for name in MODEL_NAMES:
        model = joblib.load(MODEL_DIR / f"{name}.joblib")
        scores[name] = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
        rows.append({"model": name, "split": "validation", "score_type": "uncalibrated_model_score",
                     **classification_metrics(validation[TARGET], scores[name], 0.50)})
    metrics = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    name = select_best_model(metrics)
    metrics["selected"] = metrics.model.eq(name)
    for filename in ["advanced_validation_model_metrics.csv", "validation_model_metrics.csv", "model_metrics.csv"]:
        metrics.to_csv(REPORT_DIR / filename, index=False)
    base_model = joblib.load(MODEL_DIR / f"{name}.joblib")
    calibrated_model = CalibratedFailureModel(base_model).fit_calibration(
        scores[name][calibration.index], calibration[TARGET])
    raw_policy_scores = scores[name][policy_validation.index]
    policy_scores = calibrated_model.calibrate(raw_policy_scores)
    analysis, threshold = analyze_thresholds(policy_validation[TARGET], policy_scores)
    for filename in ["advanced_threshold_analysis.csv", "threshold_analysis.csv"]:
        analysis.to_csv(REPORT_DIR / filename, index=False)
    emergency = select_emergency_threshold(policy_scores, threshold)
    policy = AlertPolicy(threshold, emergency, cooldown_hours=float(os.environ.get("SENTINELDC_COOLDOWN_HOURS", "6")))
    chosen = analysis.loc[analysis.selected].iloc[0]
    selection = {"model_name": name, "selected_threshold": threshold,
        "validation_pr_auc": float(metrics.loc[metrics.selected, "pr_auc"].iloc[0]),
        "selection_split": "validation", "model_selection_rule": SELECTION_RULE,
        "threshold_rule": THRESHOLD_RULE, "calibrated": True,
        "score_type": "calibrated_failure_probability", "calibration_method": "unweighted_Platt_on_logit_scores",
        "calibration_window": window_metadata(calibration), "policy_validation_window": window_metadata(policy_validation),
        "calibration_policy_purge_hours": 24,
        "preferred_constraints_met": bool(chosen.preferred_constraints_met),
        "validation_metrics": classification_metrics(policy_validation[TARGET], policy_scores, threshold),
        "emergency_threshold": emergency, "emergency_rule": EMERGENCY_RULE,
        "validation_emergency_observations": int((policy_scores >= emergency).sum()),
        "alert_persistence_hours": policy.persistence_hours, "cooldown_hours": policy.cooldown_hours,
        "policy_rule": "Two consecutive hourly probabilities >= threshold OR one >= emergency; then per-server cooldown.",
        "calibration_assessment_note": "Later validation window is disjoint from calibration fitting; model choice uses all validation, so this is development evidence, not an independent holdout."}
    # Save the entire frozen policy and calibrated model before any TEST access.
    joblib.dump(calibrated_model, MODEL_DIR / "best_failure_model.joblib")
    save_json(MODEL_DIR / "decision_threshold.json", selection)
    save_json(MODEL_DIR / "best_model_metadata.json", selection)
    pd.DataFrame(calibration_report(policy_validation[TARGET], raw_policy_scores, policy_scores, "policy_validation")).to_csv(
        REPORT_DIR / "calibration_metrics.csv", index=False)
    save_reliability_plot(policy_validation[TARGET], raw_policy_scores, policy_scores, "POLICY VALIDATION", "validation_reliability_curve.png")
    fig, ax = plt.subplots(figsize=(7, 5))
    for metric in ["precision", "recall", "f1", "f2"]:
        ax.plot(analysis.threshold, analysis[metric], label=metric.upper())
    ax.axvline(threshold, color="black", linestyle="--", label=f"Selected {threshold:.3f}")
    ax.set(xlabel="Calibrated probability threshold", ylabel="Metric", ylim=(0, 1.02), title="POLICY VALIDATION: threshold tradeoff")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURE_DIR / "threshold_tradeoff.png", dpi=150); plt.close(fig)
    save_feature_importance(base_model, name, policy_validation)
    print("\nVALIDATION model comparison (uncalibrated scores, threshold 0.50)")
    print(metrics[["model", "pr_auc", "recall", "precision", "f1", "roc_auc"]].to_string(index=False))
    print(f"Selected: {name} | Calibrated threshold: {threshold:.3f} | Emergency: {emergency:.6f}")
    print(f"Preferred recall >= 0.80 AND precision >= 0.20 met: {selection['preferred_constraints_met']}")
    if not selection["preferred_constraints_met"]:
        print("Preferred operating constraints NOT met; using the prespecified highest-F2 fallback.")
    return selection


def save_feature_importance(model, name="random_forest", validation=None) -> None:
    estimator = model.named_steps["classifier"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        method = "built_in_feature_importance"
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
        method = "absolute_standardized_logistic_coefficient"
    else:
        # sklearn HGB has no public built-in feature_importances_.
        result = permutation_importance(model, validation[FEATURE_COLUMNS], validation[TARGET],
            scoring="average_precision", n_repeats=3, random_state=42, n_jobs=1)
        values = result.importances_mean
        method = "validation_permutation_AP_decrease"
    importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": values,
        "model": name, "method": method}).sort_values("importance", ascending=False)
    for filename in ["advanced_feature_importance.csv", "feature_importance.csv"]:
        importance.to_csv(REPORT_DIR / filename, index=False)
    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top.feature, top.importance, color="#327c96")
    ax.set(xlabel=method + " (not causal)", title=f"{name}: top 20 features (before calibration)")
    fig.tight_layout()
    for filename in ["advanced_feature_importance.png", "feature_importance.png"]:
        fig.savefig(FIGURE_DIR / filename, dpi=150)
    plt.close(fig)


def operational_metrics(metrics: dict, observation_count: int) -> dict:
    """Counts are positive telemetry observations, not deduplicated incidents."""
    healthy = metrics["true_negative"] + metrics["false_positive"]
    return {"failures_detected": metrics["true_positive"], "failures_missed": metrics["false_negative"],
            "false_alerts": metrics["false_positive"],
            "total_actual_failures": metrics["true_positive"] + metrics["false_negative"],
            "detection_rate": metrics["recall"],
            "false_alerts_per_1000_observations": 1000 * metrics["false_positive"] / observation_count,
            "false_alerts_per_1000_healthy_observations": 1000 * metrics["false_positive"] / healthy if healthy else np.nan,
            "healthy_telemetry_observations": healthy,
            "telemetry_observations": observation_count, "count_unit": "telemetry_observation"}


def evaluate_final_test(test: pd.DataFrame, selection: dict, split: dict) -> tuple[dict, dict]:
    """One frozen base-model inference on TEST; all other analysis reuses scores."""
    require_both_classes(test, "test")
    saved = json.loads((MODEL_DIR / "decision_threshold.json").read_text())
    if saved != selection:
        raise ValueError("Selection differs from saved threshold. Rerun validation selection.")
    model = joblib.load(MODEL_DIR / "best_failure_model.joblib")
    raw_scores, scores = model.predict_components(test[FEATURE_COLUMNS])
    threshold = selection["selected_threshold"]
    metrics = {"model": selection["model_name"], "split": "final_test",
               "score_type": "calibrated_failure_probability",
               **classification_metrics(test[TARGET], scores, threshold)}
    pd.DataFrame([metrics]).to_csv(REPORT_DIR / "final_test_metrics.csv", index=False)
    row_metrics = operational_metrics(metrics, len(test))
    pd.DataFrame([row_metrics]).to_csv(REPORT_DIR / "raw_prediction_metrics.csv", index=False)
    # Historical filename retained with explicit count_unit=telemetry_observation.
    pd.DataFrame([row_metrics]).to_csv(REPORT_DIR / "operational_metrics.csv", index=False)
    policy = AlertPolicy(threshold, selection["emergency_threshold"], selection["alert_persistence_hours"], selection["cooldown_hours"])
    scored = apply_alert_policy(test, scores, policy)
    operational = evaluate_operational_alerts(test, scored, REPORT_DIR, FIGURE_DIR)
    calibration = pd.read_csv(REPORT_DIR / "calibration_metrics.csv")
    calibration = calibration[calibration.split.ne("final_test")]
    calibration = pd.concat([calibration, pd.DataFrame(calibration_report(test[TARGET], raw_scores, scores, "final_test"))], ignore_index=True)
    calibration.to_csv(REPORT_DIR / "calibration_metrics.csv", index=False)
    save_reliability_plot(test[TARGET], raw_scores, scores, "FINAL TEST", "reliability_curve.png")
    metadata = {**selection, "test_pr_auc": metrics["pr_auc"], "test_precision": metrics["precision"],
                "test_recall": metrics["recall"], "test_f1": metrics["f1"], "feature_count": len(FEATURE_COLUMNS)}
    for name in ["training", "validation", "test"]:
        metadata[f"{name}_date_range"] = {key: split[name][key] for key in ["start", "end"]}
    save_json(MODEL_DIR / "model_metadata.json", metadata)
    save_final_plots(test[TARGET], scores, threshold, selection["model_name"])
    print("\nRAW OBSERVATION METRICS — FINAL TEST (calibrated probability)")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print("\nOPERATIONAL ALERTS — FINAL TEST")
    for key in ["total_operational_alerts", "true_operational_alerts", "false_operational_alerts",
                "operational_alert_precision", "false_operational_alerts_per_server_day"]:
        print(f"{key}: {operational[key]}")
    print_event_summary(operational)
    return metrics, operational


def save_final_plots(y_true, scores, threshold: float, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, np.asarray(scores) >= threshold, labels=[0, 1],
        display_labels=["No failure", "Failure"], ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set_title(f"FINAL TEST: {model_name}\nDecision threshold {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    for kind in ["roc", "precision_recall"]:
        fig, ax = plt.subplots(figsize=(7, 5))
        if kind == "roc":
            x, y, _ = roc_curve(y_true, scores)
            ax.plot(x, y, label=f"{model_name} (AUC {roc_auc_score(y_true, scores):.4f})")
            ax.plot([0, 1], [0, 1], "k--", label="Chance")
            ax.set(xlabel="False positive rate", ylabel="True positive rate", title="FINAL TEST: ROC curve")
        else:
            precision, recall, _ = precision_recall_curve(y_true, scores)
            ax.plot(recall, precision, label=f"{model_name} (AP {average_precision_score(y_true, scores):.4f})")
            ax.axhline(np.mean(y_true), color="black", linestyle="--", label="Test prevalence")
            ax.set(xlabel="Recall", ylabel="Precision", title="FINAL TEST: precision–recall curve")
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"{kind}_curve.png", dpi=150)
        plt.close(fig)


def load_evaluation_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Check the training split and feature schema before any evaluation."""
    _, validation, test, split = temporal_split(pd.read_csv(PROCESSED_PATH))
    if json.loads((REPORT_DIR / "split_metadata.json").read_text()) != split:
        raise ValueError("Temporal split differs from training. Run python -m src.train.")
    if json.loads((MODEL_DIR / "feature_columns.json").read_text()) != FEATURE_COLUMNS:
        raise ValueError("Feature schema changed. Retrain all models.")
    return validation, test, split


def main() -> tuple[dict, dict]:
    validation, test, split = load_evaluation_data()
    selection = select_on_validation(validation)
    metrics, operational = evaluate_final_test(test, selection, split)
    return metrics, operational


if __name__ == "__main__":
    main()
