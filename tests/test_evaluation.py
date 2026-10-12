"""Regression tests for validation-only decisions and final holdout isolation."""
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from src import evaluate, train
from src.config import FEATURE_COLUMNS, TARGET
from src.generate_data import generate_telemetry
from src.preprocess import preprocess_data


class RecordingModel:
    """A deterministic test double that records which observations it sees."""
    def __init__(self):
        self.seen = []
        self.fitted = []
        self.named_steps = {"classifier": SimpleNamespace(feature_importances_=np.ones(len(FEATURE_COLUMNS)) / len(FEATURE_COLUMNS))}

    def fit(self, features, labels):
        self.fitted.append(features.index.tolist())
        return self

    def predict_proba(self, features):
        self.seen.append(features.index.tolist())
        scores = np.clip(features.cpu_utilization.to_numpy() / 100, 0.01, 0.99)
        return np.column_stack([1 - scores, scores])


@pytest.fixture(scope="module")
def data():
    frame = preprocess_data(generate_telemetry(200, 30, 42))
    # Controlled scores for orchestration tests, not benchmark performance.
    frame["cpu_utilization"] = 10 + 80 * frame[TARGET]
    return frame


@pytest.fixture
def artifact_dirs(tmp_path, monkeypatch):
    models, reports, figures = tmp_path / "models", tmp_path / "reports", tmp_path / "reports/figures"
    for directory in [models, reports, figures]:
        directory.mkdir(parents=True, exist_ok=True)
    for module in [evaluate, train]:
        monkeypatch.setattr(module, "MODEL_DIR", models)
        monkeypatch.setattr(module, "REPORT_DIR", reports)
    monkeypatch.setattr(evaluate, "FIGURE_DIR", figures)
    return models, reports


def test_training_fits_only_training_rows(data, artifact_dirs, tmp_path, monkeypatch):
    source = tmp_path / "processed.csv"
    data.to_csv(source, index=False)
    monkeypatch.setattr(train, "PROCESSED_PATH", source)
    models = {name: RecordingModel() for name in train.MODEL_NAMES}
    passed_labels = []
    def build_from_train(labels):
        passed_labels.append(labels.copy())
        return models
    monkeypatch.setattr(train, "build_models", build_from_train)
    expected, validation, test, _ = train.temporal_split(pd.read_csv(source))
    train.main()
    pd.testing.assert_series_equal(passed_labels[0], expected[TARGET])
    for model in models.values():
        assert model.fitted == [expected.index.tolist()]
        assert set(model.fitted[0]).isdisjoint(validation.index)
        assert set(model.fitted[0]).isdisjoint(test.index)


def test_selection_ignores_test_and_final_scores_once(data, artifact_dirs, monkeypatch):
    model_dir, reports = artifact_dirs
    _, validation, test, split = train.temporal_split(data)
    originals = {name: RecordingModel() for name in train.MODEL_NAMES}
    for name, model in originals.items():
        joblib.dump(model, model_dir / f"{name}.joblib")
    calibration_fits = []
    original_fit = evaluate.CalibratedFailureModel.fit_calibration
    def record_calibration(self, scores, labels):
        calibration_fits.append((np.asarray(scores).copy(), labels.copy()))
        return original_fit(self, scores, labels)
    monkeypatch.setattr(evaluate.CalibratedFailureModel, "fit_calibration", record_calibration)
    selection = evaluate.select_on_validation(validation)
    calibration_window, policy_window = evaluate.calibration_windows(validation.reset_index(drop=True))
    pd.testing.assert_series_equal(calibration_fits[0][1], calibration_window[TARGET])
    np.testing.assert_allclose(calibration_fits[0][0], calibration_window.cpu_utilization / 100)
    assert selection["calibration_window"]["end"] < selection["policy_validation_window"]["start"]
    assert (model_dir / "decision_threshold.json").exists()
    assert 0 < selection["selected_threshold"] < 1
    assert selection["selection_split"] == "validation"
    first_analysis = pd.read_csv(reports / "threshold_analysis.csv")
    changed = data.copy()
    changed.loc[test.index, TARGET] = 1 - changed.loc[test.index, TARGET]
    changed.loc[test.index, "cpu_utilization"] = 100
    _, unchanged_validation, _, _ = train.temporal_split(changed)
    repeated = evaluate.select_on_validation(unchanged_validation)
    assert repeated == selection
    pd.testing.assert_frame_equal(first_analysis, pd.read_csv(reports / "threshold_analysis.csv"))
    frozen_model = joblib.load(model_dir / "best_failure_model.joblib")
    # The selected model has seen validation only; nothing has called fit.
    assert frozen_model.base_model.seen == []
    assert frozen_model.base_model.fitted == []
    loaded_paths = []

    def load_selected(path):
        loaded_paths.append(path.name)
        return frozen_model

    monkeypatch.setattr(evaluate.joblib, "load", load_selected)
    monkeypatch.setattr(evaluate, "select_best_model", lambda *_: pytest.fail("Final test attempted model selection"))
    monkeypatch.setattr(evaluate, "analyze_thresholds", lambda *_: pytest.fail("Final test attempted threshold tuning"))
    monkeypatch.setattr(evaluate.CalibratedFailureModel, "fit_calibration", lambda *_: pytest.fail("TEST attempted calibration fitting"))
    final, operational = evaluate.evaluate_final_test(test, selection, split)
    assert loaded_paths == ["best_failure_model.joblib"]
    assert frozen_model.base_model.seen == [test.index.tolist()]
    assert final["threshold"] == selection["selected_threshold"]
    assert operational["detected_failure_events"] + operational["missed_failure_events"] == operational["total_failure_events"]
    assert operational["true_operational_alerts"] + operational["false_operational_alerts"] == operational["total_operational_alerts"]
    assert (reports / "operational_alert_metrics.csv").exists()
    metadata = json.loads((model_dir / "model_metadata.json").read_text())
    assert metadata["model_name"] == selection["model_name"]
    assert metadata["test_pr_auc"] == final["pr_auc"]
    assert metadata["feature_count"] == len(FEATURE_COLUMNS)
    assert len(pd.read_csv(reports / "final_test_metrics.csv")) == 1


def test_threshold_rule_and_fallback():
    labels = np.array([1, 1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.4, 0.1])
    analysis, chosen = evaluate.analyze_thresholds(labels, scores)
    expected = analysis[(analysis.recall >= 0.80) & (analysis.precision >= 0.20)].sort_values(
        ["f2", "precision", "recall", "threshold"], ascending=[False, False, False, False]).iloc[0]
    assert chosen == expected.threshold
    assert analysis.selected.sum() == 1
    scores = np.array([0.001, 0.8, 0.7, 0.3, 0.4, 0.1])
    fallback, chosen = evaluate.analyze_thresholds(labels, scores)
    assert not fallback.preferred_constraints_met.any()
    assert fallback.loc[fallback.selected, "f2"].iloc[0] == fallback.f2.max()


def test_model_tie_requires_reasonable_precision():
    metrics = pd.DataFrame([
        {"model": "precise", "pr_auc": 0.8, "recall": 0.90, "precision": 0.75, "f1": 0.8},
        {"model": "noisy", "pr_auc": 0.801, "recall": 0.99, "precision": 0.10, "f1": 0.18},
    ])
    assert evaluate.select_best_model(metrics) == "noisy"


def test_no_overlap_with_irregular_row_counts(data):
    uneven = data[~((data.timestamp.dt.hour == 2) & data.server_id.eq("server_0000"))]
    first, middle, last, _ = train.temporal_split(uneven)
    assert first.timestamp.max() < middle.timestamp.min() < last.timestamp.min()
    assert middle.timestamp.max() < last.timestamp.min()
    assert set(first.timestamp).isdisjoint(middle.timestamp)
    assert set(middle.timestamp).isdisjoint(last.timestamp)
    assert set(first.timestamp).isdisjoint(last.timestamp)
    assert last.timestamp.nunique() == 108
