"""Independent guardrails for causal calibration and operational incident alerts."""
import json
import numpy as np
import pandas as pd
import pytest
from src import evaluate, train
from src.calibration import CalibratedFailureModel, calibration_windows
from src.alert_policy import AlertPolicy, apply_alert_policy, evaluate_operational_alerts
from src.config import FEATURE_COLUMNS, TARGET


def observations(hours, server="a"):
    return pd.DataFrame({"server_id": server, "timestamp": pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(hours, unit="h")})


def test_persistence_is_causal_and_sorted():
    frame = observations([0, 1, 2, 3, 4])
    scores = np.array([.7, .7, .1, .7, .7])
    policy = AlertPolicy(.6, .95, cooldown_hours=0)
    result = apply_alert_policy(frame, scores, policy)
    assert result.operational_alert.tolist() == [False, True, False, False, True]
    order = [4, 1, 3, 0, 2]
    shuffled = apply_alert_policy(frame.iloc[order], scores[order], policy).sort_values("timestamp")
    assert shuffled.operational_alert.tolist() == result.operational_alert.tolist()
    for end in range(1, len(frame) + 1):
        prefix = apply_alert_policy(frame.iloc[:end], scores[:end], policy)
        assert prefix.operational_alert.tolist() == result.operational_alert.iloc[:end].tolist()


def test_missing_hours_and_different_servers_do_not_supply_persistence():
    frame = pd.concat([observations([0, 2, 3]), observations([0], "b")], ignore_index=True)
    result = apply_alert_policy(frame, [.7] * 4, AlertPolicy(.6, .95))
    assert result.operational_alert.tolist() == [False, False, True, False]


def test_emergency_and_cooldown_boundary():
    frame = observations(range(9))
    result = apply_alert_policy(frame, [.99] * 9, AlertPolicy(.6, .95, cooldown_hours=6))
    assert result.loc[result.operational_alert, "observation_position"].tolist() == [0, 6]
    assert result.alert_reason.iloc[1] == "cooldown"
    assert result.alert_reason.iloc[6] == "emergency"


@pytest.mark.parametrize("threshold,emergency,cooldown", [(0,.9,6), (1,.99,6), (.6,.5,6), (.6,.9,-1)])
def test_policy_rejects_invalid_parameters(threshold, emergency, cooldown):
    with pytest.raises(ValueError):
        AlertPolicy(threshold, emergency, cooldown_hours=cooldown)


def test_one_event_and_repeated_true_alert_workload(tmp_path):
    frame = observations(range(28))
    failure = frame.timestamp.iloc[24]
    frame[TARGET] = (frame.timestamp < failure).astype(int)
    frame["failure_event_id"] = np.where(frame[TARGET].eq(1), "event_a", None)
    frame["actual_failure_timestamp"] = pd.NaT
    frame["actual_failure_timestamp"] = pd.to_datetime(frame.actual_failure_timestamp, utc=True)
    frame.loc[frame[TARGET].eq(1), "actual_failure_timestamp"] = failure
    frame["failure_type"] = np.where(frame[TARGET].eq(1), "disk_failure", "none")
    for column in ["temperature_c", "smart_error_count", "packet_loss_pct", "network_latency_ms"]:
        frame[column] = 1.0
    scored = apply_alert_policy(frame, [.99] * len(frame), AlertPolicy(.6, .95))
    figures = tmp_path / "figures"; figures.mkdir()
    metrics = evaluate_operational_alerts(frame, scored, tmp_path, figures)
    assert metrics["total_operational_alerts"] == 5
    assert metrics["true_operational_alerts"] == 4
    assert metrics["false_operational_alerts"] == 1  # At failure cannot warn in advance.
    assert metrics["detected_failure_events"] == 1
    assert metrics["redundant_true_operational_alerts"] == 3
    assert metrics["median_lead_time_hours"] == 24
    assert metrics["false_operational_alerts_per_server_day"] == pytest.approx(24 / 28)
    assert metrics["false_operational_alerts_per_1000_healthy_observations"] == 250
    events = pd.read_csv(tmp_path / "failure_event_analysis.csv")
    assert events.failure_event_id.is_unique
    assert (tmp_path / "operational_alert_metrics.csv").exists()


def test_calibration_is_bounded_and_never_fits_base_model():
    class Frozen:
        def fit(self, *args):
            pytest.fail("Calibration refitted the base estimator")
    scores = np.tile([0.01, 0.1, 0.5, 0.9, 0.99], 30)
    labels = np.tile([0, 0, 0, 1, 1], 30)
    model = CalibratedFailureModel(Frozen()).fit_calibration(scores, labels)
    result = model.calibrate([0, .001, .4, .99, 1])
    assert ((0 <= result) & (result <= 1)).all()
    assert (np.diff(result) > 0).all()


def test_emergency_threshold_uses_validation_distribution():
    assert evaluate.select_emergency_threshold(np.linspace(.01, .04, 100), .03) == .5
    assert evaluate.select_emergency_threshold(np.linspace(.01, .98, 100), .4) > .97
    assert .99 < evaluate.select_emergency_threshold([.01, .5], .99) < 1


def test_train_only_weight_with_arbitrary_other_prevalences():
    labels = pd.Series([0] * 98 + [1] * 2)
    for name in ["xgboost", "lightgbm"]:
        model = train.build_models(labels)[name]
        assert model.named_steps["classifier"].scale_pos_weight == 49
    assert train.training_class_weight(labels) == 49
    with pytest.raises(ValueError):
        train.training_class_weight([0, 0])


def test_no_leakage_features_in_any_model_schema():
    forbidden = {TARGET, "failure_type", "failure_event_id", "actual_failure_timestamp", "server_id", "rack_id", "switch_id", "datacenter_id"}
    assert forbidden.isdisjoint(FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == 32


def test_calibration_policy_windows_purge_future_labels():
    frame = observations(range(84))
    frame[TARGET] = np.arange(84) % 2
    calibration, policy = calibration_windows(frame)
    assert calibration.timestamp.max() + pd.Timedelta(hours=24) < policy.timestamp.min()
    assert set(calibration.index).isdisjoint(policy.index)
    assert len(calibration) == 28 and len(policy) == 32


def test_f2_search_counts_match_independent_confusion_counts():
    labels = np.array([0] * 90 + [1] * 10)
    scores = np.linspace(0, 1, 100)
    analysis, chosen = evaluate.analyze_thresholds(labels, scores)
    assert len(analysis) == 981 and 0 < chosen < 1
    for index in [0, 500, 980]:
        row = analysis.iloc[index]
        predicted = scores >= row.threshold
        tp = (predicted & (labels == 1)).sum()
        fp = (predicted & (labels == 0)).sum()
        fn = (~predicted & (labels == 1)).sum()
        assert row.f2 == pytest.approx(5 * tp / (5 * tp + fp + 4 * fn))
        assert row.false_alerts_per_1000_healthy_observations == pytest.approx(1000 * fp / 90)


def test_scientific_audit_reuses_cached_scores_and_checks_keys(tmp_path, monkeypatch):
    from src import scientific_audit
    monkeypatch.setattr(scientific_audit, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(scientific_audit.joblib, "load", lambda *_: pytest.fail("Audit must not rescore TEST"))
    frame = observations([0, 1, 2])
    cached = frame.assign(calibrated_failure_probability=[.01, .2, .3], operational_alert=[False, False, True])
    cached.to_csv(tmp_path / 'test_predictions_and_alerts.csv', index=False)
    probabilities, flags = scientific_audit.load_audit_predictions(frame)
    np.testing.assert_allclose(probabilities, [.01, .2, .3])
    assert flags.tolist() == [False, False, True]
    with pytest.raises(ValueError, match="keys/order"):
        scientific_audit.load_audit_predictions(frame.iloc[::-1])
