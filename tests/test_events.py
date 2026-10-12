"""Incident identity, metadata leakage, lead-time boundaries, and CSV artifacts."""
import numpy as np
import pandas as pd
import pytest

from src.config import FAILURE_METADATA_COLUMNS, FEATURE_COLUMNS, TARGET
from src.data_validation import validate_failure_metadata
from src.event_evaluation import analyze_failure_events, save_event_evaluation
from src.evaluate import operational_metrics
from src.generate_data import generate_telemetry
from src.preprocess import preprocess_data


@pytest.fixture(scope="module")
def generated():
    return generate_telemetry(40, 30, 42)


@pytest.fixture
def event_rows():
    # All warning rows belong to two distinct incidents, with one healthy row.
    failure = pd.Timestamp("2025-01-03T00:00:00Z")
    records = []
    for event_id, server, kind, leads in [
        ("event_a", "server_a", "disk_failure", [24, 12, 6, 1]),
        ("event_b", "server_b", "overheating", [12, 6]),
    ]:
        for lead in leads:
            records.append({"timestamp": failure - pd.Timedelta(hours=lead), "server_id": server,
                            "failure_event_id": event_id, "actual_failure_timestamp": failure,
                            "failure_type": kind, TARGET: 1})
    records.append({"timestamp": failure, "server_id": "server_c", "failure_event_id": None,
                    "actual_failure_timestamp": pd.NaT, "failure_type": "none", TARGET: 0})
    frame = pd.DataFrame(records)
    for column in ["temperature_c", "smart_error_count", "packet_loss_pct", "network_latency_ms"]:
        frame[column] = np.arange(len(frame)) + 1.0
    return frame


def test_generator_uses_one_id_per_actual_incident(generated):
    positive = generated[generated[TARGET].eq(1)]
    counts = positive.groupby("failure_event_id").size()
    assert counts.max() == 24
    assert (counts > 1).any()
    assert len(counts) < len(positive)
    assert (positive.groupby("failure_event_id")[["server_id", "failure_type", "actual_failure_timestamp"]].nunique() == 1).all().all()
    assert (positive.actual_failure_timestamp > positive.timestamp).all()
    assert ((positive.actual_failure_timestamp - positive.timestamp) <= pd.Timedelta(hours=24)).all()
    assert generated.loc[generated[TARGET].eq(0), ["failure_event_id", "actual_failure_timestamp"]].isna().all().all()
    assert 0.01 <= generated[TARGET].mean() <= 0.03
    validate_failure_metadata(generated)


def test_metadata_cannot_enter_model_features(generated):
    assert set(FAILURE_METADATA_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    data = preprocess_data(generated)
    assert set(FAILURE_METADATA_COLUMNS).issubset(data.columns)
    assert set(FAILURE_METADATA_COLUMNS).isdisjoint(data[FEATURE_COLUMNS].columns)


def test_known_lead_times_and_missed_events(event_rows):
    # First event detected at 12h, second completely missed; healthy alert is not an incident.
    events, summary = analyze_failure_events(event_rows, [0.1, 0.9, 0.95, 0.95, 0.1, 0.1, 0.99], 0.9)
    detected, missed = events.iloc[0], events.iloc[1]
    assert bool(detected.detected)
    assert detected.lead_time_hours == 12
    assert detected.first_alert_timestamp == pd.Timestamp("2025-01-02T12:00:00Z")
    assert detected.max_model_risk_score_before_failure == 0.95
    assert not bool(missed.detected)
    assert pd.isna(missed.first_alert_timestamp)
    assert pd.isna(missed.lead_time_hours)
    assert events.lead_time_hours.dropna().ge(0).all()
    assert summary["total_failure_events"] == 2
    assert summary["detected_failure_events"] == 1
    assert summary["missed_failure_events"] == 1
    assert summary["event_detection_rate"] == 0.5
    assert summary["median_lead_time_hours"] == 12
    assert summary["detected_ge_12h_pct"] == 50  # All events, including misses.
    assert summary["detected_ge_24h_pct"] == 0


def test_exact_24h_boundary_and_test_only_alerts(event_rows):
    full, full_summary = analyze_failure_events(event_rows, [0.9] * len(event_rows), 0.9)
    assert full_summary["detected_ge_24h_count"] == 1
    assert full.lead_time_hours.max() == 24
    # Remove the earliest observation as if it occurred before the test boundary.
    partial, _ = analyze_failure_events(event_rows.iloc[1:].copy(), [0.9] * (len(event_rows) - 1), 0.9)
    assert partial.loc[partial.failure_event_id.eq("event_a"), "lead_time_hours"].iloc[0] == 12
    assert not partial.complete_24h_warning_window.any()


def test_summary_quantiles_and_no_negative_leads(event_rows):
    _, summary = analyze_failure_events(event_rows, [0.9, 0.9, 0.9, 0.9, 0.1, 0.9, 0.1], 0.9)
    assert summary["mean_lead_time_hours"] == 15
    assert summary["median_lead_time_hours"] == 15
    assert summary["min_lead_time_hours"] == 6
    assert summary["max_lead_time_hours"] == 24
    assert summary["p25_lead_time_hours"] == 10.5
    assert summary["p75_lead_time_hours"] == 19.5


def test_no_detected_events_has_undefined_lead_time(event_rows):
    events, summary = analyze_failure_events(event_rows, np.zeros(len(event_rows)), 0.9)
    assert not events.detected.any()
    assert events.first_alert_timestamp.isna().all()
    assert events.lead_time_hours.isna().all()
    assert np.isnan(summary["mean_lead_time_hours"])
    assert summary["detected_ge_6h_pct"] == 0


def test_no_actual_events_has_undefined_detection_rate(event_rows, tmp_path):
    healthy = event_rows[event_rows[TARGET].eq(0)]
    summary = save_event_evaluation(healthy, [0.99], 0.9, tmp_path, tmp_path / "figures")
    assert summary["total_failure_events"] == 0
    assert np.isnan(summary["event_detection_rate"])
    assert pd.read_csv(tmp_path / "failure_event_analysis.csv").empty


def test_event_csv_and_plots_created(event_rows, tmp_path):
    save_event_evaluation(event_rows, [0.9] * len(event_rows), 0.9, tmp_path, tmp_path / "figures")
    events = pd.read_csv(tmp_path / "failure_event_analysis.csv")
    summary = pd.read_csv(tmp_path / "failure_lead_time_summary.csv")
    assert len(events) == 2
    assert events.failure_event_id.is_unique
    assert summary.total_failure_events.iloc[0] == 2
    for name in ["failure_lead_time_distribution", "detection_by_failure_type", "example_failure_timeline"]:
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 0


@pytest.mark.parametrize("case", ["at_failure", "after_failure", "too_early", "different_server", "two_ids", "missing_id"])
def test_rejects_inconsistent_incidents(event_rows, case):
    bad = event_rows.copy()
    if case == "at_failure":
        bad.loc[0, "timestamp"] = bad.loc[0, "actual_failure_timestamp"]
    elif case == "after_failure":
        bad.loc[0, "timestamp"] = bad.loc[0, "actual_failure_timestamp"] + pd.Timedelta(hours=1)
    elif case == "too_early":
        bad.loc[0, "timestamp"] = bad.loc[0, "actual_failure_timestamp"] - pd.Timedelta(hours=25)
    elif case == "different_server":
        bad.loc[0, "server_id"] = "wrong_server"
    elif case == "two_ids":
        bad.loc[0, "failure_event_id"] = "duplicate_incident"
    else:
        bad.loc[0, "failure_event_id"] = None
    with pytest.raises(ValueError):
        validate_failure_metadata(bad)


def test_healthy_false_alert_denominator():
    result = operational_metrics({"true_positive": 4, "false_negative": 1,
                                  "true_negative": 90, "false_positive": 5, "recall": 0.8}, 100)
    assert result["false_alerts_per_1000_observations"] == 50
    assert result["false_alerts_per_1000_healthy_observations"] == pytest.approx(1000 * 5 / 95)


def test_timeline_prefers_available_history_not_best_lead_time():
    from src.event_evaluation import select_timeline_event
    failure = pd.Timestamp("2025-01-04T00:00:00Z")
    history = pd.DataFrame({"timestamp": pd.date_range(failure - pd.Timedelta(hours=48), periods=48, freq="h"),
                            "server_id": "server_full"})
    detected = pd.DataFrame([
        {"failure_event_id": "partial", "server_id": "server_partial", "actual_failure_timestamp": failure - pd.Timedelta(hours=6), "lead_time_hours": 24},
        {"failure_event_id": "full", "server_id": "server_full", "actual_failure_timestamp": failure, "lead_time_hours": 6},
    ])
    assert select_timeline_event(history, detected).failure_event_id == "full"
    assert select_timeline_event(history.iloc[:1], detected).failure_event_id == "partial"
