"""Fast generator, validation, feature, and temporal leakage regression tests."""
import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, IDENTIFIERS, NUMERIC_COLUMNS, PERCENT_COLUMNS, REQUIRED_COLUMNS, TARGET
from src.data_validation import validate_data
from src.generate_data import generate_telemetry
from src.preprocess import engineer_features, preprocess_data
from src.train import temporal_split


@pytest.fixture(scope="module")
def telemetry() -> pd.DataFrame:
    return generate_telemetry(n_servers=40, days=30, seed=42)


def test_dataset_generation(telemetry):
    assert len(telemetry) == 40 * 30 * 24
    assert telemetry.server_id.nunique() == 40
    assert telemetry.timestamp.nunique() == 720
    assert not telemetry.duplicated(["server_id", "timestamp"]).any()


def test_required_columns(telemetry):
    assert list(telemetry.columns) == REQUIRED_COLUMNS
    assert not telemetry.drop(columns=["failure_event_id", "actual_failure_timestamp"]).isna().any().any()


def test_failure_labels_and_rarity(telemetry):
    assert set(telemetry[TARGET].unique()) == {0, 1}
    assert telemetry[TARGET].sum() > 0
    assert 0.01 <= telemetry[TARGET].mean() <= 0.03
    assert (telemetry.failure_type.ne("none") == telemetry[TARGET].eq(1)).all()


def test_percentage_ranges(telemetry):
    assert telemetry[PERCENT_COLUMNS].ge(0).all().all()
    assert telemetry[PERCENT_COLUMNS].le(100).all().all()
    validate_data(telemetry, verbose=False)


def test_reproducible_generation():
    pd.testing.assert_frame_equal(generate_telemetry(10, 7, 19), generate_telemetry(10, 7, 19))


def test_labels_precede_repair_events(telemetry):
    # An observed repair resets uptime. Its preceding 24 hourly rows must be positive.
    events = 0
    for _, server in telemetry.groupby("server_id"):
        server = server.sort_values("timestamp")
        for event_time in server.loc[server.uptime_days.diff() < 0, "timestamp"]:
            expected = (server.timestamp < event_time) & (server.timestamp >= event_time - pd.Timedelta(hours=24))
            assert server.loc[expected, TARGET].eq(1).all()
            assert len(server.loc[expected]) == 24
            assert server.loc[server.timestamp.eq(event_time), TARGET].eq(0).all()
            events += 1
    assert events > 0


@pytest.mark.parametrize("column,value", [
    ("cpu_utilization", 101), ("packet_loss_pct", -1), ("temperature_c", 200),
    (TARGET, 2), ("timestamp", "not-a-date"), ("power_watts", np.inf),
    ("smart_error_count", 1.5),
])
def test_validation_rejects_invalid_values(telemetry, column, value):
    bad = telemetry.head(3).copy().astype({column: object})
    bad.loc[bad.index[0], column] = value
    with pytest.raises(ValueError, match="validation failed"):
        validate_data(bad, verbose=False)


def test_validation_rejects_missing_column(telemetry):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(telemetry.drop(columns="temperature_c"), verbose=False)


def test_conflicting_duplicates(telemetry):
    bad = pd.concat([telemetry.head(1), telemetry.head(1)], ignore_index=True)
    bad.loc[1, "cpu_utilization"] += 1
    with pytest.raises(ValueError, match="conflicting"):
        validate_data(bad, verbose=False)


def test_preprocessing_is_causal(telemetry):
    server = telemetry[telemetry.server_id.eq("server_0000")].head(5).copy()
    server["cpu_utilization"] = [np.nan, 20, np.nan, np.nan, 90]
    data = pd.concat([server, server.iloc[[1]]], ignore_index=True)
    processed = preprocess_data(data)
    assert len(processed) == 5
    assert np.isnan(processed.cpu_utilization.iloc[0])  # Never borrow from the future.
    assert processed.cpu_utilization.iloc[2] == 20
    assert processed.cpu_utilization.iloc[3] == 20
    # A long timestamp gap cannot inherit a stale reading.
    server.loc[server.index[2], "timestamp"] += pd.Timedelta(days=10)
    stale = preprocess_data(server)
    assert np.isnan(stale.cpu_utilization.iloc[-1])


def test_feature_schema_and_time_split(telemetry):
    processed = preprocess_data(telemetry)
    assert set(FEATURE_COLUMNS).issubset(processed.columns)
    assert set(FEATURE_COLUMNS).isdisjoint([TARGET, "failure_type", "failure_event_id", "actual_failure_timestamp", "timestamp", *IDENTIFIERS])
    train, validation, test, metadata = temporal_split(processed)
    assert train.timestamp.max() + pd.Timedelta(hours=24) < test.timestamp.min()
    assert set(train.timestamp).isdisjoint(test.timestamp)
    assert train.timestamp.max() + pd.Timedelta(hours=24) < validation.timestamp.min()
    assert validation.timestamp.max() + pd.Timedelta(hours=24) < test.timestamp.min()
    assert set(train.timestamp).isdisjoint(validation.timestamp)
    assert set(validation.timestamp).isdisjoint(test.timestamp)
    assert test.timestamp.nunique() == 108
    assert validation.timestamp.nunique() == 84
    assert train.timestamp.nunique() == 480
    assert metadata["purged_rows"] == 48 * 40


def test_inference_feature_parity(telemetry):
    raw = telemetry.iloc[[20]]
    batch = preprocess_data(raw)[FEATURE_COLUMNS].reset_index(drop=True)
    single = engineer_features(raw[["timestamp", *NUMERIC_COLUMNS]])[FEATURE_COLUMNS].reset_index(drop=True)
    pd.testing.assert_frame_equal(batch, single)
