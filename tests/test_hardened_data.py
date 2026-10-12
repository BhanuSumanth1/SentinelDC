"""Causality, server isolation, history parity, and simulator mechanism checks."""
import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, FAILURE_METADATA_COLUMNS, NUMERIC_COLUMNS, TARGET, TREND_COLUMNS
from src.generate_data import generate_telemetry
from src.preprocess import engineer_features, preprocess_data
from src.predict import EXAMPLE_RECORD, prepare_model_inputs
from src.train import temporal_split


@pytest.fixture(scope="module")
def hardened():
    return generate_telemetry(40, 30, 42)


@pytest.fixture
def history_rows():
    frame = pd.DataFrame([{**EXAMPLE_RECORD, 'server_id': 'server_a',
                           'timestamp': pd.Timestamp('2025-01-01', tz='UTC') + pd.Timedelta(hours=i)} for i in range(15)])
    frame['temperature_c'] = np.arange(15) + 20.0
    frame['smart_error_count'] = np.arange(15)
    frame['power_watts'] = np.arange(15) * 10 + 200
    return frame


def test_rolling_current_past_only(history_rows):
    original = engineer_features(history_rows)
    prefix = engineer_features(history_rows.iloc[:8])
    pd.testing.assert_frame_equal(original.iloc[:8][TREND_COLUMNS], prefix[TREND_COLUMNS])
    changed = history_rows.copy()
    changed.loc[8:, NUMERIC_COLUMNS] = 0
    pd.testing.assert_frame_equal(original.iloc[:8][TREND_COLUMNS], engineer_features(changed).iloc[:8][TREND_COLUMNS])
    assert original.temperature_rolling_mean_6h.iloc[6] == 23.5
    assert original.temperature_change_6h.iloc[6] == 6
    assert original.smart_error_change_12h.iloc[12] == 12
    assert original.power_change_6h.iloc[6] == 60


def test_rolling_isolates_servers(history_rows):
    other = history_rows.copy()
    other['server_id'] = 'server_b'
    other['temperature_c'] = 100
    combined = pd.concat([history_rows, other], ignore_index=True).sort_values('timestamp', kind='stable')
    features = engineer_features(combined)
    alone = engineer_features(history_rows)
    pd.testing.assert_frame_equal(features[features.server_id.eq('server_a')][TREND_COLUMNS].reset_index(drop=True),
                                  alone[TREND_COLUMNS].reset_index(drop=True))
    assert features.loc[features.server_id.eq('server_b'), 'temperature_rolling_mean_6h'].eq(100).all()


def test_lags_use_clock_time_not_six_rows(history_rows):
    sparse = history_rows.drop(index=0)
    result = engineer_features(sparse)
    assert np.isnan(result.loc[6, 'temperature_change_6h'])
    assert result.loc[7, 'temperature_change_6h'] == 6


def test_history_prediction_matches_batch_features(history_rows):
    batch = engineer_features(history_rows)
    current = history_rows.iloc[[-1]][['timestamp', *NUMERIC_COLUMNS]]
    inferred, count = prepare_model_inputs(current, 'server_a', history_rows.iloc[:-1])
    assert count == 14
    pd.testing.assert_frame_equal(inferred[FEATURE_COLUMNS], batch.iloc[[-1]][FEATURE_COLUMNS].reset_index(drop=True))
    with pytest.raises(ValueError, match='strictly before'):
        prepare_model_inputs(current, 'server_a', history_rows)


def test_server_baselines_and_shared_conditions_vary(hardened):
    profiles = pd.DataFrame(hardened.attrs['server_profiles'])
    assert profiles.normal_temperature.max() - profiles.normal_temperature.min() > 15
    assert profiles.normal_network_latency.nunique() > 10
    shared = pd.DataFrame(hardened.attrs['shared_conditions'])
    assert set(shared.kind) == {'rack_heat', 'switch_congestion'}
    assert (shared.duration_hours >= 2).all()


def test_healthy_abnormal_episodes_exist(hardened):
    healthy = hardened[hardened[TARGET].eq(0)]
    assert healthy.cpu_utilization.ge(90).any()
    assert healthy.temperature_c.ge(80).any()
    assert healthy.packet_loss_pct.ge(3).any()
    assert healthy.smart_error_count.ge(5).any()
    transient = pd.DataFrame(hardened.attrs['healthy_transients'])
    assert transient.fully_healthy_24h.any()
    assert set(transient.kind) == {'cpu', 'memory', 'thermal', 'network', 'disk', 'smart', 'power', 'fan'}
    assert (transient.end_timestamp >= transient.start_timestamp).all()


def test_failure_onsets_and_trajectories_vary(hardened):
    catalog = pd.DataFrame(hardened.attrs['failure_event_catalog'])
    assert catalog.degradation_onset_hours.nunique() > 4
    assert catalog.degradation_onset_hours.min() < 24 < catalog.degradation_onset_hours.max()
    assert catalog.slope_exponent.nunique() == len(catalog)
    assert catalog.temporary_recovery.any()
    assert (~catalog.temporary_recovery).any()
    positive = hardened[hardened[TARGET].eq(1)]
    traces = [tuple(frame.sort_values('timestamp').temperature_c) for _, frame in positive.groupby('failure_event_id') if len(frame) == 24]
    assert len(traces) == len(set(traces))
    assert not catalog.loc[catalog.failure_type.eq('disk_failure'), 'affected_signals'].str.contains('packet_loss|network_latency').any()


def test_overlap_rarity_and_metadata_isolation(hardened):
    assert 0.01 <= hardened[TARGET].mean() <= 0.03
    healthy, failing = hardened[hardened[TARGET].eq(0)], hardened[hardened[TARGET].eq(1)]
    for feature in ['temperature_c', 'packet_loss_pct', 'cpu_utilization', 'memory_utilization']:
        assert max(healthy[feature].quantile(.1), failing[feature].quantile(.1)) < min(healthy[feature].quantile(.9), failing[feature].quantile(.9))
    assert set(FAILURE_METADATA_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert {'rack_id', 'switch_id', 'server_id', 'datacenter_id', 'degradation_onset_hours', 'severity'}.isdisjoint(FEATURE_COLUMNS)
    processed = preprocess_data(hardened)
    training, validation, test, _ = temporal_split(processed)
    assert training.timestamp.max() + pd.Timedelta(hours=24) < validation.timestamp.min()
    assert validation.timestamp.max() + pd.Timedelta(hours=24) < test.timestamp.min()
