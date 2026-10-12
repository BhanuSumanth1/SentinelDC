"""Exercise actual evaluation mapping, including UNKNOWN as incorrect."""
import pandas as pd
from src.root_cause_evaluation import TYPE_SIGNATURE, evaluate_signatures


def test_failure_type_mapping_in_evaluation(tmp_path):
    expected = {
        'network_failure': 'NETWORK_LIKE', 'overheating': 'THERMAL_LIKE',
        'power_failure': 'POWER_LIKE', 'disk_failure': 'DISK_LIKE',
        'memory_failure': 'MEMORY_COMPUTE_LIKE',
    }
    assert TYPE_SIGNATURE == expected
    cases = list(expected.items()) + [('network_failure', 'UNKNOWN')]
    truth, alerts, details = [], [], []
    for i, (failure_type, predicted) in enumerate(cases):
        timestamp = pd.Timestamp('2025-01-01', tz='UTC') + pd.Timedelta(hours=3*i)
        alerts.append(dict(server_id=f's{i}', timestamp=timestamp, operational_alert=True))
        truth.append(dict(server_id=f's{i}', timestamp=timestamp, datacenter_id='dc',
                          failure_event_id=f'e{i}', failure_type=failure_type,
                          actual_failure_timestamp=timestamp+pd.Timedelta(hours=1),
                          failure_within_24h=1))
        details.append(dict(incident_id=f'i{i}', window_start=str(timestamp),
                            dominant_datacenter='dc', telemetry_signature=predicted))
    result = evaluate_signatures(details, pd.DataFrame(alerts), pd.DataFrame(truth),
                                 tmp_path, tmp_path)
    table = pd.read_csv(tmp_path / 'root_cause_signature_evaluation.csv')
    assert table.true_signature.tolist() == [expected[t] for t, _ in cases]
    assert table.correct.tolist() == [True]*5 + [False]
    assert result['signature_classification_accuracy'] == 5/6
    assert result['evaluated_unique_failure_events'] == 6
