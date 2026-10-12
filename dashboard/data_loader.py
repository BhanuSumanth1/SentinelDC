"""Cached reads of frozen artifacts only. No model deserialization or inference."""
from functools import lru_cache
from pathlib import Path
import copy
import json
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from dashboard.formatting import RISK_ORDER

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports'
KEYS = ['server_id', 'timestamp']


@lru_cache(maxsize=40)
def _csv(path, modified):
    return pd.read_csv(path)


def csv(name):
    path = REPORTS/name
    return _csv(str(path), path.stat().st_mtime_ns).copy(deep=True)


@lru_cache(maxsize=20)
def _json(path, modified):
    return json.loads(Path(path).read_text())


def json_file(name):
    path = ROOT/name
    return copy.deepcopy(_json(str(path), path.stat().st_mtime_ns))


@lru_cache(maxsize=5)
def _jsonl(path, modified):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def json_lines(name):
    path = REPORTS/name
    return copy.deepcopy(_jsonl(str(path), path.stat().st_mtime_ns))


def records(frame):
    # Pandas serializes NaN/NaT as null instead of invalid JSON NaN.
    return json.loads(frame.to_json(orient='records', date_format='iso'))


@lru_cache(maxsize=1)
def topology():
    tree = ET.parse(ROOT/'models/infrastructure_graph.graphml')
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
    names = {k.attrib['id']: k.attrib['attr.name'] for k in tree.findall('g:key', ns)}
    rows = []
    for node in tree.findall('.//g:node', ns):
        attrs = {names[d.attrib['key']]: d.text for d in node.findall('g:data', ns)}
        if attrs.get('node_type') == 'SERVER':
            rows.append({'server_id': attrs['component_id'], 'datacenter_id': attrs['datacenter_id']})
    result = pd.DataFrame(rows)
    if result.server_id.duplicated().any():
        raise ValueError('Conflicting infrastructure server membership.')
    return result


def snapshot():
    hybrid, explanations = csv('hybrid_live_view.csv'), csv('explainability_live_view.csv')
    for frame in [hybrid, explanations]:
        frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True).astype(str)
        if frame.duplicated(KEYS).any() or frame.server_id.duplicated().any():
            raise ValueError('Duplicate snapshot server or observation.')
    if set(map(tuple, hybrid[KEYS].values)) != set(map(tuple, explanations[KEYS].values)):
        raise ValueError('Hybrid and SHAP snapshot keys do not match.')
    check = hybrid.merge(explanations, on=KEYS, suffixes=('_hybrid', '_shap'), validate='one_to_one')
    for column in ['failure_probability', 'anomaly_score', 'anomaly_percentile']:
        if not np.allclose(check[column+'_hybrid'], check[column+'_shap'], rtol=1e-7, atol=1e-10):
            raise ValueError('Inconsistent saved snapshot scores.')
    if not check.hybrid_risk_level_hybrid.eq(check.hybrid_risk_level_shap).all():
        raise ValueError('Inconsistent saved risk levels.')
    cols = [c for c in explanations.columns if c not in hybrid.columns or c in KEYS]
    result = hybrid.merge(explanations[cols], on=KEYS, validate='one_to_one')
    result = result.merge(topology(), on='server_id', how='left', validate='one_to_one')
    if result.datacenter_id.isna().any() or not result.hybrid_risk_level.isin(RISK_ORDER).all():
        raise ValueError('Missing topology or unsupported risk level.')
    result['_rank'] = result.hybrid_risk_level.map(dict(zip(RISK_ORDER, range(4))))
    return result.sort_values(['_rank', 'failure_probability', 'server_id'], ascending=[False, False, True]).drop(columns='_rank').reset_index(drop=True)


def performance():
    rows = []
    for system, filename, rawfile in [
        ('SUPERVISED', 'operational_alert_metrics.csv', 'final_test_metrics.csv'),
        ('ANOMALY', 'anomaly_operational_metrics.csv', 'anomaly_test_metrics.csv'),
        ('HYBRID', 'hybrid_test_metrics.csv', 'hybrid_test_metrics.csv')]:
        op, raw = csv(filename).iloc[0], csv(rawfile).iloc[0]
        rows.append({'system': system, 'pr_auc': raw.get('pr_auc', raw.get('row_pr_auc')),
            'roc_auc': raw.get('roc_auc', raw.get('row_roc_auc')),
            'row_precision': raw.get('precision', raw.get('row_precision')),
            'row_recall': raw.get('recall', raw.get('row_recall')),
            'total_failure_events': int(op.total_failure_events),
            'detected_failure_events': int(op.detected_failure_events), 'missed_failure_events': int(op.missed_failure_events),
            'event_detection_rate': op.event_detection_rate, 'median_lead_time_hours': op.median_lead_time_hours,
            'total_operational_alerts': int(op.total_operational_alerts),
            'false_operational_alerts': int(op.false_operational_alerts),
            'operational_precision': op.get('operational_alert_precision', op.get('operational_precision')),
            'false_alerts_per_server_day': op.get('false_operational_alerts_per_server_day', op.get('false_alerts_per_server_day'))})
    return records(pd.DataFrame(rows))


def summary():
    view = snapshot()
    graph = json_file('reports/infrastructure_graph_summary.json')
    hybrid = next(p for p in performance() if p['system'] == 'HYBRID')
    return {'servers_monitored': len(view), 'datacenters': graph['number_of_datacenters'],
            'racks': graph['number_of_racks'], 'switches': graph['number_of_switches'],
            'snapshot_timestamp': view.timestamp.max(), 'snapshot_kind': 'historical frozen TEST snapshot',
            'test_failure_events': hybrid['total_failure_events'],
            'hybrid_detected_failure_events': hybrid['detected_failure_events'],
            'hybrid_detection_rate': hybrid['event_detection_rate'],
            'median_warning_lead_time_hours': hybrid['median_lead_time_hours'],
            'risk_counts': {risk: int(view.hybrid_risk_level.eq(risk).sum()) for risk in RISK_ORDER}}


def server_detail(server_id):
    row = snapshot().loc[lambda x: x.server_id.eq(server_id)]
    if row.empty:
        raise KeyError(server_id)
    result = records(row)[0]
    result['supervised_threshold'] = json_file('models/decision_threshold.json')['selected_threshold']
    result['reason_codes'] = str(result.get('reason_code', '')).split('|')
    result['top_explanation_drivers'] = [
        {'feature': result.get('top_risk_driver'), 'value': result.get('top_risk_driver_value'), 'shap_value': result.get('top_risk_driver_shap')},
        {'feature': result.get('second_risk_driver'), 'value': None, 'shap_value': result.get('second_risk_driver_shap')}]
    result['explanation_note'] = 'Saved snapshot top two SHAP drivers; second driver value was not saved. Raw model log-odds contributions, not causal effects.'
    return result


TELEMETRY = ['temperature_c', 'cpu_utilization', 'memory_utilization', 'packet_loss_pct',
             'network_latency_ms', 'disk_utilization', 'power_watts']


@lru_cache(maxsize=1)
def _telemetry():
    frame = pd.read_csv(ROOT/'data/processed/server_telemetry_processed.csv', usecols=[*KEYS, *TELEMETRY])
    frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True)
    return frame


def server_telemetry(server_id):
    frame = _telemetry()
    result = frame[frame.server_id.eq(server_id)].copy().sort_values('timestamp')
    alerts = csv('hybrid_test_scores.csv')
    alerts = alerts[alerts.server_id.eq(server_id)][KEYS+['operational_alert']]
    alerts['timestamp'] = pd.to_datetime(alerts.timestamp, utc=True)
    return result.merge(alerts, on=KEYS, how='left', validate='one_to_one')
