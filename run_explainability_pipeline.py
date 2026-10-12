"""Explain frozen predictions only; no model, policy or root-cause fitting."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/sentineldc-explainability-mpl')
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from src.config import ROOT, REPORT_DIR, FIGURE_DIR, MODEL_DIR, PROCESSED_PATH
from src.explainability import (FrozenTreeExplainer, DISPLAY_NAMES, DISCLAIMER, verify_hashes,
                                local_record, save_global, save_live_view)

KEYS = ['server_id', 'timestamp']


def keyed(frame):
    frame = frame.copy()
    frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True)
    if frame.duplicated(KEYS).any():
        raise ValueError('Duplicate observation keys.')
    return frame.set_index(KEYS).sort_index()


def main():
    manifest_path = REPORT_DIR/'explainability_protected_artifacts.json'
    if not manifest_path.exists():
        raise ValueError('Protected pre-explainability manifest is required.')
    protected = json.loads(manifest_path.read_text())
    verify_hashes(ROOT, protected)
    before = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in protected}
    features = json.loads((MODEL_DIR/'feature_columns.json').read_text())
    model = joblib.load(MODEL_DIR/'best_failure_model.joblib')
    explainer = FrozenTreeExplainer(model, features)
    threshold = json.loads((MODEL_DIR/'decision_threshold.json').read_text())['selected_threshold']
    split = json.loads((REPORT_DIR/'split_metadata.json').read_text())['test']
    # Outcome label used only to preserve prevalence in the explanation sample.
    data = pd.read_csv(PROCESSED_PATH, usecols=[*KEYS, *features, 'failure_within_24h'])
    data['timestamp'] = pd.to_datetime(data.timestamp, utc=True)
    test = keyed(data[data.timestamp.between(pd.Timestamp(split['start']), pd.Timestamp(split['end']))])
    del data
    hybrid = keyed(pd.read_csv(REPORT_DIR/'hybrid_test_scores.csv'))
    supervised = keyed(pd.read_csv(REPORT_DIR/'test_predictions_and_alerts.csv'))
    live = keyed(pd.read_csv(REPORT_DIR/'hybrid_live_view.csv'))
    if not test.index.equals(hybrid.index) or not test.index.equals(supervised.index):
        raise ValueError('Frozen test observation keys differ.')
    n = min(2000, len(test))
    positive = test[test.failure_within_24h.eq(1)]
    negative = test[test.failure_within_24h.eq(0)]
    n_positive = min(len(positive), max(1, round(n*len(positive)/len(test)))) if len(positive) else 0
    n_negative = min(len(negative), n-n_positive)
    global_keys = positive.sample(n_positive, random_state=42).index.union(
        negative.sample(n_negative, random_state=42).index).sort_values()
    high = hybrid[hybrid.hybrid_risk_level.isin(['HIGH', 'CRITICAL'])]
    local_parts = [part.sample(min(100, len(part)), random_state=42).index
                   for _, part in high.groupby('hybrid_risk_level', sort=True)]
    local_keys = high.index[:0]
    for part in local_parts:
        local_keys = local_keys.union(part)
    # All warning rows selected without event IDs/types/times, then explained once.
    warnings = hybrid.index[hybrid.operational_alert | supervised.operational_alert]
    union = global_keys.union(local_keys).union(warnings).union(live.index).sort_values()
    print(f'Explaining {len(union)} unique TEST observations in batches; {len(global_keys)} global sample.', flush=True)
    explanations, probabilities = explainer.explain(test.loc[union, features])
    if not np.allclose(probabilities, hybrid.loc[union, 'failure_probability'], rtol=1e-7, atol=1e-10):
        raise ValueError('Displayed probability differs from frozen saved calibrated probability.')
    positions = {key: i for i, key in enumerate(union)}
    def record(key):
        i = positions[key]
        meta = hybrid.loc[key].to_dict()
        meta.update(server_id=key[0], timestamp=key[1])
        return local_record(meta, explanations[i], probabilities[i], threshold, features)
    global_positions = [positions[key] for key in global_keys]
    importance = save_global(explanations[global_positions], REPORT_DIR, FIGURE_DIR)
    print(importance.head(20).to_string(index=False), flush=True)
    sampled = test.loc[global_keys, ['failure_within_24h']].reset_index()
    sampled.to_csv(REPORT_DIR/'shap_global_sample.csv', index=False)
    local_records = [record(key) for key in local_keys]
    (REPORT_DIR/'shap_local_explanations.jsonl').write_text(
        ''.join(json.dumps(r, allow_nan=False)+'\n' for r in local_records))
    save_live_view([record(key) for key in live.index], REPORT_DIR/'explainability_live_view.csv')
    # Durable explanation cache written BEFORE event outcome metadata is read.
    np.savez_compressed(REPORT_DIR/'shap_explanation_cache.npz', values=explanations.values,
                        base_values=explanations.base_values, data=explanations.data,
                        calibrated_probability=probabilities, features=np.asarray(features))
    pd.DataFrame(list(union), columns=KEYS).to_csv(REPORT_DIR/'shap_explanation_cache_keys.csv', index=False)
    print('SHAP computation complete. Loading existing event reports for reporting joins only.', flush=True)
    events = []
    for source, name in [('supervised', 'failure_event_analysis.csv'), ('hybrid', 'hybrid_failure_event_analysis.csv')]:
        event = pd.read_csv(REPORT_DIR/name)
        event = event[event.detected].copy()
        event['warning_source'] = source
        events.append(event)
    events = pd.concat(events, ignore_index=True)
    events['first_alert_timestamp'] = pd.to_datetime(events.first_alert_timestamp, utc=True)
    events['actual_failure_timestamp'] = pd.to_datetime(events.actual_failure_timestamp, utc=True)
    events = events.sort_values(['first_alert_timestamp', 'failure_event_id', 'warning_source']).drop_duplicates('failure_event_id')
    event_rows = []
    for event in events.itertuples(index=False):
        key = (event.server_id, event.first_alert_timestamp)
        if key not in positions or key not in warnings:
            raise ValueError('First event warning lacks a previously computed explanation.')
        explanation = record(key)
        lead = (event.actual_failure_timestamp-event.first_alert_timestamp).total_seconds()/3600
        if not 0 < lead <= 24:
            raise ValueError('Event reporting warning outside prediction horizon.')
        row = {'failure_event_id': event.failure_event_id, 'server_id': event.server_id,
               'first_warning_timestamp': str(event.first_alert_timestamp),
               'actual_failure_timestamp': str(event.actual_failure_timestamp),
               'warning_lead_time_hours': lead, 'warning_source': event.warning_source,
               'calibrated_failure_probability': explanation['calibrated_failure_probability'],
               'hybrid_risk_level': explanation['hybrid_risk_level']}
        for rank in range(1, 4):
            factor = explanation['top_risk_drivers'][rank-1] if len(explanation['top_risk_drivers']) >= rank else {}
            row[f'top_driver_{rank}'] = factor.get('feature')
            row[f'top_driver_{rank}_shap'] = factor.get('shap_value')
        event_rows.append(row)
    event_table = pd.DataFrame(event_rows)
    event_table.to_csv(REPORT_DIR/'failure_event_explanations.csv', index=False)
    # Deterministic example: earliest first warning with HIGH/CRITICAL, event-ID tie-break.
    preferred = event_table[event_table.hybrid_risk_level.isin(['HIGH', 'CRITICAL'])]
    example_event = (preferred if len(preferred) else event_table).iloc[0]
    key = (example_event.server_id, pd.Timestamp(example_event.first_warning_timestamp))
    example = record(key)
    local_explanation = explanations[positions[key]]
    display = shap.Explanation(values=local_explanation.values, base_values=local_explanation.base_values,
                               data=local_explanation.data, feature_names=[DISPLAY_NAMES[f] for f in features])
    shap.plots.waterfall(display, max_display=14, show=False)
    plt.gcf().set_size_inches(13, 9)
    plt.title(f'{key[0]}: base-model log-odds explanation\nCalibrated failure probability {example["calibrated_failure_probability"]:.4f}; {example["hybrid_risk_level"]}')
    plt.tight_layout(); plt.savefig(FIGURE_DIR/'example_shap_waterfall.png', dpi=160, bbox_inches='tight'); plt.close()
    # Frozen robust anomaly statistics, transform/explain only; no fit.
    anomaly_preprocessor = joblib.load(MODEL_DIR/'anomaly_preprocessor.joblib')
    robust = joblib.load(MODEL_DIR/'anomaly_robust_detector.joblib')
    anomaly_features = json.loads((MODEL_DIR/'anomaly_statistics.json').read_text())['feature_names']
    anomaly_input = anomaly_preprocessor.transform(test.loc[[key], anomaly_features]).astype(np.float32)
    deviations = robust.explain(anomaly_input[0])
    if not np.isclose(deviations['overall_anomaly_score'], example['anomaly_score'], rtol=1e-6, atol=1e-6):
        raise ValueError('Robust explanation differs from saved anomaly score.')
    root_details = [json.loads(line) for line in (REPORT_DIR/'root_cause_details.jsonl').read_text().splitlines()]
    matching = [d for d in root_details if key[0] in d['affected_servers']
                and pd.Timestamp(d['window_start']) <= key[1] < pd.Timestamp(d['window_end_exclusive'])]
    root = None
    if matching:
        d = matching[0]
        root = {k: d[k] for k in ['incident_id', 'suspected_component', 'suspected_component_type',
                'root_cause_confidence_score', 'telemetry_signature', 'inspection_priority',
                'affected_server_count', 'suspected_component_affected_servers', 'connected_servers',
                'potential_additional_exposure', 'inference_as_of', 'limitations']}
        root['evidence_status'] = 'UNKNOWN_LOW_EVIDENCE' if d['suspected_component_type'] == 'UNKNOWN' else 'UNCONFIRMED_HYPOTHESIS'
    combined = {'server_id': key[0], 'timestamp': str(key[1]),
                'selection_basis': 'Earliest detected-event first warning with HIGH/CRITICAL hybrid risk; event-ID tie-break. Metadata joined after SHAP.',
                'supervised': example,
                'anomaly': {'anomaly_score': example['anomaly_score'], 'anomaly_percentile': example['anomaly_percentile'],
                            'robust_deviations': deviations},
                'hybrid': {'risk_level': example['hybrid_risk_level'],
                           'reason_codes': str(hybrid.loc[key, 'reason_code']).split('|')},
                'root_cause_intelligence': root,
                'root_cause_availability_note': 'Closed-window hypothesis available at inference_as_of, not necessarily at the first warning. Null means no associated saved incident.',
                'failure_event_reporting_metadata': example_event.to_dict(), 'disclaimer': DISCLAIMER,
                'root_cause_benchmark_limitation': 'Experimental: 0/41 correct signatures, 36 UNKNOWN and 5 incorrect non-UNKNOWN; no component causal ground truth. Rack/switch memberships are identical.'}
    (REPORT_DIR/'example_sentinel_intelligence_report.json').write_text(json.dumps(combined, indent=2, allow_nan=False)+'\n')
    count = verify_hashes(ROOT, before)
    summary = {'shap_version': shap.__version__, 'underlying_model': type(explainer.estimator).__name__,
               'explained_output': 'positive-class raw log-odds; calibration is reported separately',
               'feature_perturbation': 'tree_path_dependent; frozen training path counts; no TEST background fitting',
               'global_observations': len(global_keys), 'global_positive_observations': int(sampled.failure_within_24h.sum()),
               'global_sample_method': 'Proportional class-stratified TEST sample without replacement, seed 42; explanation only',
               'unique_observations_explained': len(union), 'local_explanations': len(local_records),
               'failure_event_explanations': len(event_rows), 'live_view_rows': len(live),
               'protected_file_count': count, 'frozen_supervised_artifacts_unchanged': True,
               'frozen_anomaly_artifacts_unchanged': True, 'frozen_hybrid_artifacts_unchanged': True,
               'frozen_root_cause_artifacts_unchanged': True, 'dataset_hashes_unchanged': True,
               'training_or_tuning_performed': False, 'outcome_metadata_loaded_after_shap': True,
               'requirements_change': 'User-authorized addition shap>=0.46,<0.49. Legacy manifests are preserved; their requirements.txt hash predates this addition.',
               'example_server': key[0], 'example_probability': example['calibrated_failure_probability'],
               'example_hybrid_risk': example['hybrid_risk_level'],
               'top_10_global_features': importance.head(10).feature.tolist(),
               'example_top_drivers': example['top_risk_drivers'][:3]}
    (REPORT_DIR/'explainability_integrity_check.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
