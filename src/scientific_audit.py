"""Read-only benchmark diagnostics. Run: python -m src.scientific_audit.

Writes new audit reports only; never fits models or rewrites benchmark artifacts.
Test-label analyses are retrospective diagnostics, not fresh holdout estimates.
"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

from src.config import (ROOT, RAW_PATH, PROCESSED_PATH, MODEL_DIR, REPORT_DIR,
                        FEATURE_COLUMNS, NUMERIC_COLUMNS, FAILURE_METADATA_COLUMNS, TARGET)
from src.preprocess import engineer_features, preprocess_data
from src.train import temporal_split, MODEL_NAMES
from src.event_evaluation import analyze_failure_events

SIGNALS = ['temperature_c', 'smart_error_count', 'packet_loss_pct', 'network_latency_ms',
           'cpu_utilization', 'memory_utilization', 'power_watts', 'disk_utilization']


def feature_audit(test):
    """Use test observations only. No fitted single-feature classifiers."""
    auc_rows, overlap = [], []
    for feature in FEATURE_COLUMNS:
        values = test[feature]
        valid = values.notna()
        auc = roc_auc_score(test.loc[valid, TARGET], values[valid])
        strength = max(auc, 1 - auc)
        auc_rows.append({'feature': feature, 'roc_auc': auc, 'predictive_strength': strength,
                         'realism_review_flag': strength >= 0.90, 'test_rows_used': int(valid.sum())})
        healthy = values[test[TARGET].eq(0)].dropna().to_numpy()
        failing = values[test[TARGET].eq(1)].dropna().to_numpy()
        combined = np.concatenate([healthy, failing])
        unique = np.unique(combined)
        overlaps = {}
        for bins in [25, 50, 100]:
            if len(unique) <= 32:
                h = pd.Series(healthy).value_counts(normalize=True).reindex(unique, fill_value=0).to_numpy()
                f = pd.Series(failing).value_counts(normalize=True).reindex(unique, fill_value=0).to_numpy()
            else:
                edges = np.linspace(combined.min(), combined.max(), bins + 1)
                h = np.histogram(healthy, bins=edges)[0] / len(healthy)
                f = np.histogram(failing, bins=edges)[0] / len(failing)
            overlaps[bins] = float(np.minimum(h, f).sum())
        fpr, tpr, _ = roc_curve(test.loc[valid, TARGET], values[valid] if auc >= 0.5 else -values[valid])
        best_balanced_accuracy = float(np.max((tpr + 1 - fpr) / 2))
        overlap.append({'feature': feature, 'healthy_rows': len(healthy), 'failure_rows': len(failing),
                        'healthy_min': healthy.min(), 'healthy_p05': np.quantile(healthy, .05),
                        'healthy_median': np.median(healthy), 'healthy_p95': np.quantile(healthy, .95), 'healthy_max': healthy.max(),
                        'failure_min': failing.min(), 'failure_p05': np.quantile(failing, .05),
                        'failure_median': np.median(failing), 'failure_p95': np.quantile(failing, .95), 'failure_max': failing.max(),
                        'histogram_overlap_25': overlaps[25], 'histogram_overlap_50': overlaps[50],
                        'histogram_overlap_100': overlaps[100], 'predictive_strength': strength,
                        'best_single_cut_balanced_accuracy_diagnostic_only': best_balanced_accuracy,
                        'disjoint_observed_ranges': healthy.max() < failing.min() or failing.max() < healthy.min(),
                        'nearly_separable_flag': overlaps[50] < .10 or best_balanced_accuracy >= .95 or strength >= .99})
    auc = pd.DataFrame(auc_rows).sort_values(['predictive_strength', 'feature'], ascending=[False, True])
    auc.to_csv(REPORT_DIR / 'single_feature_auc.csv', index=False)
    pd.DataFrame(overlap).sort_values('histogram_overlap_50').to_csv(REPORT_DIR / 'healthy_failure_overlap.csv', index=False)
    print('\nTOP 10 SINGLE-FEATURE TEST AUCs (strength >= .90 is a realism flag, not leakage)')
    print(auc.head(10).to_string(index=False))
    return auc


def window_audit(raw, test, scored):
    """Compare the first labeled row and first qualifying model alert for each event."""
    rows = []
    global_groups = raw[raw[TARGET].eq(1)].groupby('failure_event_id')
    for event_id, group in test[test[TARGET].eq(1)].groupby('failure_event_id'):
        group = group.sort_values('timestamp')
        original = global_groups.get_group(event_id)
        failure = group.actual_failure_timestamp.iloc[0]
        positive = scored.loc[group.index]
        alerts = group.loc[positive.predicted_failure]
        first_alert = alerts.timestamp.min() if len(alerts) else pd.NaT
        before = scored[(scored.server_id == group.server_id.iloc[0]) &
                        (scored.timestamp < failure - pd.Timedelta(hours=24)) &
                        (scored.timestamp >= failure - pd.Timedelta(hours=48))]
        rows.append({'failure_event_id': event_id, 'server_id': group.server_id.iloc[0],
                     'actual_failure_timestamp': failure, 'first_positive_label_timestamp': group.timestamp.min(),
                     'first_model_alert_timestamp': first_alert,
                     'label_window_lead_hours': (failure - group.timestamp.min()).total_seconds() / 3600,
                     'model_lead_hours': (failure - first_alert).total_seconds() / 3600 if len(alerts) else np.nan,
                     'global_first_positive_label_timestamp': original.timestamp.min(),
                     'global_label_window_lead_hours': (failure - original.timestamp.min()).total_seconds() / 3600,
                     'alert_on_first_test_positive_label': bool(len(alerts) and first_alert == group.timestamp.min()),
                     'prewindow_test_observations_24_48h': len(before),
                     'prewindow_positive_alerts_24_48h': int(before.predicted_failure.sum())})
    result = pd.DataFrame(rows)
    result.to_csv(REPORT_DIR / 'failure_window_audit.csv', index=False)
    global_windows = global_groups.agg(server_id=('server_id', 'first'), actual_failure_timestamp=('actual_failure_timestamp', 'first'),
                                      first_positive_label_timestamp=('timestamp', 'min'), positive_rows=('timestamp', 'size')).reset_index()
    global_windows['label_window_lead_hours'] = (global_windows.actual_failure_timestamp - global_windows.first_positive_label_timestamp).dt.total_seconds() / 3600
    global_windows.to_csv(REPORT_DIR / 'all_event_window_audit.csv', index=False)
    print('\nTEST EVENT LABEL-WINDOW DISTRIBUTION (hours: count)')
    print(result.label_window_lead_hours.value_counts().sort_index().to_string())
    print('\nTEST EVENT MODEL WARNING DISTRIBUTION (hours: count)')
    print(result.model_lead_hours.value_counts().sort_index().to_string())
    return result, global_windows


def degradation_audit(raw, test):
    """Retrospective test-incident histories, allowing context before test starts.

    Bins are lower < hours_to_failure <= upper: each hourly reading appears once.
    This historical context is descriptive only, never fed back into models.
    """
    registry = test[test[TARGET].eq(1)][['failure_event_id', 'server_id', 'failure_type', 'actual_failure_timestamp']].drop_duplicates()
    histories = []
    servers = {key: frame for key, frame in raw.groupby('server_id')}
    for event in registry.itertuples(index=False):
        frame = servers[event.server_id].copy()
        frame['hours_to_failure'] = (event.actual_failure_timestamp - frame.timestamp).dt.total_seconds() / 3600
        frame = frame[frame.hours_to_failure.gt(0) & frame.hours_to_failure.le(72)].copy()
        frame['audit_event_id'] = event.failure_event_id
        frame['audit_failure_type'] = event.failure_type
        frame['within_test'] = frame.timestamp.ge(test.timestamp.min())
        histories.append(frame)
    history = pd.concat(histories, ignore_index=True)
    rows = []
    for kind, group in history.groupby('audit_failure_type'):
        for low, high in [(48, 72), (24, 48), (12, 24), (6, 12), (0, 6)]:
            subset = group[group.hours_to_failure.gt(low) & group.hours_to_failure.le(high)]
            for feature in SIGNALS:
                values = subset[feature]
                rows.append({'failure_type': kind, 'lead_time_bin_hours': f'{low}-{high}', 'lower_exclusive_hours': low,
                             'upper_inclusive_hours': high, 'feature': feature, 'observations': len(subset),
                             'events_represented': subset.audit_event_id.nunique(),
                             'context_rows_before_test': int((~subset.within_test).sum()),
                             'mean': values.mean(), 'std': values.std(), 'median': values.median(),
                             'p25': values.quantile(.25), 'p75': values.quantile(.75), 'min': values.min(), 'max': values.max()})
    pd.DataFrame(rows).to_csv(REPORT_DIR / 'failure_degradation_analysis.csv', index=False)
    # Adjacent hours determine whether an apparent 24h change is uniquely discontinuous.
    boundary = []
    for kind, group in history.groupby('audit_failure_type'):
        for hours in [26, 25, 24, 23, 22]:
            subset = group[group.hours_to_failure.eq(hours)]
            for feature in SIGNALS:
                boundary.append({'failure_type': kind, 'hours_before_failure': hours, 'feature': feature,
                                 'count': len(subset), 'mean': subset[feature].mean(), 'std': subset[feature].std()})
    pd.DataFrame(boundary).to_csv(REPORT_DIR / 'failure_boundary_diagnostics.csv', index=False)


def structure_audit(raw, test):
    """Check observed duplicate trajectories and close shapes, not latent RNG proofs."""
    registry = raw[raw[TARGET].eq(1)][['failure_event_id', 'server_id', 'failure_type', 'actual_failure_timestamp']].drop_duplicates()
    scale = test.loc[test[TARGET].eq(0), SIGNALS].std().replace(0, 1).to_numpy()
    traces, ids, types = [], [], []
    servers = {key: frame.sort_values('timestamp') for key, frame in raw.groupby('server_id')}
    for event in registry.itertuples(index=False):
        frame = servers[event.server_id]
        expected = pd.date_range(event.actual_failure_timestamp - pd.Timedelta(hours=24), periods=24, freq='h')
        subset = frame.set_index('timestamp').reindex(expected)
        if subset[SIGNALS].isna().any().any():
            continue
        traces.append(subset[SIGNALS].to_numpy())
        ids.append(event.failure_event_id)
        types.append(event.failure_type)
    values = np.array(traces)
    shapes = ((values - values.mean(axis=1, keepdims=True)) / scale).reshape(len(values), -1)
    normalized = shapes / np.linalg.norm(shapes, axis=1, keepdims=True)
    similarities = normalized @ normalized.T
    norms = np.sum(shapes ** 2, axis=1)
    rmse = np.sqrt(np.maximum(norms[:, None] + norms[None, :] - 2 * shapes @ shapes.T, 0) / shapes.shape[1])
    pairs = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            pairs.append({'event_a': ids[i], 'event_b': ids[j], 'type_a': types[i], 'type_b': types[j],
                          'shape_correlation': similarities[i, j], 'standardized_shape_rmse': rmse[i, j],
                          'exact_trajectory_match': bool(np.array_equal(values[i], values[j])),
                          'near_duplicate_flag': bool(similarities[i, j] >= .995 or rmse[i, j] <= .25)})
    comparisons = pd.DataFrame(pairs).sort_values('shape_correlation', ascending=False)
    comparisons.to_csv(REPORT_DIR / 'event_pattern_similarity.csv', index=False)
    repeated = []
    for feature in NUMERIC_COLUMNS:
        signatures = pd.Series([hashlib.sha256(frame[feature].to_numpy().tobytes()).hexdigest() for frame in servers.values()])
        repeated.append({'feature': feature, 'server_trajectories': len(signatures), 'repeated_exact_trajectories': int(signatures.duplicated().sum())})
    pd.DataFrame(repeated).to_csv(REPORT_DIR / 'repeated_pattern_audit.csv', index=False)
    summary = {'exact_duplicate_rows': int(raw.duplicated().sum()),
               'duplicate_server_timestamp_keys': int(raw.duplicated(['server_id', 'timestamp']).sum()),
               'duplicate_numeric_telemetry_vectors': int(raw.duplicated(NUMERIC_COLUMNS).sum()),
               'full_24h_events_compared': len(values), 'partial_events_excluded': len(registry) - len(values),
               'event_pairs_compared': len(comparisons), 'identical_event_pairs': int(comparisons.exact_trajectory_match.sum()),
               'near_duplicate_pairs': int(comparisons.near_duplicate_flag.sum()),
               'maximum_shape_correlation': float(comparisons.shape_correlation.max()),
               'minimum_standardized_shape_rmse': float(comparisons.standardized_shape_rmse.min()),
               'repeated_exact_server_feature_traces': sum(row['repeated_exact_trajectories'] for row in repeated)}
    (REPORT_DIR / 'structure_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def leakage_checks(raw, processed, train, validation, test, split):
    """Combine reviewed source provenance with dynamic schema/causality checks."""
    checks = []
    def check(name, passed, evidence):
        checks.append({'check': name, 'status': 'PASS' if passed else 'FAIL', 'evidence': evidence})
    forbidden = [TARGET, *FAILURE_METADATA_COLUMNS]
    saved_features = json.loads((MODEL_DIR / 'feature_columns.json').read_text())
    check('Explicit saved feature allowlist', saved_features == FEATURE_COLUMNS, ', '.join(saved_features))
    for name in forbidden:
        check(f'No direct ML input: {name}', name not in FEATURE_COLUMNS, f'Excluded from all {len(FEATURE_COLUMNS)} model inputs.')
    for name in [*MODEL_NAMES, 'best_failure_model']:
        model = joblib.load(MODEL_DIR / f'{name}.joblib')
        model = getattr(model, 'base_model', model)  # Audit the TRAIN pipeline beneath calibration.
        check(f'{name}: fitted feature schema', list(model.feature_names_in_) == FEATURE_COLUMNS,
              'Persisted pipeline feature_names_in_ equals the audited allowlist.')
        check(f'{name}: imputer fitted on train only', np.allclose(model.named_steps['imputer'].statistics_, train[FEATURE_COLUMNS].median().to_numpy()),
              'Persisted imputer statistics match training medians (also reviewed train.main fit path).')
        if 'scaler' in model.named_steps:
            transformed = model.named_steps['imputer'].transform(train[FEATURE_COLUMNS])
            check(f'{name}: scaler training moments', np.allclose(model.named_steps['scaler'].mean_, transformed.mean(axis=0)),
                  'Saved scaler mean matches imputed training features only.')
    sample = raw[raw.server_id.isin(raw.server_id.unique()[:8])].copy()
    original = engineer_features(sample)[FEATURE_COLUMNS]
    without_metadata = engineer_features(sample.drop(columns=forbidden))[FEATURE_COLUMNS]
    check('No indirect target/metadata dependency in feature engineering', original.equals(without_metadata),
          'All forbidden columns removed before engineer_features; outputs identical. Raw telemetry dependencies reviewed below.')
    cutoff = sample.timestamp.sort_values().iloc[len(sample)//2]
    # Create missing observations in the prefix to exercise causal forward filling.
    sample.loc[sample.index[::37], 'cpu_utilization'] = np.nan
    prefix = sample[sample.timestamp < cutoff]
    full_features = preprocess_data(sample)
    prefix_features = preprocess_data(prefix)
    left = full_features[full_features.timestamp < cutoff][FEATURE_COLUMNS].reset_index(drop=True)
    right = prefix_features[FEATURE_COLUMNS].reset_index(drop=True)
    check('Prefix invariance including missing-data handling', left.equals(right),
          'Removing all observations at/after midpoint leaves earlier engineered features unchanged, including injected missing CPU values.')
    changed = sample.copy()
    changed.loc[changed.timestamp >= cutoff, NUMERIC_COLUMNS] = 0
    changed_features = preprocess_data(changed)
    check('Future-value perturbation invariance', left.equals(changed_features[changed_features.timestamp < cutoff][FEATURE_COLUMNS].reset_index(drop=True)),
          'Replacing every future telemetry value with zero does not change earlier features.')
    check('Strict train/validation/test chronology', train.timestamp.max() < validation.timestamp.min() and validation.timestamp.max() < test.timestamp.min(), json.dumps(split))
    check('24h outcome windows do not cross split boundaries', train.timestamp.max() + pd.Timedelta(hours=24) < validation.timestamp.min() and validation.timestamp.max() + pd.Timedelta(hours=24) < test.timestamp.min(),
          'Purges at both boundaries; no future target window enters the next partition.')
    event_sets = [set(frame.failure_event_id.dropna()) for frame in [train, validation, test]]
    check('No incident shared across split warning rows', all(event_sets[i].isdisjoint(event_sets[j]) for i, j in [(0,1),(0,2),(1,2)]),
          f'Incident counts in train/validation/test: {[len(x) for x in event_sets]}')
    check('Stored split metadata matches data', json.loads((REPORT_DIR / 'split_metadata.json').read_text()) == split, 'Reconstructed split equals persisted boundaries, counts, and prevalence.')
    selection = json.loads((MODEL_DIR / 'decision_threshold.json').read_text())
    check('Validation-only model and threshold selection', selection['selection_split'] == 'validation',
          'Reviewed evaluate.select_on_validation: receives validation only. Final evaluator loads the frozen model/threshold and does not fit, select, or tune. Existing isolation tests verify this.')
    check('No random split / internal early stopping split', not joblib.load(MODEL_DIR / 'hist_gradient_boosting.joblib').named_steps['classifier'].early_stopping,
          'Unique timestamp boundaries; HGB early_stopping=False. Random Forest bootstrap is confined to training rows.')
    provenance = {
        'hour_of_day': 'current timestamp hour (UTC)', 'day_of_week': 'current timestamp weekday (UTC)',
        'cpu_memory_interaction': 'current CPU * current memory / 100',
        'temperature_deviation': 'current temperature - fixed constant 55',
        'network_total_mb_s': 'current network_in + current network_out',
        'disk_total_mb_s': 'current disk_read + current disk_write',
        'error_rate_indicator': 'log1p(current SMART errors) + current packet_loss / 5',
        'high_temperature_flag': 'current temperature > fixed constant 75',
        'high_packet_loss_flag': 'current packet_loss > fixed constant 1',
        'temperature_rolling_mean_6h': 'same-server mean of observed temperature in (t-6h, t]',
        'packet_loss_rolling_mean_6h': 'same-server mean of observed packet loss in (t-6h, t]',
        'latency_rolling_mean_6h': 'same-server mean of observed latency in (t-6h, t]',
        'cpu_rolling_mean_6h': 'same-server mean of observed CPU in (t-6h, t]',
        'memory_rolling_mean_6h': 'same-server mean of observed memory in (t-6h, t]',
        'temperature_change_6h': 'current temperature minus same-server observation exactly 6h earlier',
        'power_change_6h': 'current power minus same-server observation exactly 6h earlier',
        'smart_error_change_12h': 'current SMART count minus same-server observation exactly 12h earlier',
    }
    for feature, source in provenance.items():
        check(f'Engineered feature causality: {feature}', True, source + '; no future row, target, event ID, failure type, or failure timestamp.')
    lines = ['SCIENTIFIC LEAKAGE AUDIT', 'Scope: current implementation and saved artifacts; diagnostic-only.',
             'PASS means inspected/tested check passed, not proof of real-world validity.', '']
    lines.extend(f"{x['status']} | {x['check']}\n  {x['evidence']}" for x in checks)
    lines += ['', 'RAW FEATURE PROVENANCE REVIEW:',
              'All 15 raw inputs are current simulated telemetry, not copied labels or future observations.',
              'Generator uses a latent scheduled event to synthesize preceding degradation. This is backward simulation, not runtime target leakage; it creates strong synthetic shortcuts.',
              'uptime_days resets only at/after an observed event, not before it. Fan speed depends on current simulated temperature.',
              'Forward filling uses sorted same-server past readings only, with <=2h age; residual imputation and scaling are fit on train only.',
              'Validation outcomes select models/policy and fit only the probability calibrator; they never fit the base estimator or enter features.',
              'Final metadata is joined only after model prediction for retrospective incident evaluation.',
              'LIMITATION: source review plus finite perturbation tests cannot prove absence of every possible distributional shortcut.',
              'HISTORICAL HOLDOUT WARNING: this dataset was previously used in baseline model selection and inspected in earlier stages; current code isolation does not make it a fresh independent benchmark.']
    (REPORT_DIR / 'leakage_audit.txt').write_text('\n'.join(lines) + '\n')
    return checks


def load_audit_predictions(test):
    """Reuse final cached probabilities and policy decisions; do not rescore TEST."""
    cached = pd.read_csv(REPORT_DIR / 'test_predictions_and_alerts.csv')
    expected_keys = test[['server_id', 'timestamp']].reset_index(drop=True).copy()
    actual_keys = cached[['server_id', 'timestamp']].copy()
    for frame in [expected_keys, actual_keys]:
        frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True)
    if not actual_keys.equals(expected_keys):
        raise ValueError('Cached TEST predictions do not match audit observation keys/order.')
    return cached.calibrated_failure_probability.to_numpy(), cached.operational_alert.to_numpy(dtype=bool)


def main():
    raw = pd.read_csv(RAW_PATH, parse_dates=['timestamp', 'actual_failure_timestamp'])
    processed = pd.read_csv(PROCESSED_PATH)
    for col in ['timestamp', 'actual_failure_timestamp']:
        raw[col] = pd.to_datetime(raw[col], utc=True)
        processed[col] = pd.to_datetime(processed[col], utc=True)
    train, validation, test, split = temporal_split(processed)
    checks = leakage_checks(raw, processed, train, validation, test, split)
    auc = feature_audit(test)
    threshold = json.loads((MODEL_DIR / 'decision_threshold.json').read_text())['selected_threshold']
    scores, operational_alerts = load_audit_predictions(test)
    scored = test.copy()
    scored['model_risk_score'] = scores
    scored['predicted_failure'] = scores >= threshold
    row_metrics = pd.read_csv(REPORT_DIR / 'final_test_metrics.csv').iloc[0]
    assert np.isclose(average_precision_score(test[TARGET], scores), row_metrics.pr_auc)
    reconstructed, event_summary = analyze_failure_events(test, operational_alerts.astype(float), 0.5)
    persisted = pd.read_csv(REPORT_DIR / 'failure_event_analysis.csv')
    assert len(reconstructed) == len(persisted)
    for column in ['lead_time_hours']:
        joined = reconstructed.set_index('failure_event_id')[column].sort_index()
        expected = persisted.set_index('failure_event_id')[column].sort_index()
        assert np.allclose(joined, expected, equal_nan=True)
    windows, global_windows = window_audit(raw, test, scored)
    degradation_audit(raw, test)
    structure = structure_audit(raw, test)
    ever_failed = set(raw.loc[raw[TARGET].eq(1), 'server_id'])
    hard_negatives = []
    for feature, cutoff in [('cpu_utilization',90), ('temperature_c',75), ('packet_loss_pct',1), ('smart_error_count',5), ('smart_error_count',1)]:
        for cohort, subset in [('no_failure_next_24h', test[test[TARGET].eq(0)]),
                               ('servers_without_any_simulated_incident', test[~test.server_id.isin(ever_failed)])]:
            count = int(subset[feature].ge(cutoff).sum())
            hard_negatives.append({'feature': feature, 'abnormal_threshold_ge': cutoff, 'cohort': cohort,
                                   'observations': len(subset), 'abnormal_observations': count,
                                   'abnormal_pct': 100 * count / len(subset)})
    pd.DataFrame(hard_negatives).to_csv(REPORT_DIR / 'healthy_abnormal_observations.csv', index=False)
    diagnostics = {'window_audit_unit': 'raw_telemetry_predictions', 'event_summary_unit': 'operational_alerts', 'checks_passed': sum(x['status'] == 'PASS' for x in checks),
                   'checks_failed': [x['check'] for x in checks if x['status'] == 'FAIL'],
                   'strongest_feature': auc.iloc[0].feature, 'strongest_auc': float(auc.iloc[0].roc_auc),
                   'single_feature_flags': auc.loc[auc.realism_review_flag, 'feature'].tolist(),
                   'test_incidents': len(windows), 'global_incidents': len(global_windows),
                   'global_windows_start_at_24h': int(global_windows.label_window_lead_hours.eq(24).sum()),
                   'test_windows_start_at_24h': int(windows.label_window_lead_hours.eq(24).sum()),
                   'alert_on_first_test_label': int(windows.alert_on_first_test_positive_label.sum()),
                   'alerts_at_exactly_24h': int(windows.model_lead_hours.eq(24).sum()),
                   'median_label_window_lead_hours': float(windows.label_window_lead_hours.median()),
                   'median_model_lead_hours': float(windows.model_lead_hours.median()),
                   'prewindow_test_rows': int(windows.prewindow_test_observations_24_48h.sum()),
                   'prewindow_test_alerts': int(windows.prewindow_positive_alerts_24_48h.sum()),
                   'shared_train_test_servers': len(set(train.server_id) & set(test.server_id)),
                   'train_servers': train.server_id.nunique(), 'test_servers': test.server_id.nunique(),
                   'structure': structure, 'event_summary': event_summary}
    (REPORT_DIR / 'audit_diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
    print('\nAUDIT DIAGNOSTICS\n' + json.dumps(diagnostics, indent=2))


if __name__ == '__main__':
    main()
