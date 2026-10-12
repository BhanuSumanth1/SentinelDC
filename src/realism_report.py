"""Describe the hardened dataset from persisted data and measured test results."""
import hashlib
import json

import pandas as pd

from src.config import RAW_PATH, REPORT_DIR, ROOT, TARGET, FEATURE_COLUMNS
from src.generate_data import GENERATOR_VERSION


def write_realism_report(selection: dict, metrics: dict, operational: dict, auc: pd.DataFrame) -> None:
    """Write facts from artifacts; never alter generation, fitting, or thresholds."""
    labels = pd.read_csv(RAW_PATH, usecols=['server_id', TARGET, 'failure_event_id'])
    transients = pd.read_csv(RAW_PATH.parent / 'healthy_transients.csv')
    catalog = pd.read_csv(RAW_PATH.parent / 'failure_event_catalog.csv')
    shared = pd.read_csv(RAW_PATH.parent / 'shared_conditions.csv')
    profiles = pd.read_csv(RAW_PATH.parent / 'server_profiles.csv')
    events = operational['failure_events']
    strongest = auc.iloc[0]
    flags = auc.loc[auc.predictive_strength.ge(.90), 'feature'].tolist()
    healthy_count = int(transients.fully_healthy_24h.sum())
    statistics = {
        'generator_version': GENERATOR_VERSION, 'seed': 42,
        'rows': len(labels), 'servers': int(labels.server_id.nunique()),
        'actual_failure_events_dataset': int(labels.failure_event_id.nunique()),
        'actual_failure_events_test': events['total_failure_events'],
        'positive_fraction': float(labels[TARGET].mean()), 'healthy_transient_anomaly_count': healthy_count,
        'all_server_transients': len(transients), 'shared_condition_episodes': len(shared),
        'strongest_feature': strongest.feature, 'single_feature_roc_auc': float(strongest.roc_auc),
        'single_feature_predictive_strength': float(strongest.predictive_strength),
        'realism_warning_features': flags, 'selected_model': selection['model_name'],
        'selected_threshold': selection['selected_threshold'],
        'test_pr_auc': metrics['pr_auc'], 'test_roc_auc': metrics['roc_auc'],
        'test_recall': metrics['recall'], 'test_precision': metrics['precision'], 'test_f1': metrics['f1'],
        'test_false_alerts': metrics['false_positive'],
        'false_alerts_per_1000_healthy_observations': operational['false_alerts_per_1000_healthy_observations'],
        'event_detection_rate': events['event_detection_rate'],
        'median_warning_lead_hours': events['median_lead_time_hours'],
        'mean_warning_lead_hours': events['mean_lead_time_hours'],
        'feature_count': len(FEATURE_COLUMNS),
        'generator_source_sha256': hashlib.sha256((ROOT / 'src/generate_data.py').read_bytes()).hexdigest(),
    }
    (REPORT_DIR / 'hardened_dataset_statistics.json').write_text(json.dumps(statistics, indent=2) + '\n')
    def table(headers, rows):
        return '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' + ''.join('| ' + ' | '.join(map(str, row)) + ' |\n' for row in rows)
    text = f'''# Hardened dataset realism report

Generator: `{GENERATOR_VERSION}`, seed 42. This is a synthetic portfolio benchmark, not validated data-center reliability physics. Generator parameters were specified before evaluating the regenerated test data; they were not adjusted to attain a score, AUC cutoff, or missed-event count.

## Generated dataset

'''
    text += table(['Quantity', 'Measured value'], [
        ('Telemetry rows', f'{len(labels):,}'), ('Servers', labels.server_id.nunique()),
        ('Actual incidents in the full dataset', labels.failure_event_id.nunique()),
        ('Failure-positive observations', f'{labels[TARGET].mean():.4%}'),
        ('Server-local recovering disturbances', len(transients)),
        ('Disturbances entirely outside any 24h warning window', healthy_count),
        ('Shared rack/switch condition episodes', len(shared)),
        ('TEST incidents', events['total_failure_events']),
    ])
    text += '''
A healthy transient is one finite server-local disturbance whose every observed row has label 0; it is counted once, not once per telemetry row. Other exogenous disturbances may overlap an unrelated fault warning window. Shared episodes are counted separately, once per affected rack or switch, rather than once per server. Catalog counts identify injected mechanisms, not model alarms or proof that all samples crossed a fixed abnormality threshold. No difficult observations were removed to adjust prevalence.

Incident incidence is deliberately accelerated: achieving about 1–3% positive hourly rows with 24-hour labels implies far more incidents per server-month than many real fleets. At most one incident per server and immediate simulated repair remain simplifying assumptions.

## Changes from the original benchmark

- Persistent, heterogeneous normal CPU, memory, temperature, latency, packet loss, power, disk activity, and fan profiles prevent a universal absolute baseline. Normal temperature spans ''' + f'{profiles.normal_temperature.min():.1f}–{profiles.normal_temperature.max():.1f}°C in this run.\n' + '''- Recovering CPU/memory/thermal/network/disk/SMART/power/fan disturbances create non-failure lookalikes. Shared switch congestion and rack thermal/power changes affect neighboring servers without determining their fault labels.
- Fault onset is drawn by type from a bounded lognormal distribution (4–72h); event strength, slope, affected channels, correlated noise, and intermittent recovery vary independently. A short-onset failure can have many normal-looking positive-label rows. Some weak or sparse signatures can be hard to detect.
- Fault-specific mechanisms replace the universal packet-loss/latency/SMART ramp. Disk faults need not disturb networking, and network faults need not disturb temperature. Current CPU and disk load influence temperature/power/fan through noisy nonlinear relationships.
- Eight past-only trends supplement the 24 original inputs. Means use `(t−6h, t]`; differences use the same server's exact timestamp at t−6h or t−12h. Missing lag observations remain NaN for a training-fitted imputer. Identifiers and all outcome metadata remain excluded.

Observed onset variation by type:

'''
    grouped = catalog.groupby('failure_type').degradation_onset_hours.agg(['count', 'min', 'median', 'max', 'nunique'])
    text += table(['Failure type', 'Events', 'Min onset h', 'Median onset h', 'Max onset h', 'Unique onsets'],
                  [(kind, int(row['count']), int(row['min']), row['median'], int(row['max']), int(row['nunique'])) for kind, row in grouped.iterrows()])
    text += f'''
The target remains **an actual event in (t, t + 24h]**, independent of when degradation begins. The four simulation catalogs in `data/raw/` are provenance only: server profiles, failure onset/severity, benign episodes, and shared conditions are never joined into model features.

## Frozen final test results

Train/validation/test retain chronological 70/15/15 timestamp windows and a 24h purge before each new partition. All estimators, imputers, and scaling fit on train only. Model selection and operating threshold selection use validation only. The selected model is scored once on test; event analysis reuses those scores. The single-feature diagnostic computes rankings directly from held-out features without fitting alternative models or changing decisions.

'''
    text += table(['Metric', 'Measured value'], [
        ('Selected model', selection['model_name']), ('Selected threshold', f"{selection['selected_threshold']:.2f}"),
        ('Validation PR-AUC', f"{selection['validation_pr_auc']:.6f}"),
        ('Final TEST PR-AUC', f"{metrics['pr_auc']:.6f}"), ('Final TEST ROC-AUC', f"{metrics['roc_auc']:.6f}"),
        ('Final TEST recall', f"{metrics['recall']:.6f}"), ('Final TEST precision', f"{metrics['precision']:.6f}"),
        ('Final TEST F1', f"{metrics['f1']:.6f}"),
        ('Strongest individual feature', strongest.feature), ('Raw-direction single-feature ROC-AUC', f'{strongest.roc_auc:.6f}'),
        ('Direction-adjusted single-feature strength', f'{strongest.predictive_strength:.6f}'),
        ('Detected / missed TEST incidents', f"{events['detected_failure_events']} / {events['missed_failure_events']}"),
        ('TEST event detection rate', f"{events['event_detection_rate']:.2%}"),
        ('Median warning lead time', f"{events['median_lead_time_hours']:.2f} hours"),
        ('Mean warning lead time', f"{events['mean_lead_time_hours']:.2f} hours"),
        ('Lead time p25 / p75', f"{events['p25_lead_time_hours']:.2f} / {events['p75_lead_time_hours']:.2f} hours"),
    ])
    text += f'''
**Operational interpretation:** the frozen rule selected threshold {selection['selected_threshold']:.2f} and generated {metrics['false_positive']:,} false-positive test rows, or {operational['false_alerts_per_1000_healthy_observations']:.2f} false alerts per 1,000 healthy observations. Precision is {metrics['precision']:.2%}. Event detection and lead time must not be read as useful operational success without this alert burden. The pre-existing recall constraint can favor a very low threshold when benign/failure distributions overlap; that rule was preserved rather than changed after inspecting test performance.

The model test ROC-AUC ({metrics['roc_auc']:.4f}) {'exceeds' if metrics['roc_auc'] > strongest.predictive_strength else 'does not exceed'} the strongest single-feature direction-adjusted ROC-AUC ({strongest.predictive_strength:.4f}). Thus a benefit from combining features and trends is not automatically established by adding them. Model Average Precision ({metrics['pr_auc']:.4f}) should instead be compared with the test positive prevalence ({(metrics['true_positive'] + metrics['false_negative']) / operational['telemetry_observations']:.4f}) for a no-skill ranking reference. No score was forced downward or upward.

`uptime_days` can retain a structural shortcut: after a simulated repair the single-incident-per-server model cannot fail again in this realization. This is not future-data leakage, but can make low post-repair uptime unusually predictive of no subsequent incident. Addressing recurrent faults or removing that simplification requires a separately assessed future change.

Single-feature realism warning (strength ≥0.90): **{', '.join(flags) if flags else 'none'}**. These flags diagnose predictability, not automatically leakage. The model ROC-AUC and strongest individual ROC-AUC are comparable ranking summaries; model PR-AUC is a different metric and must not be compared numerically to a single-feature ROC-AUC. No raw-only versus trend-feature ablation was run, so these results do not isolate the causal contribution of trends.

The original benchmark's strongest individual feature was packet loss (ROC-AUC 0.9770), final PR-AUC 0.9497, event detection 100%, and median lead time 24h. Those measurements came from a different, easier data realization and are historical references, not scores to match. The original audit and source snapshots are preserved under `reports/baseline_reference/`. Latest results must come from the current CSVs and this report.

Event detection credits at least one positive warning in the event's available test `[failure−24h, failure)` window. Lead-time statistics condition on detected incidents; missed events have no artificial zero lead time. Partial windows and simulated future-tail outcomes remain flagged in the incident CSV. A median at 24h, if observed, is still limited by the scoring horizon. The test cohort has {events['partial_warning_window_events']} partial warning windows and {events['events_after_test_end']} failures after the last test observation with simulator-known outcomes.

## Remaining scientific limits

This revised generator uses plausible mechanisms rather than an empirically calibrated physical reliability model. All servers still recur across time partitions, scenarios come from a known simulator, and real production generalization remains unproven. Threshold rules prioritize recall and can produce substantial false alerts; incident detection must be interpreted with row precision and alert burden. Scores remain uncalibrated. Trend means are not server-normalized residuals, and sudden failures may lack enough precursor evidence. A genuinely independent future dataset and event-level alert-cost study are needed before operational claims.
'''
    (REPORT_DIR / 'hardened_dataset_realism.md').write_text(text)
    print(f"\nStrongest individual feature: {strongest.feature}")
    print(f"Single-feature ROC-AUC: {strongest.roc_auc:.6f} | Predictive strength: {strongest.predictive_strength:.6f}")
    print(f"Realism warning (strength >= 0.90): {', '.join(flags) if flags else 'none'}")
    print(f"Realism report: {REPORT_DIR / 'hardened_dataset_realism.md'}")
