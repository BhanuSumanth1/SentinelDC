"""Read-only verification of frozen artifacts; no fitting, tuning or inference."""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src import evaluate
from src.alert_policy import AlertPolicy, apply_alert_policy, evaluate_operational_alerts
from src.config import ROOT, TARGET
from src.train import temporal_split


def fingerprints(root):
    """Protect all saved data, models and reports against accidental writes."""
    paths = [p for directory in ['data', 'models', 'reports']
             for p in (root / directory).rglob('*') if p.is_file()]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def same_metrics(actual, saved):
    for key, value in actual.items():
        if isinstance(value, str):
            if value != saved[key]:
                raise ValueError(f'Saved metric differs: {key}')
        elif not np.isclose(value, saved[key], rtol=1e-10, atol=1e-12, equal_nan=True):
            raise ValueError(f'Saved metric differs: {key}: {value} versus {saved[key]}')


def verify_frozen_artifacts(root=ROOT):
    """Replay saved probabilities with the frozen policy; write only to tempdir."""
    root = Path(root)
    before = fingerprints(root)
    expected = json.loads((root / 'reports/advanced_input_integrity.json').read_text())
    for name, digest in expected.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen benchmark input changed: {name}')
    protocol = json.loads((root / 'reports/advanced_prespecified_protocol.json').read_text())
    # Entry-point verification and audit compatibility can change; behavior cannot.
    for name in ['train', 'calibration', 'evaluate', 'alert_policy', 'config', 'preprocess', 'generate_data', 'event_evaluation']:
        source = f'src/{name}.py'
        if hashlib.sha256((root / source).read_bytes()).hexdigest() != protocol['source_sha256'][source]:
            raise ValueError(f'Frozen scientific implementation changed: {source}')
    data = pd.read_csv(root / 'data/processed/server_telemetry_processed.csv')
    training, validation, test, split = temporal_split(data)
    if split != json.loads((root / 'reports/split_metadata.json').read_text()):
        raise ValueError('TRAIN/VALIDATION/TEST boundaries or counts changed.')
    selection = json.loads((root / 'models/decision_threshold.json').read_text())
    if selection['selection_split'] != 'validation':
        raise ValueError('Selection is not recorded as validation-only.')
    comparison = pd.read_csv(root / 'reports/advanced_validation_model_metrics.csv')
    chosen = comparison[comparison.selected]
    if len(chosen) != 1 or chosen.iloc[0]['model'] != selection['model_name'] or chosen.iloc[0].pr_auc != comparison.pr_auc.max():
        raise ValueError('Saved winner is inconsistent with validation PR-AUC.')
    # Check saved choices; never fit or run selection again.
    analysis = pd.read_csv(root / 'reports/advanced_threshold_analysis.csv')
    selected_rows = analysis[analysis.selected]
    if len(selected_rows) != 1 or selected_rows.iloc[0].threshold != selection['selected_threshold']:
        raise ValueError('Saved threshold differs from validation analysis.')
    eligible = analysis[analysis.preferred_constraints_met]
    selected_row = selected_rows.iloc[0]
    if (not eligible.empty and not selected_row.preferred_constraints_met) or not np.isclose(
            selected_row.f2, (eligible if not eligible.empty else analysis).f2.max()):
        raise ValueError('Saved threshold does not satisfy its validation F2 rule.')
    if pd.Timestamp(selection['calibration_window']['end']) + pd.Timedelta(hours=24) >= pd.Timestamp(selection['policy_validation_window']['start']):
        raise ValueError('Calibration/policy outcome-window purge is missing.')
    if pd.Timestamp(selection['policy_validation_window']['end']) >= test.timestamp.min():
        raise ValueError('Policy window overlaps TEST.')
    weights = json.loads((root / 'models/training_class_weights.json').read_text())
    if weights['negative_samples'] != int(training[TARGET].eq(0).sum()) or weights['positive_samples'] != int(training[TARGET].eq(1).sum()):
        raise ValueError('Saved imbalance weights do not use TRAIN counts.')
    if not np.isclose(weights['scale_pos_weight'], training[TARGET].eq(0).sum() / training[TARGET].eq(1).sum()):
        raise ValueError('Saved imbalance ratio differs from TRAIN.')
    cached = pd.read_csv(root / 'reports/test_predictions_and_alerts.csv')
    keys = test[['server_id', 'timestamp']].reset_index(drop=True)
    cached_keys = cached[['server_id', 'timestamp']].copy()
    cached_keys['timestamp'] = pd.to_datetime(cached_keys.timestamp, utc=True)
    if not cached_keys.equals(keys):
        raise ValueError('Cached predictions differ from TEST keys/order.')
    scores = cached.calibrated_failure_probability.to_numpy()
    raw_metrics = evaluate.classification_metrics(test[TARGET], scores, selection['selected_threshold'])
    same_metrics(raw_metrics, pd.read_csv(root / 'reports/final_test_metrics.csv').iloc[0])
    policy = AlertPolicy(selection['selected_threshold'], selection['emergency_threshold'],
                         selection['alert_persistence_hours'], selection['cooldown_hours'])
    scored = apply_alert_policy(test, scores, policy)
    # Compare every saved decision, streak and cooldown reason, not just totals.
    pd.testing.assert_frame_equal(scored.drop(columns='timestamp'), cached.drop(columns='timestamp'),
                                  check_dtype=False, rtol=1e-10, atol=1e-12)
    with tempfile.TemporaryDirectory(prefix='sentineldc-verification-') as temp:
        directory = Path(temp)
        figures = directory / 'figures'; figures.mkdir()
        operational = evaluate_operational_alerts(test, scored, directory, figures)
        same_metrics(operational, pd.read_csv(root / 'reports/operational_alert_metrics.csv').iloc[0])
        for filename in ['operational_alerts.csv', 'failure_event_analysis.csv', 'failure_lead_time_summary.csv']:
            pd.testing.assert_frame_equal(pd.read_csv(directory / filename), pd.read_csv(root / 'reports' / filename),
                                          check_dtype=False, rtol=1e-10, atol=1e-12)
    if before != fingerprints(root):
        raise ValueError('Verification unexpectedly changed frozen data/model/report artifacts.')
    print('Dataset hashes unchanged: YES')
    print('TRAIN/VALIDATION/TEST methodology unchanged: YES')
    print('Validation-only decisions and frozen scientific sources verified: YES')
    print('Alert policy and saved operational metrics reproduce: YES')
    print('No training, calibration fitting, selection, tuning, or TEST inference performed.')
    print('All data, model and report artifacts unchanged: YES')
    return {'selection': selection, 'raw_metrics': raw_metrics, 'operational_metrics': operational}
