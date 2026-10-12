"""Replay evaluation against saved inference; never fit or run inference."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from src.config import ROOT, REPORT_DIR, FIGURE_DIR, PROCESSED_PATH
from src.root_cause_evaluation import TYPE_SIGNATURE, evaluate_signatures


def main():
    # Preserve all existing benchmark and inference artifacts except the requested figure.
    figure = FIGURE_DIR / 'root_cause_signature_confusion_matrix.png'
    protected = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ['data', 'models', 'reports', 'src']
        for p in (ROOT / folder).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p != figure
        and p.name not in {'root_cause_signature_event_details.csv',
                           'root_cause_signature_diagnostic.json'}
    }
    expected_mapping = {
        'network_failure': 'NETWORK_LIKE', 'overheating': 'THERMAL_LIKE',
        'power_failure': 'POWER_LIKE', 'disk_failure': 'DISK_LIKE',
        'memory_failure': 'MEMORY_COMPUTE_LIKE',
    }
    assert TYPE_SIGNATURE == expected_mapping
    details = [json.loads(line) for line in
               (REPORT_DIR / 'root_cause_details.jsonl').read_text().splitlines()]
    existing = pd.read_csv(REPORT_DIR / 'root_cause_signature_evaluation.csv')
    hybrid = pd.read_csv(REPORT_DIR / 'hybrid_test_scores.csv',
                         usecols=['server_id', 'timestamp', 'operational_alert'])
    truth = pd.read_csv(PROCESSED_PATH, usecols=[
        'server_id', 'timestamp', 'datacenter_id', 'failure_event_id',
        'failure_type', 'actual_failure_timestamp', 'failure_within_24h'])
    truth['timestamp'] = pd.to_datetime(truth.timestamp, utc=True)
    split = json.loads((REPORT_DIR / 'split_metadata.json').read_text())['test']
    truth = truth[truth.timestamp.between(pd.Timestamp(split['start']),
                                          pd.Timestamp(split['end']))]
    protocol = json.loads((REPORT_DIR / 'root_cause_prespecified_protocol.json').read_text())
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        evaluate_signatures(details, hybrid, truth, temporary, temporary,
                            protocol['window_hours'])
        replay = pd.read_csv(temporary / 'root_cause_signature_evaluation.csv')
        pd.testing.assert_frame_equal(existing, replay)
        shutil.copyfile(temporary / figure.name, figure)
    assert existing.failure_event_id.is_unique
    table = existing[['failure_event_id', 'true_failure_type', 'inferred_signature',
                      'incident_id']].copy()
    table['expected_signature'] = table.true_failure_type.map(expected_mapping)
    assert table.expected_signature.notna().all()
    assert table.expected_signature.eq(existing.true_signature).all()
    table['correct'] = table.expected_signature.eq(table.inferred_signature)
    assert table.correct.eq(existing.correct).all()
    by_id = {d['incident_id']: d for d in details}
    assert all(row.inferred_signature == by_id[row.incident_id]['telemetry_signature']
               for row in table.itertuples())
    table['root_cause_confidence_score'] = table.incident_id.map(
        {key: value['root_cause_confidence_score'] for key, value in by_id.items()})
    table = table[['failure_event_id', 'true_failure_type', 'expected_signature',
                   'inferred_signature', 'correct', 'incident_id',
                   'root_cause_confidence_score']]
    table.to_csv(REPORT_DIR / 'root_cause_signature_event_details.csv', index=False)
    known = table.inferred_signature.ne('UNKNOWN')
    result = {
        'evaluation_mapping_bug_found': False,
        'evaluation_replay_matches_saved_events': True,
        'mapping': expected_mapping,
        'evaluated_events': len(table),
        'correct_signatures': int(table.correct.sum()),
        'incorrect_non_unknown': int((known & ~table.correct).sum()),
        'unknown': int((~known).sum()),
        'signature_accuracy_including_unknown': float(table.correct.mean()) if len(table) else None,
        'conditional_accuracy_excluding_unknown': float(table.loc[known, 'correct'].mean()) if known.any() else None,
        'conditional_accuracy_denominator': int(known.sum()),
        'true_failure_type_counts': table.true_failure_type.value_counts().to_dict(),
        'inferred_signature_counts': table.inferred_signature.value_counts().to_dict(),
        'interpretation': 'Signature agreement on alert-covered events; UNKNOWN counts as incorrect. Conditional accuracy is supplementary, not the main metric. No component causal accuracy is available.',
    }
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest
               for p, digest in protected.items())
    frozen = json.loads((REPORT_DIR / 'root_cause_protected_artifacts.json').read_text())
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
               for p, digest in frozen.items())
    result.update(inference_outputs_changed=False, frozen_artifacts_unchanged=True,
                  dataset_hashes_unchanged=True)
    (REPORT_DIR / 'root_cause_signature_diagnostic.json').write_text(
        json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
