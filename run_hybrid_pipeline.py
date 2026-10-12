"""Compare frozen-signal policies on validation, freeze, then replay TEST once."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')

import hashlib
import json
from pathlib import Path
from src.config import ROOT,MODEL_DIR,REPORT_DIR
from src.anomaly_detection import load_anomaly_splits
from src.hybrid_evaluation import load_validation_signals,select_on_validation,final_test,UTILITY_RULE,SELECTION_RULE


def check_integrity():
    expected=json.loads((REPORT_DIR/'hybrid_protected_artifacts.json').read_text())
    changed=[name for name,digest in expected.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    if changed:raise ValueError(f'Protected baseline artifact changed: {changed}')
    return len(expected)


def main():
    count=check_integrity()
    _,validation,test,split=load_anomaly_splits()
    supervised=json.loads((MODEL_DIR/'decision_threshold.json').read_text())
    anomaly=json.loads((MODEL_DIR/'anomaly_threshold.json').read_text())
    sources=[ROOT/'src/hybrid_risk.py',ROOT/'src/hybrid_evaluation.py',Path(__file__).resolve()]
    protocol={'utility_formula':UTILITY_RULE,'selection_rule':SELECTION_RULE,'outer_split':split,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'test_use':'Only frozen selected hybrid policy applied to cached TEST scores. Detector overlap is inspected afterward, never for selection.',
        'underlying_models_refit':False}
    (REPORT_DIR/'hybrid_prespecified_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('[1/3] Loading frozen models for VALIDATION inference only...',flush=True)
    p,a=load_validation_signals(validation)
    print('[2/3] Normalization and operational policy comparison on VALIDATION...',flush=True)
    frozen=select_on_validation(validation,p,a,supervised,anomaly)
    print('[3/3] Frozen policy replay on cached TEST signals, then detector overlap...',flush=True)
    metrics,overlap=final_test(test,frozen)
    check_integrity()
    integrity={'frozen_supervised_artifacts_unchanged':True,'frozen_anomaly_artifacts_unchanged':True,
        'dataset_hashes_unchanged':True,'outer_split_unchanged':True,'protected_file_count':count,
        'normalization_fit_split':'VALIDATION','policy_selection_split':'VALIDATION',
        'risk_boundary_selection_split':'prespecified constants anchored to frozen validation thresholds',
        'test_used_for_tuning':False,'test_used_for_underlying_inference':False,
        'selected_hybrid_test_applications':1,'overlap_inspected_only_after_policy_frozen':True,
        'frozen_hybrid_policy_sha256':hashlib.sha256((MODEL_DIR/'hybrid_policy.json').read_bytes()).hexdigest(),
        'source_hashes_unchanged_during_run':all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==d for p,d in protocol['source_sha256'].items())}
    (REPORT_DIR/'hybrid_integrity_check.json').write_text(json.dumps(integrity,indent=2)+'\n')
    print('Hybrid pipeline completed; all pre-existing frozen artifacts unchanged.')


if __name__=='__main__':main()
