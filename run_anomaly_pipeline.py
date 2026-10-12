"""Fit independent healthy-TRAIN anomaly models; freeze validation policy, score TEST once."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')

import hashlib
import json
from pathlib import Path
import time
from src.config import ROOT, REPORT_DIR
from src.anomaly_detection import load_anomaly_splits, fit_anomaly_models
from src.anomaly_evaluation import select_on_validation, final_test


def check_frozen_artifacts():
    path=REPORT_DIR/'anomaly_frozen_supervised_hashes.json'
    if not path.exists():
        raise FileNotFoundError('Missing frozen supervised integrity manifest; do not run without a benchmark snapshot.')
    expected=json.loads(path.read_text())
    changed=[name for name,digest in expected.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    if changed:
        raise ValueError(f'Frozen supervised benchmark changed: {changed}')
    return len(expected)


def main():
    protected=check_frozen_artifacts()
    started=time.perf_counter()
    training,validation,test,split=load_anomaly_splits()
    print('[1/3] Fitting preprocessing and anomaly detectors on healthy TRAIN only...',flush=True)
    preprocessor,detectors,training_metadata=fit_anomaly_models(training)
    print(f"Healthy training rows: {training_metadata['healthy_training_rows']:,}")
    print(f"Isolation Forest fitting sample: {training_metadata['isolation_training_sample_rows']:,}")
    print('[2/3] Complete VALIDATION comparison and threshold selection...',flush=True)
    selection=select_on_validation(validation,preprocessor,detectors)
    protocol={'selection':selection,'training_metadata':training_metadata,'outer_split':split,
              'test_previously_inspected_in_supervised_stages':True,
              'test_use':'One selected anomaly detector inference after all choices are frozen; no TEST tuning.',
              'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [ROOT/'src/anomaly_detection.py',ROOT/'src/anomaly_evaluation.py',Path(__file__).resolve()]}}
    (REPORT_DIR/'anomaly_frozen_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('[3/3] One frozen anomaly detector scoring pass on TEST...',flush=True)
    metrics,operational=final_test(test,selection)
    check_frozen_artifacts()
    integrity={'all_frozen_supervised_files_unchanged':True,'protected_file_count':protected,
               'pipeline_seconds':time.perf_counter()-started}
    (REPORT_DIR/'anomaly_integrity_verification.json').write_text(json.dumps(integrity,indent=2)+'\n')
    print(json.dumps({'row_metrics':metrics,'operational_metrics':operational},indent=2))
    print('Anomaly pipeline completed; frozen supervised artifacts unchanged.')


if __name__=='__main__':
    main()
