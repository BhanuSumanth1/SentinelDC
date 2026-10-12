"""Run SentinelDC's existing-data training and frozen final evaluation."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import hashlib
import json
from pathlib import Path
from src import data_validation, evaluate, generate_data, preprocess, train
from src.config import RAW_PATH, PROCESSED_PATH, REPORT_DIR


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true',
                        help='Replay cached probabilities and check frozen artifacts without training or writes.')
    args = parser.parse_args(argv)
    if args.verify_only:
        from src.verify_frozen import verify_frozen_artifacts
        verify_frozen_artifacts()
        print('SentinelDC verification pipeline completed successfully.')
        return
    # Reuse the hardened benchmark; generating new scores must not change data.
    print("[1/6] Checking existing telemetry...", flush=True)
    if not RAW_PATH.exists():
        generate_data.main()
    print("\n[2/6] Validating data...", flush=True)
    data_validation.main()
    print("\n[3/6] Checking processed features...", flush=True)
    if not PROCESSED_PATH.exists():
        preprocess.main()
    else:
        print("Reusing existing processed data; generator and feature values unchanged.")
    integrity_path = REPORT_DIR / "advanced_input_integrity.json"
    if integrity_path.exists():
        expected = json.loads(integrity_path.read_text())
        root = Path(__file__).resolve().parent
        changed = [p for p, digest in expected.items() if hashlib.sha256((root / p).read_bytes()).hexdigest() != digest]
        if changed:
            raise ValueError(f"Hardened benchmark differs from frozen input snapshot: {changed}")
    print("\n[4/6] Training five models on TRAIN only...", flush=True)
    train.main()
    print("\n[5/6] Validation model choice, calibration, and alert policy...", flush=True)
    validation, test, split = evaluate.load_evaluation_data()
    selection = evaluate.select_on_validation(validation)
    print("\n[6/6] One frozen model evaluation on TEST...", flush=True)
    evaluate.evaluate_final_test(test, selection, split)
    print("\nSentinelDC pipeline completed successfully.")


if __name__ == "__main__":
    main()
