"""Score one telemetry record with the same feature transformations as training."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_DIR, NUMERIC_COLUMNS, PERCENT_COLUMNS
from src.preprocess import engineer_features, fill_recent_readings

EXAMPLE_RECORD = {
    "timestamp": "2025-02-01T12:00:00Z",
    "server_id": "server_0000",
    "cpu_utilization": 92, "memory_utilization": 94, "disk_utilization": 96,
    "gpu_utilization": 45, "temperature_c": 88,
    "disk_read_mb_s": 210, "disk_write_mb_s": 120,
    "network_in_mb_s": 160, "network_out_mb_s": 115,
    "packet_loss_pct": 3.2, "network_latency_ms": 48,
    "smart_error_count": 14, "fan_speed_rpm": 5200,
    "power_watts": 540, "uptime_days": 145,
}


def risk_level(score: float) -> str:
    """Illustrative operational buckets; these are not calibrated probabilities."""
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Model risk score must be a finite number between 0 and 1.")
    if score >= 0.8:
        return "CRITICAL"
    if score >= 0.5:
        return "HIGH"
    if score >= 0.2:
        return "MEDIUM"
    return "LOW"


def predict_record(record: dict, model_path: Path | None = None, history: pd.DataFrame | None = None) -> dict:
    """Load a trusted local model and score a record; null telemetry is imputed."""
    required = ["timestamp", *NUMERIC_COLUMNS]
    missing = sorted(set(required) - set(record))
    if missing:
        raise ValueError(f"Missing telemetry fields: {missing}. Supply null for unknown numeric readings.")
    frame = pd.DataFrame([{key: record[key] for key in required}])
    timestamp = pd.to_datetime(frame.timestamp, errors="coerce", utc=True)
    if timestamp.isna().any():
        raise ValueError("timestamp must be a valid date/time.")
    frame["timestamp"] = timestamp
    for column in NUMERIC_COLUMNS:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{column} must be numeric or null.") from error
        value = frame[column].iloc[0]
        if pd.isna(value):
            continue
        maximum = 100 if column in PERCENT_COLUMNS else (120 if column == "temperature_c" else np.inf)
        if not np.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"{column} is outside its valid range: 0 to {maximum}.")
        if column == "smart_error_count" and value % 1:
            raise ValueError("smart_error_count must be an integer.")
    model_inputs, history_count = prepare_model_inputs(frame, record.get("server_id"), history)
    path = model_path or MODEL_DIR / "best_failure_model.joblib"
    if not path.exists():
        raise FileNotFoundError("Train and evaluate first: python run_pipeline.py")
    features = json.loads((path.parent / "feature_columns.json").read_text())
    threshold_path = path.parent / "decision_threshold.json"
    if not threshold_path.exists():
        raise FileNotFoundError("Missing decision_threshold.json. Run validation selection first.")
    selection = json.loads(threshold_path.read_text())
    threshold = float(selection["selected_threshold"])
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Saved decision threshold must be between 0 and 1.")
    model = joblib.load(path)
    score = float(model.predict_proba(model_inputs[features])[0, 1])
    return {"calibrated_failure_probability": score if selection.get("calibrated", False) else None,
            "score_type": "calibrated_failure_probability" if selection.get("calibrated", False) else "uncalibrated_model_score",
            "operational_alert": None, "alert_note": "A single record is a raw prediction; run the chronological alert policy for incidents.",
            "model_risk_score": score, "decision_threshold": threshold,
            "predicted_failure_within_24h": bool(score >= threshold), "risk_level": risk_level(score),
            "history_observations": history_count}


def prepare_model_inputs(current: pd.DataFrame, server_id: str | None = None,
                         history: pd.DataFrame | None = None) -> tuple[pd.DataFrame, int]:
    """Reuse batch features with strictly historical readings for one server."""
    current = current.copy()
    current["timestamp"] = pd.to_datetime(current.timestamp, utc=True, errors="raise")
    past = pd.DataFrame(columns=["timestamp", *NUMERIC_COLUMNS, "server_id"])
    if history is not None and not history.empty:
        missing = sorted(set(["timestamp", *NUMERIC_COLUMNS]) - set(history))
        if missing:
            raise ValueError(f"History is missing fields: {missing}")
        past = history.copy()
        if "server_id" in past:
            if server_id is None:
                if past.server_id.nunique() != 1:
                    raise ValueError("Supply server_id in the current record when history includes multiple servers.")
                server_id = str(past.server_id.iloc[0])
            past = past[past.server_id.eq(server_id)].copy()
        past["timestamp"] = pd.to_datetime(past.timestamp, utc=True, errors="raise")
        if past.timestamp.isna().any() or (past.timestamp >= current.timestamp.iloc[0]).any():
            raise ValueError("History must contain only valid timestamps strictly before the current observation.")
        for column in NUMERIC_COLUMNS:
            past[column] = pd.to_numeric(past[column], errors="raise")
            observed = past[column].dropna()
            maximum = 100 if column in PERCENT_COLUMNS else (120 if column == "temperature_c" else np.inf)
            if not np.isfinite(observed).all() or not observed.between(0, maximum).all():
                raise ValueError(f"History {column} contains invalid values.")
            if column == "smart_error_count" and (observed % 1 != 0).any():
                raise ValueError("Historical smart_error_count must contain integers.")
    server_id = server_id or "single_record"
    current["server_id"] = server_id
    if past.empty:
        combined = current
    else:
        past["server_id"] = server_id
        combined = pd.concat([past[["timestamp", *NUMERIC_COLUMNS, "server_id"]], current], ignore_index=True)
    if combined.duplicated(["server_id", "timestamp"]).any():
        raise ValueError("History requires unique server/timestamp records.")
    featured = engineer_features(fill_recent_readings(combined))
    return featured[featured.timestamp.eq(current.timestamp.iloc[0])].reset_index(drop=True), len(past)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Path to one telemetry record as a JSON object.")
    parser.add_argument("--history", type=Path, help="CSV of prior telemetry; current JSON must identify its server when needed.")
    args = parser.parse_args()
    record = json.loads(args.input.read_text()) if args.input else EXAMPLE_RECORD
    history = pd.read_csv(args.history) if args.history else None
    result = predict_record(record, history=history)
    print(f"{result['score_type'].replace('_', ' ').title()}: {result['model_risk_score']:.4f}")
    print(result["alert_note"])
    print(f"Decision threshold: {result['decision_threshold']:.2f}")
    print(f"Predicted failure within 24h: {'YES' if result['predicted_failure_within_24h'] else 'NO'}")
    print(f"Risk level: {result['risk_level']}")
    print(f"Historical observations used: {result['history_observations']}")
    if result['history_observations'] == 0:
        print("No history: rolling means use current readings; missing lag changes use training medians.")


if __name__ == "__main__":
    main()
