"""Causal cleanup and stateless features shared by training and prediction."""
import numpy as np
import pandas as pd

from src.config import NUMERIC_COLUMNS, PROCESSED_PATH, RAW_PATH, ensure_directories
from src.data_validation import validate_data


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create features using only the current observation and fixed constants."""
    result = data.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise", utc=True)
    result["hour_of_day"] = result.timestamp.dt.hour
    result["day_of_week"] = result.timestamp.dt.dayofweek
    result["cpu_memory_interaction"] = result.cpu_utilization * result.memory_utilization / 100
    # Fixed engineering reference; never estimate this from the entire dataset.
    result["temperature_deviation"] = result.temperature_c - 55.0
    result["network_total_mb_s"] = result.network_in_mb_s + result.network_out_mb_s
    result["disk_total_mb_s"] = result.disk_read_mb_s + result.disk_write_mb_s
    # A unitless stress indicator, not a physical error rate.
    result["error_rate_indicator"] = np.log1p(result.smart_error_count) + result.packet_loss_pct / 5
    for feature, source, threshold in [
        ("high_temperature_flag", "temperature_c", 75),
        ("high_packet_loss_flag", "packet_loss_pct", 1),
    ]:
        result[feature] = result[source].gt(threshold).astype(float).where(result[source].notna())
    return result


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate and forward-fill at most two hours within each server.

    Residual missing values stay NaN for training-only median imputation.
    Never backfill from a future observation or compute full-data medians.
    """
    validate_data(data, verbose=False)
    result = data.drop_duplicates().copy()
    result["timestamp"] = pd.to_datetime(result.timestamp, errors="raise", utc=True)
    result = result.sort_values(["server_id", "timestamp"], kind="stable")
    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column])
        source_time = result.timestamp.where(result[column].notna())
        source_time = source_time.groupby(result.server_id).ffill()
        filled = result.groupby("server_id", sort=False)[column].ffill()
        fresh = (result.timestamp - source_time) <= pd.Timedelta(hours=2)
        result[column] = result[column].fillna(filled.where(fresh))
    result = engineer_features(result)
    return result.sort_values(["timestamp", "server_id"], kind="stable").reset_index(drop=True)


def main() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError("Generate data first: python -m src.generate_data")
    ensure_directories()
    data = preprocess_data(pd.read_csv(RAW_PATH))
    data.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved {len(data):,} processed rows: {PROCESSED_PATH}")
    print("Residual missing values use medians fitted on training data only.")
    return data


if __name__ == "__main__":
    main()
