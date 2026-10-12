"""Causal cleanup and current/past features shared by training and prediction."""
import numpy as np
import pandas as pd

from src.config import NUMERIC_COLUMNS, PROCESSED_PATH, RAW_PATH, ensure_directories
from src.data_validation import validate_data


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create same-row and per-server past/current features; preserve row order."""
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
    return add_trend_features(result)


def add_trend_features(data: pd.DataFrame) -> pd.DataFrame:
    """Use (t-6h, t] means and exact 6h/12h lag differences per server.

    A missing lag stays NaN for the training-fitted imputer. No future backfill,
    centered windows, or cross-server observations enter a trend.
    """
    result = data.copy()
    has_server = "server_id" in result
    if not has_server:
        result["server_id"] = "single_record"
    result["_original_order"] = np.arange(len(result))
    result = result.sort_values(["server_id", "timestamp"], kind="stable")
    if result.duplicated(["server_id", "timestamp"]).any():
        raise ValueError("Rolling features require unique server/timestamp observations.")
    means = {"temperature_c": "temperature_rolling_mean_6h", "packet_loss_pct": "packet_loss_rolling_mean_6h",
             "network_latency_ms": "latency_rolling_mean_6h", "cpu_utilization": "cpu_rolling_mean_6h",
             "memory_utilization": "memory_rolling_mean_6h"}
    parts = []
    for _, group in result.groupby("server_id", sort=False):
        group = group.copy()
        timeline = group.set_index("timestamp")
        for source, feature in means.items():
            group[feature] = timeline[source].rolling("6h", min_periods=1, closed="right").mean().to_numpy()
        for source, feature, lag in [("temperature_c", "temperature_change_6h", 6),
                                     ("smart_error_count", "smart_error_change_12h", 12),
                                     ("power_watts", "power_change_6h", 6)]:
            earlier = timeline[source].reindex(group.timestamp - pd.Timedelta(hours=lag)).to_numpy()
            group[feature] = group[source].to_numpy() - earlier
        parts.append(group)
    result = pd.concat(parts).sort_values("_original_order", kind="stable").drop(columns="_original_order")
    return result if has_server else result.drop(columns="server_id")


def fill_recent_readings(data: pd.DataFrame) -> pd.DataFrame:
    """Fill from the same server at most two hours ago; usable at inference."""
    result = data.drop_duplicates().copy()
    result.attrs = {}
    result["timestamp"] = pd.to_datetime(result.timestamp, errors="raise", utc=True)
    result = result.sort_values(["server_id", "timestamp"], kind="stable")
    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column])
        source_time = result.timestamp.where(result[column].notna())
        source_time = source_time.groupby(result.server_id).ffill()
        filled = result.groupby("server_id", sort=False)[column].ffill()
        fresh = (result.timestamp - source_time) <= pd.Timedelta(hours=2)
        result[column] = result[column].fillna(filled.where(fresh))
    return result


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate and forward-fill at most two hours within each server.

    Residual missing values stay NaN for training-only median imputation.
    Never backfill from a future observation or compute full-data medians.
    """
    validate_data(data, verbose=False)
    result = fill_recent_readings(data)
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
