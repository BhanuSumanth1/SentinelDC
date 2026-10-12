"""Shared paths, schema, and reproducible defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/server_telemetry.csv"
PROCESSED_PATH = ROOT / "data/processed/server_telemetry_processed.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
SEED = 42
TARGET = "failure_within_24h"
HORIZON_HOURS = 24
IDENTIFIERS = ["datacenter_id", "rack_id", "server_id", "switch_id"]
PERCENT_COLUMNS = [
    "cpu_utilization", "memory_utilization", "disk_utilization",
    "gpu_utilization", "packet_loss_pct",
]
NUMERIC_COLUMNS = [
    "cpu_utilization", "memory_utilization", "disk_utilization", "gpu_utilization",
    "temperature_c", "disk_read_mb_s", "disk_write_mb_s", "network_in_mb_s",
    "network_out_mb_s", "packet_loss_pct", "network_latency_ms", "smart_error_count",
    "fan_speed_rpm", "power_watts", "uptime_days",
]
DERIVED_COLUMNS = [
    "hour_of_day", "day_of_week", "cpu_memory_interaction", "temperature_deviation",
    "network_total_mb_s", "disk_total_mb_s", "error_rate_indicator",
    "high_temperature_flag", "high_packet_loss_flag",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + DERIVED_COLUMNS
FAILURE_METADATA_COLUMNS = ["failure_type", "failure_event_id", "actual_failure_timestamp"]
REQUIRED_COLUMNS = ["timestamp", *IDENTIFIERS, *NUMERIC_COLUMNS, TARGET, *FAILURE_METADATA_COLUMNS]
FAILURE_TYPES = ["disk_failure", "overheating", "network_failure", "power_failure", "memory_failure"]


def ensure_directories() -> None:
    """Create artifact folders without depending on the current directory."""
    for path in [RAW_PATH.parent, PROCESSED_PATH.parent, MODEL_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)
