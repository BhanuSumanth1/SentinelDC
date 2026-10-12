"""Validate schema and physical ranges with actionable errors."""
import numpy as np
import pandas as pd

from src.config import FAILURE_METADATA_COLUMNS, FAILURE_TYPES, IDENTIFIERS, NUMERIC_COLUMNS, PERCENT_COLUMNS, RAW_PATH, REQUIRED_COLUMNS, TARGET


def validate_data(data: pd.DataFrame, verbose: bool = True) -> dict:
    """Reject invalid values; report fixable missing telemetry and exact duplicates."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if data.empty:
        raise ValueError("Telemetry dataset is empty.")
    errors = []
    timestamps = pd.to_datetime(data["timestamp"], errors="coerce", utc=True, format="mixed")
    if timestamps.isna().any():
        errors.append("timestamp contains missing or invalid dates")
    if data[IDENTIFIERS].isna().any().any() or data[IDENTIFIERS].eq("").any().any():
        errors.append("identifiers must be present on every row")
    if not data[TARGET].isin([0, 1]).all():
        errors.append(f"{TARGET} must contain only 0 or 1 without missing values")
    if not data["failure_type"].isin(["none", *FAILURE_TYPES]).all():
        errors.append("failure_type contains missing or unknown categories")
    if (data["failure_type"].ne("none") != data[TARGET].eq(1)).any():
        errors.append("failure_type must be none exactly when failure_within_24h is 0")
    try:
        validate_failure_metadata(data)
    except ValueError as error:
        errors.append(str(error))
    for column in NUMERIC_COLUMNS:
        numeric = pd.to_numeric(data[column], errors="coerce")
        if (data[column].notna() & numeric.isna()).any():
            errors.append(f"{column} contains nonnumeric values")
        observed = numeric.dropna()
        if not np.isfinite(observed).all():
            errors.append(f"{column} contains infinite values")
        lower, upper = (0, 100) if column in PERCENT_COLUMNS else (0, np.inf)
        if column == "temperature_c":
            lower, upper = 0, 120
        if not observed.between(lower, upper).all():
            errors.append(f"{column} must be between {lower} and {upper}")
        if column == "smart_error_count" and (observed % 1 != 0).any():
            errors.append("smart_error_count must be a nonnegative integer")
    unique = data.drop_duplicates().copy()
    unique["timestamp"] = pd.to_datetime(unique["timestamp"], errors="coerce", utc=True, format="mixed")
    if unique.duplicated(["server_id", "timestamp"]).any():
        errors.append("conflicting records exist for the same server_id and timestamp")
    report = {
        "rows": len(data), "unique_servers": data.server_id.nunique(),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_values": int(data.drop(columns=["failure_event_id", "actual_failure_timestamp"]).isna().sum().sum()),
        "expected_null_event_metadata": int(data.loc[data[TARGET].eq(0), ["failure_event_id", "actual_failure_timestamp"]].isna().sum().sum()),
        "failure_events": int(data.failure_event_id.nunique()),
        "positive_rate": float(data[TARGET].eq(1).mean()),
    }
    if verbose:
        print("Validation report")
        for key, value in report.items():
            print(f"  {key}: {value}")
        if report["missing_values"] or report["duplicate_rows"]:
            print("  Exact duplicates and missing telemetry will be handled during preprocessing/model fitting.")
    if errors:
        raise ValueError("Data validation failed:\n- " + "\n- ".join(errors))
    if verbose:
        print("Validation passed.")
    return report


def validate_failure_metadata(data: pd.DataFrame) -> None:
    """Validate incident consistency; null metadata is expected on healthy rows."""
    required = ["timestamp", "server_id", TARGET, *FAILURE_METADATA_COLUMNS]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing failure metadata columns: {missing}")
    warning = data[TARGET].eq(1)
    times = pd.to_datetime(data.timestamp, errors="coerce", utc=True, format="mixed")
    failure_times = pd.to_datetime(data.actual_failure_timestamp, errors="coerce", utc=True, format="mixed")
    ids_present = data.failure_event_id.notna() & data.failure_event_id.astype(str).str.strip().ne("")
    if (warning & (~ids_present | failure_times.isna())).any():
        raise ValueError("Every positive row needs a failure_event_id and valid actual_failure_timestamp.")
    if data.loc[~warning, ["failure_event_id", "actual_failure_timestamp"]].notna().any().any():
        raise ValueError("Non-warning rows must have null failure event metadata.")
    lead_hours = (failure_times - times).dt.total_seconds() / 3600
    if not (lead_hours.loc[warning].gt(0) & lead_hours.loc[warning].le(24)).all():
        raise ValueError("Failure timestamps must be later than warning timestamps and at most 24 hours ahead.")
    events = data.loc[warning, ["failure_event_id", "server_id", "failure_type"]].copy()
    events["actual_failure_timestamp"] = failure_times.loc[warning]
    if events.server_id.isna().any() or not events.failure_type.isin(FAILURE_TYPES).all():
        raise ValueError("Failure events need a server and a valid failure type.")
    if (events.groupby("failure_event_id").nunique(dropna=False) > 1).any().any():
        raise ValueError("Each failure_event_id must identify one server, timestamp, and failure type.")
    incidents = events.drop_duplicates()
    if incidents.duplicated(["server_id", "actual_failure_timestamp"]).any():
        raise ValueError("One hardware incident cannot have multiple failure_event_ids.")


def main() -> dict:
    if not RAW_PATH.exists():
        raise FileNotFoundError("Generate data first: python -m src.generate_data")
    return validate_data(pd.read_csv(RAW_PATH))


if __name__ == "__main__":
    main()
