"""Causal per-server persistence/cooldown, separate from outcome matching."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.config import HORIZON_HOURS, TARGET
from src.event_evaluation import analyze_failure_events, save_event_figures


@dataclass(frozen=True)
class AlertPolicy:
    threshold: float
    emergency_threshold: float
    persistence_hours: int = 2
    cooldown_hours: float = 6.0

    def __post_init__(self):
        if not 0 < self.threshold < self.emergency_threshold < 1:
            raise ValueError("Require 0 < threshold < emergency threshold < 1.")
        if self.persistence_hours < 2 or int(self.persistence_hours) != self.persistence_hours:
            raise ValueError("Persistence must be an integer of at least two hourly readings.")
        if not np.isfinite(self.cooldown_hours) or self.cooldown_hours < 0:
            raise ValueError("Cooldown must be finite and nonnegative.")


def apply_alert_policy(observations, scores, policy):
    """Only server, timestamp and scores enter; sort before using past state.

    A missing hour or below-threshold observation resets persistence. At exactly
    cooldown expiry a fresh alert is allowed. State starts empty for each cohort.
    """
    data = observations[["server_id", "timestamp"]].copy()
    probabilities = np.asarray(scores, dtype=float)
    if probabilities.shape != (len(data),) or not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("Supply one finite probability in [0, 1] per observation.")
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True, errors="raise")
    if data.timestamp.isna().any() or data.server_id.isna().any() or data.duplicated(["server_id", "timestamp"]).any():
        raise ValueError("Observations need valid, unique server/timestamp keys.")
    data["calibrated_failure_probability"] = probabilities
    data["observation_position"] = np.arange(len(data))
    data = data.sort_values(["server_id", "timestamp"], kind="stable")
    alerts, reasons, streaks = [], [], []
    previous_server, previous_time, last_alert, streak = None, None, None, 0
    hour = pd.Timedelta(hours=1)
    cooldown = pd.Timedelta(hours=policy.cooldown_hours)
    for row in data.itertuples(index=False):
        if row.server_id != previous_server:
            previous_time, last_alert, streak = None, None, 0
        exceeds = row.calibrated_failure_probability >= policy.threshold
        streak = (streak + 1 if previous_time is not None and row.timestamp - previous_time == hour else 1) if exceeds else 0
        emergency = row.calibrated_failure_probability >= policy.emergency_threshold
        eligible = emergency or streak >= policy.persistence_hours
        suppressed = eligible and last_alert is not None and row.timestamp - last_alert < cooldown
        alert = eligible and not suppressed
        reason = ("emergency" if emergency else "persistence") if alert else ("cooldown" if suppressed else "none")
        if alert:
            last_alert = row.timestamp
        alerts.append(alert); reasons.append(reason); streaks.append(streak)
        previous_server, previous_time = row.server_id, row.timestamp
    data["raw_positive_prediction"] = data.calibrated_failure_probability >= policy.threshold
    data["consecutive_high_observations"] = streaks
    data["operational_alert"] = alerts
    data["alert_reason"] = reasons
    return data.sort_values("observation_position").reset_index(drop=True)


def evaluate_operational_alerts(test, scored, report_dir, figure_dir):
    """Match alerts retrospectively; metadata never enters the alert policy.

    True alert: same server, [failure-horizon, failure). Multiple true alerts
    for one incident remain workload; event detection/lead time use its first
    alert only. Report redundant true alerts explicitly, not as extra incidents.
    """
    # Binary alerts let the existing incident scorer retain its exact cohort,
    # partial-window flags and first-alert-only matching semantics.
    flags = scored.operational_alert.to_numpy(dtype=float)
    events, summary = analyze_failure_events(test, flags, 0.5)
    events = events.drop(columns=["max_model_risk_score_before_failure"])
    alerts = scored[scored.operational_alert].copy()
    matched = []
    catalog = events.sort_values(["actual_failure_timestamp", "failure_event_id"])
    by_server = {server: group for server, group in catalog.groupby("server_id")}
    for row in alerts.itertuples(index=False):
        candidates = by_server.get(row.server_id)
        event_id = None
        if candidates is not None:
            delta = candidates.actual_failure_timestamp - row.timestamp
            within = candidates[(delta > pd.Timedelta(0)) & (delta <= pd.Timedelta(hours=HORIZON_HOURS))]
            if not within.empty:
                event_id = within.iloc[0].failure_event_id
        matched.append(event_id)
    alerts["matched_failure_event_id"] = matched
    alerts["true_operational_alert"] = alerts.matched_failure_event_id.notna()
    alerts = alerts.sort_values(["timestamp", "server_id"])
    alerts["first_alert_for_event"] = alerts.true_operational_alert & ~alerts.matched_failure_event_id.duplicated()
    total = len(alerts)
    true = int(alerts.true_operational_alert.sum())
    false = total - true
    healthy = int(test[TARGET].eq(0).sum())
    # Each unique hourly row represents one server-hour of monitored exposure.
    server_days = len(test) / 24.0
    metrics = {**summary, "total_operational_alerts": total,
        "true_operational_alerts": true, "false_operational_alerts": false,
        "operational_alert_precision": true / total if total else np.nan,
        "redundant_true_operational_alerts": true - int(alerts.first_alert_for_event.sum()),
        "false_operational_alerts_per_1000_healthy_observations": 1000 * false / healthy if healthy else np.nan,
        "false_operational_alerts_per_server_day": false / server_days if server_days else np.nan,
        "observed_server_days": server_days, "healthy_telemetry_observations": healthy,
        "server_day_denominator": "observed_hourly_rows_divided_by_24", "alert_state_at_test_start": "empty"}
    assert int(alerts.first_alert_for_event.sum()) == summary["detected_failure_events"]
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(report_dir / "operational_alert_metrics.csv", index=False)
    alerts.to_csv(report_dir / "operational_alerts.csv", index=False)
    scored.to_csv(report_dir / "test_predictions_and_alerts.csv", index=False)
    events.to_csv(report_dir / "failure_event_analysis.csv", index=False)
    pd.DataFrame([summary]).to_csv(report_dir / "failure_lead_time_summary.csv", index=False)
    save_event_figures(test, events, figure_dir)
    return metrics
