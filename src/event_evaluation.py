"""Incident detection and warning lead time from cached chronological test scores.

Only warnings in an incident's labeled [failure - 24h, failure) window count,
including the exact 24h boundary. No fitting or prediction occurs here.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURE_DIR, REPORT_DIR, TARGET
from src.data_validation import validate_failure_metadata

EVENT_COLUMNS = [
    "failure_event_id", "server_id", "failure_type", "actual_failure_timestamp",
    "detected", "first_alert_timestamp", "lead_time_hours",
    "max_model_risk_score_before_failure", "observed_warning_rows",
    "observed_max_lead_time_hours", "complete_24h_warning_window",
    "failure_after_test_end",
]


def analyze_failure_events(test: pd.DataFrame, scores, threshold: float) -> tuple[pd.DataFrame, dict]:
    """Reduce test warning rows to incidents; never use validation or train alerts.

    Lead-time statistics condition on detected incidents. Early-detection
    percentages instead use ALL incidents with at least one test warning row.
    Known simulated failures in the future tail remain part of this cohort.
    """
    validate_failure_metadata(test)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(test):
        raise ValueError("Supply exactly one ordered risk score per test observation.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Risk scores must be finite numbers between 0 and 1.")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("The decision threshold must be between 0 and 1.")
    data = test.copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True, errors="raise")
    data["actual_failure_timestamp"] = pd.to_datetime(data.actual_failure_timestamp, utc=True, errors="coerce")
    if data.timestamp.isna().any() or data.duplicated(["server_id", "timestamp"]).any():
        raise ValueError("Test observations need valid, unique server/timestamp keys.")
    data["model_risk_score"] = scores  # Positional assignment matches model output.
    rows = []
    for event_id, group in data[data[TARGET].eq(1)].groupby("failure_event_id", sort=True):
        group = group.sort_values("timestamp")
        failure_time = group.actual_failure_timestamp.iloc[0]
        alerts = group[group.model_risk_score >= threshold]
        detected = not alerts.empty
        first_alert = alerts.timestamp.iloc[0] if detected else pd.NaT
        lead = (failure_time - first_alert).total_seconds() / 3600 if detected else np.nan
        expected = pd.date_range(failure_time - pd.Timedelta(hours=24), periods=24, freq="h")
        rows.append({
            "failure_event_id": event_id, "server_id": group.server_id.iloc[0],
            "failure_type": group.failure_type.iloc[0], "actual_failure_timestamp": failure_time,
            "detected": detected, "first_alert_timestamp": first_alert, "lead_time_hours": lead,
            "max_model_risk_score_before_failure": float(group.model_risk_score.max()),
            "observed_warning_rows": len(group),
            "observed_max_lead_time_hours": (failure_time - group.timestamp.min()).total_seconds() / 3600,
            "complete_24h_warning_window": bool(expected.isin(group.timestamp).all()),
            "failure_after_test_end": bool(failure_time > data.timestamp.max()),
        })
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    leads = events.loc[events.detected.eq(True), "lead_time_hours"].astype(float)
    total, detected = len(events), len(leads)
    summary = {
        "total_failure_events": total, "detected_failure_events": detected,
        "missed_failure_events": total - detected,
        "event_detection_rate": detected / total if total else np.nan,
        "mean_lead_time_hours": float(leads.mean()), "median_lead_time_hours": float(leads.median()),
        "min_lead_time_hours": float(leads.min()), "max_lead_time_hours": float(leads.max()),
        "p25_lead_time_hours": float(leads.quantile(0.25)), "p75_lead_time_hours": float(leads.quantile(0.75)),
        "complete_warning_window_events": int(events.complete_24h_warning_window.sum()),
        "partial_warning_window_events": int((events.complete_24h_warning_window == False).sum()),
        "events_after_test_end": int(events.failure_after_test_end.sum()),
        "early_detection_denominator": "all_failure_events_in_test_warning_rows",
        "lead_time_statistics_population": "detected_events_only",
    }
    for hours in [24, 12, 6, 3]:
        count = int(leads.ge(hours).sum())
        summary[f"detected_ge_{hours}h_count"] = count
        summary[f"detected_ge_{hours}h_pct"] = 100 * count / total if total else np.nan
    return events, summary


def save_event_evaluation(test: pd.DataFrame, scores, threshold: float,
                          report_dir: Path = REPORT_DIR, figure_dir: Path = FIGURE_DIR) -> dict:
    """Persist incident analysis and figures from already-computed test scores."""
    events, summary = analyze_failure_events(test, scores, threshold)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(report_dir / "failure_event_analysis.csv", index=False)
    pd.DataFrame([summary]).to_csv(report_dir / "failure_lead_time_summary.csv", index=False)
    save_event_figures(test, events, figure_dir)
    return summary


def save_event_figures(test: pd.DataFrame, events: pd.DataFrame, figure_dir: Path) -> None:
    """Show lead times, type-level detection, and one deterministic event example."""
    detected = events[events.detected.eq(True)]
    fig, ax = plt.subplots(figsize=(8, 5))
    if detected.empty:
        ax.text(0.5, 0.5, "No detected failure events; lead time is undefined", ha="center", transform=ax.transAxes)
    else:
        ax.hist(detected.lead_time_hours.astype(float), bins=np.arange(0, 26, 2), color="#327c96", edgecolor="white")
        ax.axvline(detected.lead_time_hours.median(), linestyle="--", color="#b64b26", label="Median")
        ax.legend()
    ax.set(xlabel="Warning lead time (hours)", ylabel="Detected failure events", xlim=(0, 24),
           title="FINAL TEST: warning lead times\nDetected events only; available test warning windows")
    fig.tight_layout()
    fig.savefig(figure_dir / "failure_lead_time_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if events.empty:
        ax.text(0.5, 0.5, "No failure events in test", ha="center", transform=ax.transAxes)
    else:
        types = events.groupby("failure_type").detected.agg(["sum", "count"])
        rates = 100 * types["sum"] / types["count"]
        bars = ax.bar(types.index, rates, color="#327c96")
        for bar, (_, row) in zip(bars, types.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{int(row['sum'])}/{int(row['count'])}", ha="center")
        ax.tick_params(axis="x", rotation=20)
    ax.set(ylabel="Events detected (%)", ylim=(0, 115), title="FINAL TEST: detection by failure type\nAnnotations: detected / actual events")
    fig.tight_layout()
    fig.savefig(figure_dir / "detection_by_failure_type.png", dpi=150)
    plt.close(fig)
    save_example_timeline(test, detected, figure_dir)


def select_timeline_event(test: pd.DataFrame, detected: pd.DataFrame) -> pd.Series:
    """Prefer the earliest detected event with 48h of test context, not best scores."""
    ordered = detected.sort_values(["actual_failure_timestamp", "failure_event_id"])
    times = pd.to_datetime(test.timestamp, utc=True)
    for _, candidate in ordered.iterrows():
        failure = pd.Timestamp(candidate.actual_failure_timestamp)
        available = times[(test.server_id == candidate.server_id) &
                          (times >= failure - pd.Timedelta(hours=48)) & (times < failure)]
        expected = pd.date_range(failure - pd.Timedelta(hours=48), periods=48, freq="h")
        if expected.isin(available).all():
            return candidate
    return ordered.iloc[0]  # Short datasets: honestly show the available partial context.


def save_example_timeline(test: pd.DataFrame, detected: pd.DataFrame, figure_dir: Path) -> None:
    """Show a full trajectory where available, without ranking warning quality."""
    if detected.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, "No detected event available for a timeline", ha="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        example = select_timeline_event(test, detected)
        failure = pd.Timestamp(example.actual_failure_timestamp)
        times = pd.to_datetime(test.timestamp, utc=True)
        window = test[(test.server_id == example.server_id) & (times >= failure - pd.Timedelta(hours=48)) & (times < failure)].copy()
        window["hours_before_failure"] = (failure - pd.to_datetime(window.timestamp, utc=True)).dt.total_seconds() / 3600
        window = window.sort_values("hours_before_failure", ascending=False)
        signals = [("temperature_c", "Temperature (°C)"), ("smart_error_count", "SMART errors"),
                   ("packet_loss_pct", "Packet loss (%)"), ("network_latency_ms", "Latency (ms)")]
        fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        for ax, (column, label) in zip(axes, signals):
            ax.plot(window.hours_before_failure, window[column], marker=".", color="#327c96")
            ax.axvline(example.lead_time_hours, color="#c68216", linestyle="--", label="First test alert")
            ax.axvline(0, color="#b83232", linestyle="-", label="Actual failure")
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
        axes[0].legend(loc="upper left", fontsize=8)
        axes[-1].set(xlabel="Hours before actual failure (test observations only)", xlim=(max(24, window.hours_before_failure.max()), -1))
        fig.suptitle(f"FINAL TEST: {example.failure_event_id} · {example.failure_type}\n"
                     f"{example.server_id} | Failure: {failure.strftime('%Y-%m-%d %H:%M UTC')} | "
                     f"First alert: {example.lead_time_hours:.0f}h before failure", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figure_dir / "example_failure_timeline.png", dpi=150)
    plt.close(fig)


def print_event_summary(summary: dict) -> None:
    """Display undefined rates/lead times honestly when there are no events/alerts."""
    def display(value, suffix=""):
        return f"{value:.2f}{suffix}" if pd.notna(value) else "N/A"

    print("\nFAILURE EVENT ANALYSIS")
    print(f"Actual failure events: {summary['total_failure_events']:,}")
    print(f"Detected failure events: {summary['detected_failure_events']:,}")
    print(f"Missed failure events: {summary['missed_failure_events']:,}")
    print(f"Event detection rate: {display(summary['event_detection_rate'] * 100, '%')}")
    print("\nFAILURE WARNING LEAD TIME (detected events)")
    for label, key in [("Median lead time", "median"), ("Mean lead time", "mean"),
                       ("25th percentile", "p25"), ("75th percentile", "p75")]:
        print(f"{label}: {display(summary[key + '_lead_time_hours'], ' hours')}")
    for hours in [12, 6]:
        print(f"Detected at least {hours} hours early: {summary[f'detected_ge_{hours}h_count']} "
              f"({display(summary[f'detected_ge_{hours}h_pct'], '%')} of all test events)")
