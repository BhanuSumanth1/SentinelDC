"""Generate hourly telemetry with gradual, type-specific fault episodes."""
import argparse

import numpy as np
import pandas as pd

from src.config import FAILURE_TYPES, RAW_PATH, REQUIRED_COLUMNS, SEED, TARGET, ensure_directories


def generate_telemetry(n_servers: int = 500, days: int = 30, seed: int = SEED) -> pd.DataFrame:
    """Simulate fault trajectories and label actual future events in (t, t + 24h].

    Events are spaced per server; degradation starts 48 hours before an event.
    A 24-hour unobserved future tail ensures that final-day labels are known.
    This is an illustrative simulator, not an empirical reliability model.
    """
    if n_servers < 10 or days < 7:
        raise ValueError("Use at least 10 servers and 7 days for a meaningful rare-event dataset.")
    rng = np.random.default_rng(seed)
    hours = days * 24
    timestamps = pd.date_range("2025-01-01", periods=hours, freq="h")
    time = np.arange(hours)
    frames = []
    # One episode per selected server keeps the 30-day default near 2% positives.
    affected_count = max(1, min(n_servers, round(n_servers * hours * 0.02 / 24)))
    affected = set(rng.choice(n_servers, size=affected_count, replace=False).tolist())
    for server in range(n_servers):
        load = np.empty(hours)
        load[0] = rng.normal()
        innovations = rng.normal(0, 0.65, hours)
        for index in range(1, hours):
            load[index] = 0.88 * load[index - 1] + innovations[index]
        daily = np.sin(2 * np.pi * (time % 24 - 8) / 24)
        cpu = rng.uniform(25, 55) + 13 * daily + 6 * load + rng.normal(0, 5, hours)
        memory = rng.uniform(35, 65) + 3 * daily + 3 * load + rng.normal(0, 4, hours)
        disk = rng.uniform(35, 75) + time * rng.uniform(0, 0.015) + rng.normal(0, 3, hours)
        gpu = rng.uniform(5, 40) + 6 * daily + rng.normal(0, 8, hours)
        temperature = 32 + 0.36 * cpu + rng.normal(0, 3, hours)
        packet_loss = rng.gamma(0.7, 0.13, hours)
        latency = 1.5 + 0.045 * np.maximum(cpu, 0) + rng.gamma(2, 0.7, hours)
        smart = rng.poisson(0.12, hours).astype(float)
        power = 100 + 2.8 * cpu + 1.5 * gpu + rng.normal(0, 12, hours)
        initial_uptime = rng.uniform(10, 360)
        uptime = initial_uptime + time / 24
        label = np.zeros(hours, dtype=int)
        types = np.full(hours, "none", dtype=object)
        event_ids = np.full(hours, None, dtype=object)
        failure_times = np.full(hours, np.datetime64("NaT"), dtype="datetime64[ns]")

        # Short benign spikes create hard negatives: high readings need not imply failure.
        benign = rng.random(hours) < 0.025
        cpu += benign * rng.uniform(5, 25, hours)
        temperature += benign * rng.uniform(4, 14, hours)
        latency += benign * rng.uniform(2, 15, hours)

        if server in affected:
            event_hour = int(rng.integers(48, hours + 24))
            kind = str(rng.choice(FAILURE_TYPES, p=[0.30, 0.25, 0.20, 0.10, 0.15]))
            remaining = event_hour - time
            precursor = (remaining > 0) & (remaining <= 48)
            severity = np.where(precursor, (49 - remaining) / 48, 0.0)
            severity *= rng.uniform(0.7, 1.25)
            positive = (remaining > 0) & (remaining <= 24)
            label[positive] = 1
            types[positive] = kind
            # One incident ID shared by all observations in its warning window.
            event_ids[positive] = f"failure_{server:06d}"
            failure_times[positive] = (timestamps[0] + pd.Timedelta(hours=event_hour)).to_datetime64()
            # Shared stress plus distinct signatures; neither labels nor type are features.
            cpu += 10 * severity
            memory += 8 * severity
            temperature += 9 * severity
            disk += 8 * severity
            packet_loss += 0.5 * severity
            latency += 7 * severity
            smart += rng.poisson(2 * severity)
            if kind == "disk_failure":
                disk += 20 * severity
                smart += rng.poisson(18 * severity ** 2)
            elif kind == "overheating":
                temperature += 26 * severity
                cpu += 15 * severity
            elif kind == "network_failure":
                packet_loss += 5 * severity ** 2
                latency += 65 * severity
            elif kind == "power_failure":
                power += 190 * severity + rng.normal(0, 45, hours) * severity
                temperature += 9 * severity
            else:
                memory += 30 * severity
                cpu += 18 * severity
            repaired = time >= event_hour
            uptime[repaired] = (time[repaired] - event_hour) / 24

        cpu, memory, disk, gpu = [np.clip(x, 0, 100) for x in (cpu, memory, disk, gpu)]
        rack = server // 20
        frame = pd.DataFrame({
            "timestamp": timestamps,
            "datacenter_id": f"dc_{server // 100:02d}",
            "rack_id": f"rack_{rack:03d}",
            "server_id": f"server_{server:04d}",
            "switch_id": f"switch_{rack:03d}",
            "cpu_utilization": cpu, "memory_utilization": memory,
            "disk_utilization": disk, "gpu_utilization": gpu,
            "temperature_c": np.clip(temperature, 15, 110),
            "disk_read_mb_s": np.maximum(0, 2.2 * cpu + rng.normal(0, 25, hours)),
            "disk_write_mb_s": np.maximum(0, 1.1 * cpu + rng.normal(0, 15, hours)),
            "network_in_mb_s": np.maximum(0, 1.7 * cpu + rng.normal(0, 20, hours)),
            "network_out_mb_s": np.maximum(0, 1.2 * cpu + rng.normal(0, 15, hours)),
            "packet_loss_pct": np.clip(packet_loss, 0, 100),
            "network_latency_ms": np.maximum(0.1, latency),
            "smart_error_count": smart.astype(int),
            "fan_speed_rpm": np.clip(1400 + 65 * (temperature - 30) + rng.normal(0, 150, hours), 500, 8000),
            "power_watts": np.clip(power, 30, 1000),
            "uptime_days": uptime,
            TARGET: label, "failure_type": types,
            "failure_event_id": event_ids, "actual_failure_timestamp": failure_times,
        })
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["timestamp", "server_id"], kind="stable"
    ).reset_index(drop=True)
    return result[REQUIRED_COLUMNS].round(3)


def main(n_servers: int = 500, days: int = 30, seed: int = SEED) -> pd.DataFrame:
    """Generate, save, and summarize the raw dataset."""
    ensure_directories()
    data = generate_telemetry(n_servers, days, seed)
    data.to_csv(RAW_PATH, index=False)
    print(f"Rows: {len(data):,} | Servers: {data.server_id.nunique():,}")
    print(f"Positive telemetry rows: {data[TARGET].sum():,} ({data[TARGET].mean():.2%})")
    print(f"Distinct failure incidents with observed warning rows: {data.failure_event_id.nunique():,}")
    print(f"Date range: {data.timestamp.min()} to {data.timestamp.max()}")
    print(f"Saved: {RAW_PATH}")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--servers", type=int, default=500)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    main(args.servers, args.days, args.seed)
