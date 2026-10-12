"""Generate heterogeneous, noisy server telemetry and explicit failure incidents."""
import argparse

import numpy as np
import pandas as pd

from src.config import FAILURE_TYPES, RAW_PATH, REQUIRED_COLUMNS, SEED, TARGET, ensure_directories

GENERATOR_VERSION = "heterogeneous-v2"
ONSET_MEDIANS = {"disk_failure": 32, "overheating": 21, "network_failure": 14,
                 "power_failure": 9, "memory_failure": 26}
TRANSIENT_TYPES = ["cpu", "memory", "thermal", "network", "disk", "smart", "power", "fan"]


def pulse(length: int, rng: np.random.Generator) -> np.ndarray:
    """A noisy finite excursion that recovers, rather than a permanent step."""
    envelope = np.sin(np.linspace(0.15, np.pi - 0.15, length)) ** rng.uniform(0.4, 1.2)
    return np.maximum(0, envelope * rng.normal(1, 0.15, length))


def shared_conditions(n_racks: int, hours: int, rng: np.random.Generator):
    """Independent rack thermal/power and switch congestion episodes."""
    conditions, catalog = {}, []
    for rack in range(n_racks):
        heat, congestion = np.zeros(hours), np.zeros(hours)
        for kind, series in [("rack_heat", heat), ("switch_congestion", congestion)]:
            for _ in range(rng.poisson(hours / 90)):
                start = int(rng.integers(0, hours - 2))
                length = min(int(rng.integers(2, 11)), hours - start)
                series[start:start + length] += pulse(length, rng)
                catalog.append({"condition_id": f"shared_{len(catalog):06d}", "rack_id": f"rack_{rack:03d}",
                                "switch_id": f"switch_{rack:03d}", "kind": kind,
                                "start_hour": start, "duration_hours": length})
        conditions[rack] = (heat, congestion)
    return conditions, catalog


def degradation_profile(hours: int, event_hour: int, kind: str, rng: np.random.Generator):
    """Fault-specific onset, slope, stochastic strength and possible recovery."""
    onset = int(np.clip(np.rint(rng.lognormal(np.log(ONSET_MEDIANS[kind]), 0.48)), 4, 72))
    severity = float(np.clip(rng.lognormal(-0.05, 0.40), 0.35, 2.1))
    slope = float(rng.uniform(0.65, 2.4))
    remaining = event_hour - np.arange(hours)
    active = (remaining > 0) & (remaining <= onset)
    progress = np.clip((onset - remaining + 1) / onset, 0, 1) ** slope
    jitter = np.convolve(rng.normal(0, 0.24, hours + 4), np.ones(5) / 5, mode="valid")
    profile = np.where(active, severity * progress * np.clip(1 + jitter, 0.25, 1.8), 0)
    recovery = bool(rng.random() < 0.55)
    if recovery:
        remaining_start = int(rng.integers(2, max(3, onset)))
        duration = int(rng.integers(1, 7))
        recovering = (remaining <= remaining_start) & (remaining > remaining_start - duration)
        profile[recovering] *= rng.uniform(0.20, 0.65)
    return profile, {"degradation_onset_hours": onset, "severity": severity,
                     "slope_exponent": slope, "temporary_recovery": recovery}


def generate_telemetry(n_servers: int = 500, days: int = 30, seed: int = SEED) -> pd.DataFrame:
    """Simulate telemetry; labels remain actual events in (t, t + 24 hours].

    Sidecar diagnostics are returned in DataFrame.attrs and never become model
    inputs. Incident incidence is accelerated for a portfolio experiment.
    """
    if n_servers < 10 or days < 7:
        raise ValueError("Use at least 10 servers and 7 days for a meaningful rare-event dataset.")
    rng = np.random.default_rng(seed)
    hours = days * 24
    times = pd.date_range("2025-01-01", periods=hours, freq="h")
    t = np.arange(hours)
    affected = rng.random(n_servers) < min(0.95, days * 0.02)
    shared, shared_catalog = shared_conditions((n_servers + 19) // 20, hours, rng)
    frames, profiles, events, transients = [], [], [], []
    for server in range(n_servers):
        server_id, rack = f"server_{server:04d}", server // 20
        profile = {"server_id": server_id, "normal_cpu": rng.uniform(20, 78),
                   "normal_memory": rng.uniform(28, 86), "normal_temperature": rng.uniform(37, 76),
                   "normal_network_latency": np.clip(rng.lognormal(1.3, 0.6), 0.5, 20),
                   "normal_packet_loss": rng.uniform(0.005, 0.10) if rng.random() < 0.85 else rng.uniform(0.15, 0.65),
                   "normal_power": rng.uniform(160, 420), "normal_disk_activity": rng.uniform(20, 230),
                   "normal_fan_speed": rng.uniform(1500, 4800)}
        profiles.append(profile)
        daily = np.sin(2 * np.pi * (t % 24 - rng.uniform(0, 24)) / 24)
        load = np.empty(hours)
        load[0] = rng.normal()
        for i in range(1, hours):
            load[i] = 0.9 * load[i - 1] + rng.normal(0, 0.5)
        cpu = profile['normal_cpu'] + rng.uniform(4, 16) * daily + 5 * load + rng.normal(0, 4, hours)
        memory = profile['normal_memory'] + 2 * daily + 2.5 * load + rng.normal(0, 3, hours)
        gpu = rng.uniform(0, 45) + 5 * daily + rng.normal(0, 5, hours)
        disk = rng.uniform(25, 85) + 0.2 * load + rng.normal(0, 2, hours)
        activity = np.maximum(0, profile['normal_disk_activity'] * np.exp(0.10 * load) + rng.normal(0, 12, hours))
        smart = rng.poisson(rng.uniform(0.02, 0.4), hours).astype(float)
        heat, congestion = shared[rack]
        temperature_delta = rng.uniform(3, 11) * heat
        latency_delta = rng.uniform(5, 35) * congestion
        loss_delta = rng.uniform(0.3, 4) * congestion ** 1.3
        power_delta = rng.uniform(5, 35) * heat
        fan_delta = np.zeros(hours)
        throughput_factor = np.clip(1 - 0.2 * congestion, 0.2, 1)
        label = np.zeros(hours, dtype=int)
        types = np.full(hours, 'none', dtype=object)
        event_ids = np.full(hours, None, dtype=object)
        failure_times = np.full(hours, np.datetime64('NaT'), dtype='datetime64[ns]')
        event_hour, kind = None, None
        if affected[server]:
            event_hour = int(rng.integers(48, hours + 24))
            kind = str(rng.choice(FAILURE_TYPES, p=[0.28, 0.22, 0.18, 0.15, 0.17]))
            remaining = event_hour - t
            warning = (remaining > 0) & (remaining <= 24)
            label[warning], types[warning] = 1, kind
            event_ids[warning] = f'failure_{server:06d}'
            failure_time = times[0] + pd.Timedelta(hours=event_hour)
            failure_times[warning] = failure_time.to_datetime64()
        # Recovering non-causal disturbances are independent of the fault schedule.
        for _ in range(rng.poisson(days * 0.8)):
            start = int(rng.integers(0, hours - 2))
            length = min(int(rng.integers(2, 13)), hours - start)
            sl = slice(start, start + length)
            wave = pulse(length, rng)
            transient = str(rng.choice(TRANSIENT_TYPES))
            if transient == 'cpu':
                cpu[sl] += rng.uniform(25, 65) * wave
                memory[sl] += rng.uniform(0, 12) * wave
            elif transient == 'memory':
                memory[sl] += rng.uniform(20, 55) * wave
                cpu[sl] += rng.uniform(0, 18) * wave
            elif transient == 'thermal':
                temperature_delta[sl] += rng.uniform(8, 25) * wave
                fan_delta[sl] += rng.uniform(-600, 900) * wave
            elif transient == 'network':
                loss_delta[sl] += rng.uniform(0.5, 5) * wave
                latency_delta[sl] += rng.uniform(5, 45) * wave ** 1.2
                throughput_factor[sl] *= np.clip(1 - 0.4 * wave, 0.2, 1)
            elif transient == 'disk':
                activity[sl] += rng.uniform(100, 450) * wave
                disk[sl] += rng.uniform(5, 18) * wave
            elif transient == 'smart':
                smart[sl] += rng.poisson(rng.uniform(2, 12) * wave)
            elif transient == 'power':
                power_delta[sl] += rng.uniform(50, 180) * wave
                fan_delta[sl] += rng.uniform(-600, 600) * wave
            else:
                fan_delta[sl] += rng.uniform(-1300, 1700) * wave
            transients.append({"transient_id": f'transient_{len(transients):07d}', 'server_id': server_id,
                               'kind': transient, 'start_timestamp': times[start],
                               'end_timestamp': times[start + length - 1], 'duration_hours': length,
                               'warning_rows_during_transient': int(label[sl].sum()),
                               'fully_healthy_24h': bool(label[sl].sum() == 0)})
        if event_hour is not None:
            degradation, event_detail = degradation_profile(hours, event_hour, kind, rng)
            affected_features = []
            def affects(name, probability):
                selected = rng.random() < probability
                if selected:
                    affected_features.append(name)
                return selected
            if kind == 'disk_failure':
                if affects('smart_error_count', 0.85):
                    smart += rng.poisson(rng.uniform(4, 18) * degradation ** 1.4)
                if affects('disk_utilization', 0.75):
                    disk += rng.uniform(8, 30) * degradation
                if affects('disk_activity', 0.80):
                    activity *= np.clip(1 - rng.uniform(0.15, 0.65) * degradation, 0.08, 1.5)
                if affects('temperature_c', 0.35):
                    temperature_delta += rng.uniform(2, 8) * degradation
            elif kind == 'overheating':
                affected_features.append('temperature_c')
                temperature_delta += rng.uniform(12, 32) * degradation
                if affects('fan_speed_rpm', 0.75):
                    fan_delta += rng.choice([-1, 1]) * rng.uniform(400, 1800) * degradation
                if affects('cpu_utilization', 0.55):
                    cpu += rng.uniform(6, 25) * degradation
                    gpu += rng.uniform(0, 15) * degradation
            elif kind == 'network_failure':
                if affects('packet_loss_pct', 0.85):
                    loss_delta += rng.uniform(0.5, 5) * degradation ** rng.uniform(0.8, 1.8)
                if affects('network_latency_ms', 0.85):
                    latency_delta += rng.uniform(7, 50) * degradation
                if affects('network_throughput', 0.75):
                    throughput_factor *= np.clip(1 - rng.uniform(0.15, 0.65) * degradation, 0.05, 1)
            elif kind == 'memory_failure':
                affected_features.append('memory_utilization')
                memory += rng.uniform(10, 35) * degradation
                if affects('cpu_instability', 0.75):
                    cpu += (rng.uniform(3, 12) + rng.normal(0, 9, hours)) * degradation
            else:
                affected_features.append('power_watts')
                power_delta += (rng.uniform(-60, 90) + rng.normal(0, rng.uniform(35, 100), hours)) * degradation
                if affects('fan_speed_rpm', 0.70):
                    fan_delta += rng.normal(0, 600, hours) * degradation
                if affects('cpu_utilization', 0.45):
                    cpu -= rng.uniform(3, 15) * degradation
                if affects('temperature_c', 0.30):
                    temperature_delta += rng.uniform(-4, 5) * degradation
            events.append({'failure_event_id': f'failure_{server:06d}', 'server_id': server_id,
                           'failure_type': kind, 'actual_failure_timestamp': failure_time,
                           **event_detail, 'affected_signals': '|'.join(affected_features)})
        cpu, memory, gpu, disk = [np.clip(x, 0, 100) for x in (cpu, memory, gpu, disk)]
        cpu_change = cpu - profile['normal_cpu']
        io_change = activity - profile['normal_disk_activity']
        temperature = profile['normal_temperature'] + 0.10 * cpu_change + 0.0015 * np.sign(io_change) * np.abs(io_change) ** 1.25 + temperature_delta + rng.normal(0, 2.1, hours)
        power = profile['normal_power'] + 0.65 * np.sign(cpu_change) * np.abs(cpu_change) ** 1.3 + 0.08 * io_change + power_delta + rng.normal(0, 12, hours)
        fan = profile['normal_fan_speed'] + 45 * np.sign(temperature - profile['normal_temperature']) * np.abs(temperature - profile['normal_temperature']) ** 1.15 + fan_delta + rng.normal(0, 120, hours)
        packet_loss = rng.gamma(0.8, profile['normal_packet_loss'] / 0.8, hours) + loss_delta
        latency = profile['normal_network_latency'] + 0.01 * np.maximum(cpu_change, 0) ** 1.5 + latency_delta + rng.gamma(1.5, 0.5, hours)
        uptime = rng.uniform(10, 360) + t / 24
        if event_hour is not None:
            repaired = t >= event_hour
            uptime[repaired] = (t[repaired] - event_hour) / 24
        frame = pd.DataFrame({
            'timestamp': times, 'datacenter_id': f'dc_{server // 100:02d}', 'rack_id': f'rack_{rack:03d}',
            'server_id': server_id, 'switch_id': f'switch_{rack:03d}',
            'cpu_utilization': cpu, 'memory_utilization': memory, 'disk_utilization': disk, 'gpu_utilization': gpu,
            'temperature_c': np.clip(temperature, 10, 115),
            'disk_read_mb_s': np.maximum(0, activity + rng.normal(0, 12, hours)),
            'disk_write_mb_s': np.maximum(0, 0.5 * activity + rng.normal(0, 8, hours)),
            'network_in_mb_s': np.maximum(0, (40 + 1.4 * cpu + rng.normal(0, 14, hours)) * throughput_factor),
            'network_out_mb_s': np.maximum(0, (20 + 0.8 * cpu + rng.normal(0, 10, hours)) * throughput_factor),
            'packet_loss_pct': np.clip(packet_loss, 0, 100), 'network_latency_ms': np.maximum(0.1, latency),
            'smart_error_count': smart.astype(int), 'fan_speed_rpm': np.clip(fan, 200, 8500),
            'power_watts': np.clip(power, 20, 1300), 'uptime_days': uptime,
            TARGET: label, 'failure_type': types, 'failure_event_id': event_ids,
            'actual_failure_timestamp': failure_times,
        })
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True).sort_values(['timestamp', 'server_id'], kind='stable').reset_index(drop=True)
    result = result[REQUIRED_COLUMNS].round(3)
    result.attrs['server_profiles'] = profiles
    result.attrs['failure_event_catalog'] = events
    result.attrs['healthy_transients'] = transients
    result.attrs['shared_conditions'] = shared_catalog
    result.attrs['generator_version'] = GENERATOR_VERSION
    return result


def main(n_servers: int = 500, days: int = 30, seed: int = SEED) -> pd.DataFrame:
    """Save telemetry and separate simulation-only provenance catalogs."""
    ensure_directories()
    data = generate_telemetry(n_servers, days, seed)
    data.to_csv(RAW_PATH, index=False)
    for name in ['server_profiles', 'failure_event_catalog', 'healthy_transients', 'shared_conditions']:
        pd.DataFrame(data.attrs[name]).to_csv(RAW_PATH.parent / f'{name}.csv', index=False)
    print(f'Generator: {GENERATOR_VERSION} | Seed: {seed}')
    print(f'Rows: {len(data):,} | Servers: {data.server_id.nunique():,}')
    print(f'Positive telemetry rows: {data[TARGET].sum():,} ({data[TARGET].mean():.2%})')
    print(f'Actual failure incidents: {data.failure_event_id.nunique():,}')
    print(f'Date range: {data.timestamp.min()} to {data.timestamp.max()}')
    print(f'Saved: {RAW_PATH}')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--servers', type=int, default=500)
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    main(args.servers, args.days, args.seed)
