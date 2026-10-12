"""Display-only formatting, with no model imports."""
RISK_ORDER = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
RISK_COLORS = {'LOW': '#319e88', 'MEDIUM': '#dba534', 'HIGH': '#ed8144', 'CRITICAL': '#d94565'}
DISCLAIMER = 'SentinelDC is a portfolio research prototype evaluated on hardened synthetic data-center telemetry. It is not a production monitoring system.'
LIMITATIONS = [
    'The benchmark uses hardened synthetic telemetry, not production data.',
    'Operational precision remains low and false alerts remain substantial.',
    'Root-cause signatures performed poorly; most incidents have UNKNOWN/low evidence.',
    'Exact shared infrastructure causal ground truth is unavailable. One switch per rack makes attribution ambiguous.',
    'SHAP explains model behavior; it does not prove causation.',
]
NAMES = {'temperature_change_6h': 'Temperature trend (6h)',
         'smart_error_change_12h': 'SMART error growth (12h)',
         'packet_loss_rolling_mean_6h': 'Average packet loss (6h)',
         'network_latency_ms': 'Network latency', 'cpu_rolling_mean_6h': 'Average CPU utilization (6h)',
         'memory_rolling_mean_6h': 'Average memory utilization (6h)', 'uptime_days': 'Server uptime'}

def friendly(value):
    return NAMES.get(value, str(value).replace('_', ' ').capitalize()) if value else 'Unavailable'

def percent(value):
    return f'{value:.2%}' if value is not None else 'Unavailable'
