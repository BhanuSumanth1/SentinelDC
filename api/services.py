"""Services reuse validated artifact loaders; no ML imports."""
from dashboard import data_loader as data
from dashboard.formatting import LIMITATIONS, RISK_ORDER

summary = data.summary
server_detail = data.server_detail


def servers(risk_level=None, rack_id=None, switch_id=None):
    if risk_level is not None and risk_level not in RISK_ORDER:
        raise ValueError('risk_level must be LOW, MEDIUM, HIGH or CRITICAL')
    frame = data.snapshot()
    for col, value in [('hybrid_risk_level', risk_level), ('rack_id', rack_id), ('switch_id', switch_id)]:
        if value is not None:
            frame = frame[frame[col].eq(value)]
    return data.records(frame)


def incidents():
    return data.records(data.csv('root_cause_incidents.csv'))


def incident_detail(incident_id):
    for incident in data.json_lines('root_cause_details.jsonl'):
        if incident['incident_id'] == incident_id:
            return incident
    raise KeyError(incident_id)


def model_performance():
    return {'systems': data.performance(),
            'root_cause_signature_evaluation': data.json_file('reports/root_cause_signature_diagnostic.json'),
            'limitations': LIMITATIONS}
