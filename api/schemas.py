"""Response contracts for frozen artifact serving."""
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict

class Health(BaseModel):
    status: str
    project: str
    version: str

class Summary(BaseModel):
    servers_monitored: int
    datacenters: int
    racks: int
    switches: int
    snapshot_timestamp: str
    snapshot_kind: str
    test_failure_events: int
    hybrid_detected_failure_events: int
    hybrid_detection_rate: float
    median_warning_lead_time_hours: float
    risk_counts: dict[str, int]

class Server(BaseModel):
    model_config = ConfigDict(extra='allow')
    server_id: str
    timestamp: str
    datacenter_id: str
    rack_id: str
    switch_id: str
    failure_probability: float
    anomaly_score: float
    anomaly_percentile: float
    hybrid_risk_level: Literal['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    operational_alert: bool

class ServerDetail(Server):
    supervised_threshold: float
    reason_codes: list[str]
    top_explanation_drivers: list[dict[str, Any]]
    explanation_note: str

class Incident(BaseModel):
    model_config = ConfigDict(extra='allow')
    incident_id: str
    severity: str
    suspected_component: str | None
    suspected_component_type: str
    root_cause_confidence_score: float
    telemetry_signature: str

class ModelPerformance(BaseModel):
    systems: list[dict[str, Any]]
    root_cause_signature_evaluation: dict[str, Any]
    limitations: list[str]
