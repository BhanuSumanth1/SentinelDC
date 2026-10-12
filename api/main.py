"""Read-only API over a frozen historical research benchmark."""
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from api import services
from api.schemas import Health, Summary, Server, ServerDetail, Incident, ModelPerformance
from dashboard.formatting import DISCLAIMER, RISK_ORDER

app = FastAPI(title='SentinelDC', version='1.0.0', description=DISCLAIMER)


@app.exception_handler(Exception)
async def artifact_error(request, exc):
    logging.getLogger(__name__).error('Artifact request failed', exc_info=exc)
    return JSONResponse(status_code=500, content={'detail': 'Saved artifacts could not be loaded or validated. Check server logs.'})


@app.get('/health', response_model=Health)
def health():
    return {'status': 'ok', 'project': 'SentinelDC', 'version': '1.0.0'}


@app.get('/summary', response_model=Summary)
def summary():
    return services.summary()


@app.get('/servers', response_model=list[Server])
def servers(risk_level: str | None = None, rack_id: str | None = None, switch_id: str | None = None):
    if risk_level is not None and risk_level not in RISK_ORDER:
        raise HTTPException(400, 'risk_level must be LOW, MEDIUM, HIGH or CRITICAL')
    return services.servers(risk_level, rack_id, switch_id)


@app.get('/servers/{server_id}', response_model=ServerDetail)
def server(server_id: str):
    try:
        return services.server_detail(server_id)
    except KeyError:
        raise HTTPException(404, 'Server not found') from None


@app.get('/incidents', response_model=list[Incident])
def incidents():
    return services.incidents()


@app.get('/incidents/{incident_id}', response_model=Incident)
def incident(incident_id: str):
    try:
        return services.incident_detail(incident_id)
    except KeyError:
        raise HTTPException(404, 'Incident not found') from None


@app.get('/model-performance', response_model=ModelPerformance)
def model_performance():
    return services.model_performance()
