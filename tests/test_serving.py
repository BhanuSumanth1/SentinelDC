"""Read-only serving tests. No Streamlit server or model execution."""
from pathlib import Path
import hashlib
import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api import services
from dashboard import data_loader as data
from dashboard.app import PAGES

client = TestClient(app)


def test_dashboard_loader_snapshot_join_and_cached_copy():
    frame = data.snapshot()
    assert len(frame) == 500 and frame.server_id.is_unique
    assert frame.datacenter_id.nunique() == 5
    assert frame.top_risk_driver.notna().all()
    frame.loc[0, 'server_id'] = 'MUTATION'
    assert 'MUTATION' not in data.snapshot().server_id.values


def test_snapshot_rejects_misaligned_explanations(monkeypatch):
    original = data.csv
    def wrong(name):
        table = original(name)
        if name == 'explainability_live_view.csv': table.loc[0, 'timestamp'] = '2099-01-01 23:00:00+00:00'
        return table
    monkeypatch.setattr(data, 'csv', wrong)
    with pytest.raises(ValueError, match='keys'):
        data.snapshot()


def test_health_summary_and_docs():
    assert client.get('/health').json() == {'status': 'ok', 'project': 'SentinelDC', 'version': '1.0.0'}
    response = client.get('/summary')
    assert response.status_code == 200
    assert response.json()['servers_monitored'] == 500
    saved = data.csv('hybrid_test_metrics.csv').iloc[0]
    assert response.json()['hybrid_detected_failure_events'] == saved.detected_failure_events
    assert client.get('/docs').status_code == 200


def test_server_endpoint_and_filters():
    response = client.get('/servers', params={'risk_level': 'HIGH', 'rack_id': 'rack_000'})
    assert response.status_code == 200
    assert all(r['hybrid_risk_level'] == 'HIGH' and r['rack_id'] == 'rack_000' for r in response.json())
    response = client.get('/servers/server_0038')
    assert response.status_code == 200 and response.json()['top_explanation_drivers']
    assert client.get('/servers', params={'risk_level': 'INVALID'}).status_code == 400


def test_incident_endpoints_and_performance():
    response = client.get('/incidents')
    assert response.status_code == 200 and len(response.json()) == len(data.csv('root_cause_incidents.csv'))
    response = client.get('/incidents/INCIDENT_0001')
    assert response.status_code == 200 and response.json()['candidates']
    assert client.get('/model-performance').json()['root_cause_signature_evaluation']['correct_signatures'] == 0


@pytest.mark.parametrize('url', ['/servers/missing', '/incidents/missing'])
def test_missing_id_returns_404(url):
    assert client.get(url).status_code == 404


def test_artifact_failure_returns_500(monkeypatch):
    def broken(): raise FileNotFoundError('missing report')
    monkeypatch.setattr(services, 'summary', broken)
    with TestClient(app, raise_server_exceptions=False) as failing:
        response = failing.get('/summary')
    assert response.status_code == 500 and 'Saved artifacts' in response.json()['detail']


def test_required_files_and_readme_sections():
    for file in ['dashboard/__init__.py', 'dashboard/app.py', 'dashboard/data_loader.py',
                 'dashboard/components.py', 'dashboard/formatting.py', 'Dockerfile', 'docker-compose.yml',
                 '.dockerignore', 'docs/ARCHITECTURE.md', 'docs/DEMO_GUIDE.md',
                 'docs/INTERVIEW_GUIDE.md', 'docs/RESUME_BULLETS.md']:
        assert (data.ROOT/file).is_file()
    readme = (data.ROOT/'README.md').read_text()
    for title in ['Problem', 'Why This Matters', 'System Architecture', 'Key Features', 'Dataset',
                  'ML Methodology', 'Detection Systems', 'Root-Cause Intelligence', 'Explainability',
                  'Dashboard', 'API', 'Docker', 'Results', 'Known Limitations', 'Project Structure',
                  'Installation', 'Running Dashboard', 'Running API', 'Running Docker', 'Running Tests',
                  'Screenshots', 'Future Work']:
        assert '## '+title+'\n' in readme


def test_serving_preserves_frozen_artifacts():
    expected = data.json_file('reports/final_project_protected_artifacts.json')
    for path, digest in expected.items():
        assert hashlib.sha256((data.ROOT/path).read_bytes()).hexdigest() == digest, path


@pytest.mark.parametrize('page', PAGES)
def test_dashboard_page_app_smoke(page):
    # Streamlit's simulated runner exercises rendering without launching a server.
    from streamlit.testing.v1 import AppTest
    application = AppTest.from_file(str(data.ROOT/'dashboard/app.py'), default_timeout=30).run()
    application.sidebar.radio[0].set_value(page).run()
    assert not application.exception
    assert not application.error
