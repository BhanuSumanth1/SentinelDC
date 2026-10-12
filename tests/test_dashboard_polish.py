"""Guided investigation flows and read-only saved explanation presentation."""
from streamlit.testing.v1 import AppTest
from dashboard import data_loader as data
from dashboard.design import CSS, saved_server_factors


def app():
    return AppTest.from_file(str(data.ROOT/'dashboard/app.py'),default_timeout=30).run()


def test_attention_button_opens_exact_server():
    view=app()
    button=next(button for button in view.button if button.label=='Investigate Server')
    server=button.key.removeprefix('investigate_')
    button.click().run()
    assert not view.exception and not view.error
    assert view.sidebar.radio[0].value=='Live Server Monitor'
    assert next(box for box in view.selectbox if box.label=='Inspect server').value==server


def test_guided_demo_and_empty_search():
    view=app()
    next(button for button in view.sidebar.button if button.label=='Investigate High-Risk Server').click().run()
    assert not view.exception
    next(text for text in view.text_input if text.label=='Search server ID').set_value('no-such-server').run()
    assert not view.exception and not view.error
    assert any('No servers match' in item.value for item in view.info)
    next(button for button in view.sidebar.button if button.label=='Investigate Incident').click().run()
    expected=data.json_file('reports/example_root_cause_incident.json')['incident_id']
    assert next(box for box in view.selectbox if box.label=='Select incident').value==expected
    assert not view.exception


def test_saved_shap_signs_and_reduced_motion():
    row=data.snapshot().iloc[0]
    positive,negative=saved_server_factors(row.server_id,row.timestamp)
    assert positive and negative
    assert all(f['shap_value']>0 for f in positive)
    assert all(f['shap_value']<0 for f in negative)
    assert positive[0]['feature']==row.top_risk_driver
    assert 'prefers-reduced-motion:reduce' in CSS
