"""Reusable presentation components for saved intelligence."""
import pandas as pd
import plotly.express as px
import streamlit as st
from dashboard.formatting import friendly, percent, RISK_COLORS
from dashboard import data_loader as data


def chart(figure):
    figure.update_layout(template='plotly_dark', font=dict(family='Arial', size=13, color='#c9d9eb'),
                         margin=dict(l=15, r=15, t=55, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(figure, width='stretch')


def metrics(items):
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.metric(label, value)


def factors(items):
    if not items:
        st.info('No saved factors available for this observation.')
        return
    rows = [{'Feature': item.get('display_name', friendly(item.get('feature'))),
             'Model input': item.get('value'), 'SHAP contribution': item.get('shap_value')}
            for item in items if item.get('feature') and item.get('shap_value') is not None]
    if not rows:
        st.info('No saved contributions are available.'); return
    frame = pd.DataFrame(rows)
    frame['Direction'] = frame['SHAP contribution'].apply(lambda v: 'Raises model risk' if v > 0 else 'Lowers model risk')
    figure = px.bar(frame.iloc[::-1], x='SHAP contribution', y='Feature', orientation='h', color='Direction',
                    color_discrete_map={'Raises model risk': '#d99372', 'Lowers model risk': '#75b9b0'},
                    hover_data=['Model input'], labels={'SHAP contribution': 'Contribution to model score'})
    figure.update_layout(height=max(180, 50*len(rows)+90), showlegend=False)
    chart(figure)
    st.caption('Right / warm bars raise the model score; left / teal bars lower it. Longer bars have more influence for this observation. Units are base-model log-odds, not probability changes or causal effects.')
    with st.expander('Exact saved feature values and contributions'):
        st.dataframe(frame, hide_index=True, width='stretch')


def saved_image(filename, caption):
    path = data.REPORTS/'figures'/filename
    if path.exists():
        st.image(str(path), caption=caption, width='stretch')
    else:
        st.info(f'Saved figure unavailable: {filename}')


def combined_panel():
    report = data.json_file('reports/example_sentinel_intelligence_report.json')
    st.subheader('One warning, multiple views')
    st.caption(f"Saved example • {report['server_id']} • {report['timestamp']}")
    supervised, anomaly, hybrid, root = [report[k] for k in ['supervised', 'anomaly', 'hybrid', 'root_cause_intelligence']]
    metrics([('Calibrated failure probability', percent(supervised['calibrated_failure_probability'])),
             ('Anomaly percentile', percent(anomaly['anomaly_percentile'])),
             ('Hybrid risk', hybrid['risk_level']),
             ('Root-cause hypothesis', root['telemetry_signature'] if root else 'Unavailable')])
    left, right = st.columns(2)
    with left:
        st.markdown('**Supervised model · SHAP risk drivers**')
        factors(supervised['top_risk_drivers'])
    with right:
        st.markdown('**Anomaly · largest robust deviations**')
        st.dataframe(pd.DataFrame(anomaly['robust_deviations']['largest_deviations']), hide_index=True, width='stretch')
        st.markdown('**Hybrid reason codes**')
        st.write(', '.join(hybrid['reason_codes']))
    if root:
        st.info(f"Root-cause: {root['evidence_status']} · inspection: {root['inspection_priority']} · confidence: {root['root_cause_confidence_score']:.3f}")
        st.caption(f"Connected servers: {root['connected_servers']} · additional exposure: {root['potential_additional_exposure']}. Hypothesis available at {root['inference_as_of']}.")
    st.caption(report['disclaimer'])
