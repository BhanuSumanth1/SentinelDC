"""Streamlit presentation of saved results. Importing this module starts no server."""
from pathlib import Path
import sys
# Streamlit executes this path directly; make the project package root available.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dashboard import data_loader as data
from dashboard.design import CSS, html, esc, cards, badge, thinking, gauge, saved_server_factors
from dashboard.components import chart, metrics, factors, saved_image, combined_panel
from dashboard.formatting import DISCLAIMER, LIMITATIONS, RISK_ORDER, RISK_COLORS, percent, friendly

PAGES = ['Executive Overview', 'Live Server Monitor', 'Failure Intelligence',
         'Incident / Root-Cause Explorer', 'Model Explainability', 'Model Performance', 'Architecture & Limitations']


def overview():
    summary = data.summary()
    html('<div class="sc-hero"><div class="sc-status"><span class="sc-dot"></span>VIEWER ONLINE · FROZEN RESEARCH SNAPSHOT</div><div class="sc-label" style="margin-top:20px">SENTINELDC / OPERATIONS COMMAND CENTER</div><h1>AI-powered data center<br>failure intelligence.</h1><p>Detect failures early. Understand unusual behavior. Explain model risk. Investigate shared infrastructure.</p><div class="sc-note">'+f"{summary['servers_monitored']} servers · {summary['datacenters']} data centers · {summary['racks']} racks · {summary['switches']} switches"+'</div></div>')
    thinking()
    total, found = summary['test_failure_events'], summary['hybrid_detected_failure_events']
    lead = summary['median_warning_lead_time_hours']
    st.subheader('What is SentinelDC seeing?')
    html(f'<div class="sc-story">The TEST benchmark includes <strong>{total} failure events</strong>. The hybrid system warned on <strong>{found}</strong>, with a median <strong>{lead:g}-hour early warning</strong> among detected events. <strong>{total-found} events remained undetected.</strong><br><span class="sc-note">High event coverage is only part of the story: false warnings remain substantial.</span></div>')
    cards([('▦ Servers monitored',summary['servers_monitored'],'Simulated infrastructure in the saved snapshot'),
           ('◷ Failure events',total,'Unique events represented in TEST warning windows'),
           ('✓ Failures detected',found,f'{found} of {total} TEST events received a warning'),
           ('◉ Detection rate',percent(summary['hybrid_detection_rate']),'Event coverage, not alert precision'),
           ('↗ Median warning',f'{lead:g} h','Time before failure, among detected events')])
    st.subheader('Requires Attention')
    attention = data.snapshot().loc[lambda frame: frame.hybrid_risk_level.isin(['HIGH','CRITICAL'])].head(3)
    if attention.empty: st.info('No HIGH or CRITICAL servers in the saved snapshot.')
    else:
        for column, (_,row) in zip(st.columns(len(attention)),attention.iterrows()):
            with column:
                html(f'<div class="sc-card"><div class="sc-label">{esc(row.server_id)}</div>{badge(row.hybrid_risk_level)}<h3>{percent(row.failure_probability)} failure probability</h3><div class="sc-note">Anomaly percentile: {percent(row.anomaly_percentile)}<br>Top driver: {esc(friendly(row.top_risk_driver))}<br>{esc(row.rack_id)} · {esc(row.switch_id)}</div></div>')
                st.button('Investigate Server',key='investigate_'+row.server_id,on_click=navigate,args=('Live Server Monitor',row.server_id),width='stretch')
    st.subheader('Risk across the fleet')
    metrics([(risk, summary['risk_counts'][risk]) for risk in RISK_ORDER])
    st.caption('Hybrid risk combines supervised failure evidence and anomaly severity. HIGH and CRITICAL prioritize investigation; they do not confirm failure.')
    a, b = st.columns(2)
    with a:
        risk = pd.DataFrame(summary['risk_counts'].items(), columns=['Risk', 'Servers'])
        chart(px.bar(risk, x='Risk', y='Servers', color='Risk', color_discrete_map=RISK_COLORS, title='Latest saved risk distribution'))
    with b:
        found = summary['hybrid_detected_failure_events']; total = summary['test_failure_events']
        chart(px.pie(names=['Detected', 'Missed'], values=[found, total-found], hole=.6, title='Hybrid event coverage', color_discrete_sequence=['#318a86', '#d94565']))
    a, b = st.columns(2)
    with a:
        chart(px.bar(pd.DataFrame(data.performance()), x='system', y='detected_failure_events', title='Complementary detector coverage', labels={'detected_failure_events': 'Detected events', 'system': 'System'}))
    with b:
        events = data.csv('hybrid_failure_event_analysis.csv')
        chart(px.histogram(events[events.detected], x='lead_time_hours', nbins=24, title='Hybrid warning lead times', labels={'lead_time_hours': 'Hours before failure'}))
    st.divider()
    combined_panel()


def monitor():
    snapshot = data.snapshot()
    columns = st.columns(4)
    for col, label, key in zip(columns, ['Datacenter', 'Rack', 'Switch', 'Risk level'],
                                ['datacenter_id', 'rack_id', 'switch_id', 'hybrid_risk_level']):
        selected = col.selectbox(label, ['All']+sorted(snapshot[key].unique().tolist()), key='filter_'+key)
        if selected != 'All':
            snapshot = snapshot[snapshot[key].eq(selected)]
    search = st.text_input('Search server ID', placeholder='server_0038', key='server_search')
    if search:
        snapshot = snapshot[snapshot.server_id.str.contains(search, case=False, regex=False)]
    if st.checkbox('Operational alerts only', key='alerts_only'):
        snapshot = snapshot[snapshot.operational_alert]
    st.caption(f'{len(snapshot)} servers shown · highest risk first · timestamp-aligned saved SHAP drivers')
    columns = ['server_id', 'datacenter_id', 'rack_id', 'switch_id', 'timestamp', 'failure_probability',
               'anomaly_score', 'anomaly_percentile', 'hybrid_risk_level', 'operational_alert', 'top_risk_driver']
    with st.expander('Browse matching servers', expanded=False):
        st.dataframe(snapshot[columns], hide_index=True, width='stretch')
    if snapshot.empty:
        st.info('No servers match these filters.')
        return
    requested = st.session_state.get('requested_server')
    options = snapshot.server_id.tolist()
    initial = options.index(requested) if requested in options else 0
    server = st.selectbox('Inspect server', options, index=initial)
    detail = data.server_detail(server)
    html(f'<div class="sc-hero"><div class="sc-label">SERVER INVESTIGATION</div><h1>{esc(server)}</h1>{badge(detail["hybrid_risk_level"])}<p>{esc(detail["rack_id"])} · {esc(detail["switch_id"])} · {esc(detail["timestamp"])}</p></div>')
    cards([('Known failure signal',percent(detail['failure_probability']),'Calibrated supervised failure probability'),
           ('Unusual behavior',percent(detail['anomaly_percentile']),f"Anomaly percentile · raw score {detail['anomaly_score']:.3f}"),
           ('Final Sentinel risk',detail['hybrid_risk_level'],'Hybrid priority level, not a calibrated probability')])
    a,b = st.columns(2)
    with a: gauge(detail['failure_probability'],'Calibrated failure probability','#7bb7cb')
    with b: gauge(detail['anomaly_percentile'],'Anomaly percentile — not failure probability','#bba879')
    st.caption(f"Supervised threshold: {percent(detail['supervised_threshold'])}. Anomaly percentile describes unusualness relative to the saved reference, not failure likelihood.")
    st.subheader('Why is SentinelDC concerned?')
    st.write('Saved reasons: '+', '.join(friendly(code) for code in detail['reason_codes']))
    positive, negative = saved_server_factors(server, detail['timestamp'])
    a,b = st.columns(2)
    with a:
        st.markdown('**Pushing risk higher**'); factors(positive or detail['top_explanation_drivers'])
    with b:
        st.markdown('**Pushing risk lower**'); factors(negative)
    st.caption('These contributions come from the existing saved SHAP cache for this exact server and timestamp. No explanations or predictions are recomputed.')
    st.subheader('Trace the telemetry')
    choices = {'Temperature (°C)': 'temperature_c', 'CPU (%)': 'cpu_utilization', 'Memory (%)': 'memory_utilization',
               'Packet loss (%)': 'packet_loss_pct', 'Network latency (ms)': 'network_latency_ms',
               'Disk utilization (%)': 'disk_utilization', 'Power (W)': 'power_watts'}
    chosen = st.selectbox('Telemetry signal', list(choices))
    history = data.server_telemetry(server); feature = choices[chosen]
    figure = px.line(history, x='timestamp', y=feature, title=f'{server} · {chosen}', labels={feature: chosen})
    alerts = history[history.operational_alert.eq(True)]
    figure.add_trace(go.Scatter(x=alerts.timestamp, y=alerts[feature], mode='markers', name='Saved hybrid operational alert',
                               marker=dict(color='#d94565', size=9, symbol='x')))
    chart(figure)
    st.caption('Red markers show times when SentinelDC generated an operational warning. Telemetry is historical; overlays exist only for saved TEST observations. No new predictions are calculated.')


def failure_intelligence():
    st.subheader('Did SentinelDC warn operators before failures happened?')
    summary = data.summary()
    cards([('Failure events',summary['test_failure_events'],'Unique TEST events'),('Detected',summary['hybrid_detected_failure_events'],'At least one eligible warning'),('Missed',summary['test_failure_events']-summary['hybrid_detected_failure_events'],'No eligible warning'),('Median warning',f"{summary['median_warning_lead_time_hours']:g} h",'Detected events only')])
    events = data.csv('hybrid_failure_event_analysis.csv')
    a, b = st.columns(2)
    kind = a.selectbox('Failure type', ['All']+sorted(events.failure_type.unique()))
    detection = b.selectbox('Detection status', ['All', 'Detected', 'Missed'])
    if kind != 'All': events = events[events.failure_type.eq(kind)]
    if detection != 'All': events = events[events.detected.eq(detection == 'Detected')]
    events = events.copy()
    events['Status'] = events.detected.map({True:'DETECTED',False:'MISSED'})
    chart(px.bar(events.sort_values('actual_failure_timestamp'), x='failure_event_id',y='lead_time_hours',color='Status',hover_data=['server_id','actual_failure_timestamp'],title='How much warning did each detected event receive?',color_discrete_map={'DETECTED':'#71b5b2','MISSED':'#cf8b91'}))
    st.caption('Missed events have no lead-time bar; the counts and table retain them.')
    with st.expander('Failure-event evidence table'):
        st.dataframe(events, hide_index=True, width='stretch')
    st.caption('Events are those represented in TEST warning windows; some actual failures occur after the last TEST telemetry timestamp. See complete-window and after-test flags.')
    st.subheader('Different signals catch different events')
    cards([(row['system'],f"{row['detected_failure_events']} / {row['total_failure_events']}",'Unique TEST failures detected') for row in data.performance()])
    st.caption('Anomaly detection found events supervised ML missed, while supervised ML caught events anomaly detection missed. Their overlap is shown below.')
    overlap = data.csv('detector_overlap.csv')
    counts = overlap.category.value_counts().reindex(['BOTH', 'SUPERVISED_ONLY', 'ANOMALY_ONLY', 'NEITHER'], fill_value=0)
    chart(px.bar(x=counts.index, y=counts.values, title='Supervised / anomaly event overlap', labels={'x': 'Coverage', 'y': 'Unique failure events'}))
    st.info('Supervised and anomaly systems provide complementary detection signals. Combining them can increase coverage while also increasing false-alert burden.')


def incidents():
    st.subheader('Follow the shared infrastructure evidence')
    st.write('SentinelDC groups risky servers and searches for shared infrastructure patterns.')
    st.warning('Root-cause outputs are hypotheses, not confirmed causal diagnoses.')
    st.caption('The synthetic benchmark does not provide exact shared rack/switch component ground truth. There is one switch per rack with identical membership.')
    table = data.csv('root_cause_incidents.csv')
    with st.expander('Browse all incidents'):
        st.dataframe(table, hide_index=True, width='stretch')
    example_id = data.json_file('reports/example_root_cause_incident.json')['incident_id']
    ids = table.incident_id.tolist()
    selected = st.selectbox('Select incident', ids, index=ids.index(example_id))
    detail = next(d for d in data.json_lines('root_cause_details.jsonl') if d['incident_id'] == selected)
    html(f'<div class="sc-hero"><div class="sc-label">INCIDENT INVESTIGATION</div><h1>{esc(selected)}</h1>{badge(detail["severity"])}</div>')
    cards([('Suspected component',detail['suspected_component'] or 'UNKNOWN','Inspection target, not a diagnosis'),('Potential blast radius',detail['connected_servers'] if detail['connected_servers'] is not None else 'Unknown','Connected dependency members'),('Additional exposure',detail['potential_additional_exposure'] if detail['potential_additional_exposure'] is not None else 'Unknown','Connected members not currently affected'),('Inspection priority',detail['inspection_priority'],'Saved investigation recommendation')])
    metrics([('Severity', detail['severity']), ('Affected servers', detail['affected_server_count']),
             ('Confidence (heuristic)', f"{detail['root_cause_confidence_score']:.3f}"), ('Signature', detail['telemetry_signature'])])
    st.write(f"Window: {detail['window_start']} → {detail['window_end_exclusive']} (exclusive end)")
    st.write(f"Suspected component: {detail['suspected_component'] or 'UNKNOWN'} · Type: {detail['suspected_component_type']} · Inspection: {detail['inspection_priority']}")
    st.write(f"Component affected: {detail['suspected_component_affected_servers']} · Connected servers: {detail['connected_servers']} · Additional exposure: {detail['potential_additional_exposure']}")
    st.caption('Undefined blast radius for UNKNOWN hypotheses is not zero exposure. Confidence is not a calibrated causal probability.')
    st.write('Reason codes: '+', '.join(detail['reason_codes']))
    st.write('Affected server IDs: '+', '.join(detail['affected_servers']))
    with st.expander('Candidate evidence and limitations'):
        st.json(detail)
    with st.expander('Latest closed-window root-cause snapshot'):
        st.dataframe(data.csv('root_cause_live_view.csv'), hide_index=True)
    example = data.json_file('reports/example_root_cause_incident.json')['incident_id']
    st.subheader(f'Infrastructure context · {example}')
    st.write('Servers connected through shared racks and switches can reveal infrastructure-wide patterns. The figure shows affected servers alongside connected neighbors; this does not establish causality.')
    st.caption('Legend: datacenter → rack → switch → server. Affected servers are distinguished from neighboring members in the saved figure; the suspected component is an inspection hypothesis.')
    st.caption('These fixed example figures do not change with the selected incident.')
    saved_image('example_incident_graph.png', f'Graph context for {example}; potential exposure is dependency, not predicted failure.')
    saved_image('example_root_cause_timeline.png', f'Historical telemetry for {example}.')


def explainability():
    st.info('SHAP explains model behavior; it does not prove causation.')
    st.caption('Explanations target the frozen LightGBM raw log-odds. Calibrated probability is reported separately.')
    st.write('SHAP shows which telemetry features pushed the ML model toward higher or lower predicted failure risk.')
    st.subheader('Global Explanation')
    importance = data.csv('shap_global_importance.csv').head(10).iloc[::-1].copy()
    importance['Feature'] = importance.feature.map(friendly)
    chart(px.bar(importance,x='mean_abs_shap',y='Feature',orientation='h',title='Which features most influence the frozen model?',labels={'mean_abs_shap':'Mean absolute model contribution'}))
    st.caption('Global influence includes both risk increases and decreases; it is not a ranking of physical failure causes.')
    st.subheader('Local Server Explanation')
    local = data.json_lines('shap_local_explanations.jsonl')
    selected = st.selectbox('Saved local explanation', range(len(local)), format_func=lambda i: f"{local[i]['server_id']} · {local[i]['timestamp']} · {local[i]['hybrid_risk_level']}")
    record = local[selected]
    metrics([('Calibrated probability', percent(record['calibrated_failure_probability'])), ('Risk level', record['hybrid_risk_level'])])
    a, b = st.columns(2)
    with a:
        st.markdown('**Risk-increasing factors**'); factors(record['top_risk_drivers'])
    with b:
        st.markdown('**Risk-decreasing factors**'); factors(record['protective_factors'])
    with st.expander('Technical global SHAP plots and full feature ranking'):
        saved_image('shap_global_bar.png', 'Mean absolute influence in either direction.')
        saved_image('shap_summary_beeswarm.png', 'Color: feature value. Horizontal position: model contribution.')
        st.dataframe(data.csv('shap_global_importance.csv'),hide_index=True,width='stretch')
    st.caption("SHAP explains the model’s reasoning. It does not prove what physically caused the failure.")
    example = data.json_file('reports/example_sentinel_intelligence_report.json')
    saved_image('example_shap_waterfall.png', f"Fixed example: {example['server_id']} at {example['timestamp']}; independent of local selection.")


def performance():
    descriptions = {'SUPERVISED':('Known failure patterns','More selective on this benchmark; unfamiliar patterns can be missed.'),'ANOMALY':('Unusual server behavior','Broader event coverage, but many false alerts.'),'HYBRID':('Complementary signals','Highest event coverage here; operational precision remains low.')}
    for col, row in zip(st.columns(3),data.performance()):
        with col:
            purpose, weakness = descriptions[row['system']]
            cards([(row['system'],f"{row['detected_failure_events']}/{row['total_failure_events']}",purpose+' · '+weakness)])
            st.metric('Operational precision',percent(row['operational_precision']))
            st.caption(f"{row['false_operational_alerts']:,} false operational alerts")
    with st.expander('Technical metrics · raw observations and operational alerts'):
        st.dataframe(pd.DataFrame(data.performance()), hide_index=True, width='stretch')
    st.warning('Operational precision remains low. High event coverage does not imply a manageable false-alert burden.')
    st.caption('Raw row PR-AUC/precision/recall and incident alert metrics have different counting units. All numbers are frozen TEST results. Anomaly and hybrid scores are not calibrated failure probabilities.')
    diagnosis = data.json_file('reports/root_cause_signature_diagnostic.json')
    metrics([('Signature accuracy', percent(diagnosis['signature_accuracy_including_unknown'])),
             ('Evaluated events', diagnosis['evaluated_events']), ('UNKNOWN', diagnosis['unknown']),
             ('Incorrect non-UNKNOWN', diagnosis['incorrect_non_unknown'])])
    st.caption('UNKNOWN counts as incorrect. This measures signature agreement on alert-covered events, not exact component causal accuracy.')
    for limit in LIMITATIONS: st.write('• '+limit)


def architecture():
    thinking(full=True)
    st.subheader('Research Prototype — Current Limitations')
    cards([(f'Limit {i+1}','Research boundary',value) for i,value in enumerate(LIMITATIONS)])
    diagnosis = data.json_file('reports/root_cause_signature_diagnostic.json')
    st.caption(f"Root-cause signature agreement remains {percent(diagnosis['signature_accuracy_including_unknown'])}; {diagnosis['unknown']} of {diagnosis['evaluated_events']} evaluated events are UNKNOWN. This is not hidden or corrected through TEST tuning.")
    st.subheader('How this would evolve in production')
    cards([('Future work','Real telemetry','External validation, incident feedback and better component-level causal ground truth.'),
           ('Future work','Data platform','Kafka, Spark and cloud deployment; not implemented.'),
           ('Future work','Model lifecycle','Study drift and controlled online learning only after prospective validation and alert-budget review.')])
    st.caption('See docs/ARCHITECTURE.md and docs/SCIENTIFIC_METHODS.md for the unchanged scientific design.')


def navigate(page, server=None):
    st.session_state['workspace'] = page
    if server:
        st.session_state['requested_server'] = server
        st.session_state['server_search'] = ''
        st.session_state['alerts_only'] = False
        for key in ['datacenter_id','rack_id','switch_id','hybrid_risk_level']:
            st.session_state['filter_'+key] = 'All'


def main():
    st.set_page_config(page_title='SentinelDC | Failure Intelligence', page_icon='🛰', layout='wide')
    html(CSS)
    st.sidebar.title('SentinelDC')
    st.sidebar.caption('FAILURE INTELLIGENCE\n\nFrozen research benchmark')
    names = dict(zip(PAGES,['Command Center','Server Monitor','Failure Intelligence','Incident Investigation','Why Did the Model Raise the Risk?','Model Performance','Architecture']))
    page = st.sidebar.radio('Workspace', PAGES, key='workspace',format_func=lambda value:names[value])
    st.sidebar.subheader('Demo Mode')
    demo = [('Overview',PAGES[0]),('Investigate High-Risk Server',PAGES[1]),('Explore Failure Event',PAGES[2]),('Investigate Incident',PAGES[3]),('Explain ML Prediction',PAGES[4])]
    for label,target in demo:
        server = data.snapshot().iloc[0].server_id if target==PAGES[1] else None
        st.sidebar.button(label,on_click=navigate,args=(target,server),width='stretch')
    with st.sidebar.expander('Quick glossary'):
        st.markdown('**PR-AUC:** summarizes the precision–recall trade-off for rare failures.\n\n**Anomaly score:** unusualness, not failure probability.\n\n**SHAP:** feature contributions to the model score.\n\n**Hybrid risk:** a priority category from two signals.\n\n**Evidence score:** heuristic hypothesis strength, not causal certainty.\n\n**Warning lead time:** hours between first eligible alert and actual failure, for detected events.')
    if page != PAGES[0]: st.title(names[page])
    st.caption('SentinelDC · observe risk, inspect evidence, acknowledge uncertainty')
    st.info(DISCLAIMER)
    try:
        stamp = data.summary()['snapshot_timestamp']
        st.sidebar.info(f'Historical snapshot\n\n{stamp}\n\nRead-only · no live collection')
        {'Executive Overview': overview, 'Live Server Monitor': monitor, 'Failure Intelligence': failure_intelligence,
         'Incident / Root-Cause Explorer': incidents, 'Model Explainability': explainability,
         'Model Performance': performance, 'Architecture & Limitations': architecture}[page]()
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Saved artifacts could not be loaded or validated: {exc}')
        st.stop()


if __name__ == '__main__':
    main()
