"""Lightweight, accessible presentation primitives; no scientific computation."""
from html import escape
import streamlit as st
import plotly.graph_objects as go
from dashboard.formatting import RISK_COLORS

CSS = '''<style>
.stApp{background:#0c1422;color:#e7edf6}
[data-testid="stSidebar"]{background:#101c2c;border-right:1px solid #253449}
.block-container{max-width:1480px;padding-top:2rem;padding-bottom:3rem}
h1,h2,h3{letter-spacing:-.025em;color:#f0f5fc}p{line-height:1.6}
[data-testid="stCaptionContainer"]{color:#acbdd0}
.sc-card,[data-testid="stMetric"]{background:linear-gradient(145deg,#17263a,#111e30);border:1px solid #2a3b51;border-radius:16px;padding:20px;box-shadow:0 6px 22px #0002;animation:arrive .32s ease-out;transition:transform .18s ease,box-shadow .18s ease}
.sc-card:hover,[data-testid="stMetric"]:hover{transform:translateY(-3px);box-shadow:0 10px 28px #0004;border-color:#44617e}
.sc-card h3{font-size:1.12rem;margin:8px 0}.sc-value{font-size:2.15rem;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.2}
.sc-label{font-size:.74rem;letter-spacing:.09em;text-transform:uppercase;color:#b0c3d7;margin-bottom:10px}.sc-note{font-size:.86rem;color:#b8c8da;margin-top:10px;line-height:1.5}
.sc-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:18px 0 24px}
.sc-hero{padding:34px;border:1px solid #35516f;border-radius:22px;background:radial-gradient(ellipse at top right,#21465b80,transparent 62%),#121f31;animation:arrive .4s ease-out}
.sc-hero h1{font-size:clamp(2rem,3.8vw,3.4rem);max-width:900px;line-height:1.15;margin:16px 0}.sc-hero p{color:#c3d2e3;max-width:780px}
.sc-status{font-size:.7rem;letter-spacing:.12em;color:#9bd7ca}.sc-dot{display:inline-block;width:8px;height:8px;border-radius:100%;background:#6abca8;margin-right:8px;animation:signal 2s ease-out 3}
.sc-badge{display:inline-block;border:1px solid currentColor;border-radius:999px;font-size:.72rem;font-weight:700;padding:4px 10px;letter-spacing:.06em}.sc-critical{animation:signal 1.5s ease-out 3}
.sc-story{border-left:3px solid #74b6ca;background:#14253a;padding:22px;border-radius:0 14px 14px 0;margin:20px 0;line-height:1.7}
.sc-track{height:6px;background:#2a394d;border-radius:6px;overflow:hidden;margin-top:14px}.sc-fill{height:100%;background:#79b8c8;transform-origin:left;animation:grow .45s ease-out}
.stButton>button{border-radius:10px;transition:background .15s ease,transform .15s ease}.stButton>button:hover{transform:translateY(-1px)}
@keyframes arrive{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
@keyframes signal{0%,100%{box-shadow:0 0 0 0 #74baa500}50%{box-shadow:0 0 0 5px #74baa520}}
@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@media(max-width:850px){.sc-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.sc-hero{padding:22px}.sc-value{font-size:1.7rem}}
@media(max-width:520px){.sc-grid{grid-template-columns:1fr}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important;transform:none!important}}
</style>'''


def html(value): st.markdown(value, unsafe_allow_html=True)
def esc(value): return escape(str(value))


def cards(items):
    """(label, value, explanatory subtitle); HTML text always escaped."""
    html('<div class="sc-grid">'+''.join(f'<div class="sc-card"><div class="sc-label">{esc(label)}</div><div class="sc-value">{esc(value)}</div><div class="sc-note">{esc(note)}</div></div>' for label,value,note in items)+'</div>')


def badge(level):
    color=RISK_COLORS.get(level,'#a6b5c8')
    return f'<span class="sc-badge {"sc-critical" if level=="CRITICAL" else ""}" style="color:{color}">{esc(level)}</span>'


def thinking(full=False):
    st.subheader('How SentinelDC Thinks' if not full else 'From telemetry to investigation')
    stages=[('01 · Server telemetry','Hourly temperature, load, power and network signals describe each server.'),
            ('02 · Failure prediction + anomaly detection','In parallel: known failure patterns and unusual behavior provide two different signals.'),
            ('03 · Hybrid risk','Combine both signals to prioritize servers; persistence and cooldown reduce repeated warnings.'),
            ('04 · Root-cause intelligence','Search shared infrastructure patterns and propose inspection targets, including UNKNOWN.')]
    html('<div class="sc-grid">'+''.join(f'<div class="sc-card"><div class="sc-label">{esc(title)}</div><div class="sc-note">{esc(note)}</div></div>' for title,note in stages)+'</div>')
    st.caption('Failure prediction looks for known patterns. Anomaly detection looks for unusual behavior even when the failure pattern is unfamiliar. Neither proves a fault.')
    if full:
        cards([('Feature engineering','Past only','Rolling features use earlier telemetry; outcome metadata stays outside model inputs.'),
               ('Model explanation','SHAP','A parallel branch from LightGBM explains its score, not the physical cause.'),
               ('Presentation','Dashboard + API','Read saved evidence without recomputing predictions.')])
        st.graphviz_chart('digraph { graph [bgcolor="transparent",rankdir="TB"]; node [shape=box,style="rounded,filled",fillcolor="#17263a",fontcolor="#e7edf6",color="#46617d",fontname="Arial"]; edge [color="#7fa1b8",fontcolor="#c5d9eb"]; Telemetry -> "Past-only features"; "Past-only features" -> "Supervised LightGBM"; "Past-only features" -> "Anomaly detection"; "Supervised LightGBM" -> "Hybrid risk"; "Anomaly detection" -> "Hybrid risk"; "Hybrid risk" -> "Infrastructure hypotheses"; "Supervised LightGBM" -> "SHAP explanations"; "Infrastructure hypotheses" -> "Saved evidence"; "SHAP explanations" -> "Saved evidence"; "Saved evidence" -> "Streamlit / FastAPI"; }')


def gauge(value,title,color):
    figure=go.Figure(go.Indicator(mode='gauge+number',value=value*100,number={'suffix':'%','font':{'size':32}},title={'text':title,'font':{'size':15}},
        gauge={'axis':{'range':[0,100]},'bar':{'color':color},'bgcolor':'#22334a','borderwidth':0}))
    figure.update_layout(height=235,margin=dict(t=45,b=15,l=30,r=30),paper_bgcolor='rgba(0,0,0,0)',font={'color':'#dce8f5'})
    st.plotly_chart(figure,width='stretch')


@st.cache_data(show_spinner=False, max_entries=1)
def saved_shap_cache():
    """Read already computed contributions; never run SHAP or a model."""
    import numpy as np
    import pandas as pd
    from dashboard.data_loader import REPORTS
    with np.load(REPORTS/'shap_explanation_cache.npz', allow_pickle=False) as cache:
        values, inputs, features = cache['values'], cache['data'], cache['features'].tolist()
    keys = pd.read_csv(REPORTS/'shap_explanation_cache_keys.csv')
    keys['timestamp'] = pd.to_datetime(keys.timestamp, utc=True).astype(str)
    return keys, values, inputs, features


def saved_server_factors(server, timestamp):
    from dashboard.formatting import friendly
    keys, values, inputs, features = saved_shap_cache()
    matched = keys.index[keys.server_id.eq(server) & keys.timestamp.eq(str(timestamp))]
    if len(matched) != 1:
        return [], []
    row = int(matched[0])
    factors = [{'feature': name, 'display_name': friendly(name), 'value': float(inputs[row,i]),
                'shap_value': float(values[row,i])} for i,name in enumerate(features)]
    positive = sorted([f for f in factors if f['shap_value'] > 0], key=lambda f:-f['shap_value'])[:4]
    negative = sorted([f for f in factors if f['shap_value'] < 0], key=lambda f:f['shap_value'])[:4]
    return positive,negative
