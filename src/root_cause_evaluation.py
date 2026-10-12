"""Post-inference evaluation and explanatory figures. Outcome fields enter here only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from src.root_cause import SIGNATURES
from src.infrastructure_graph import node_id,connected_servers

TYPE_SIGNATURE={'network_failure':'NETWORK_LIKE','overheating':'THERMAL_LIKE','power_failure':'POWER_LIKE','disk_failure':'DISK_LIKE','memory_failure':'MEMORY_COMPUTE_LIKE'}


def evaluate_signatures(details,hybrid,truth,report_dir,figure_dir,window_hours=3):
    """One outcome per actual event at its first eligible hybrid alert incident.

    Accuracy conditions on events with an alert in their 24-hour warning window.
    It measures signature agreement, not infrastructure component accuracy.
    """
    alerts=hybrid.loc[hybrid.operational_alert,['server_id','timestamp']].copy()
    alerts['timestamp']=pd.to_datetime(alerts.timestamp,utc=True)
    outcome=truth[['server_id','timestamp','datacenter_id','failure_event_id','failure_type','actual_failure_timestamp','failure_within_24h']].copy()
    outcome['timestamp']=pd.to_datetime(outcome.timestamp,utc=True)
    merged=alerts.merge(outcome,on=['server_id','timestamp'],validate='one_to_one')
    warning=merged.failure_within_24h.eq(1)
    failure_time=pd.to_datetime(merged.actual_failure_timestamp,utc=True,errors='coerce')
    lead=(failure_time-merged.timestamp).dt.total_seconds()/3600
    if not lead[warning].gt(0).all() or not lead[warning].le(24).all():
        raise ValueError('Evaluation labels must identify a future failure within the 24-hour horizon.')
    merged=merged[warning].sort_values(['timestamp','server_id']).drop_duplicates('failure_event_id',keep='first')
    lookup={(pd.Timestamp(d['window_start']),d['dominant_datacenter']):d for d in details}
    rows=[]
    for row in merged.itertuples(index=False):
        detail=lookup[(row.timestamp.floor(f'{window_hours}h'),row.datacenter_id)]
        actual=TYPE_SIGNATURE[row.failure_type];predicted=detail['telemetry_signature']
        rows.append({'failure_event_id':row.failure_event_id,'server_id':row.server_id,'incident_id':detail['incident_id'],
            'alert_timestamp':str(row.timestamp),'actual_failure_timestamp':str(row.actual_failure_timestamp),
            'true_failure_type':row.failure_type,'true_signature':actual,'inferred_signature':predicted,'correct':actual==predicted})
    columns=['failure_event_id','server_id','incident_id','alert_timestamp','actual_failure_timestamp','true_failure_type','true_signature','inferred_signature','correct']
    table=pd.DataFrame(rows,columns=columns);table.to_csv(Path(report_dir)/'root_cause_signature_evaluation.csv',index=False)
    total=int(truth.failure_event_id.nunique());accuracy=float(table.correct.mean()) if len(table) else None
    summary={'signature_classification_accuracy':accuracy,'evaluated_unique_failure_events':len(table),'actual_test_failure_events':total,
        'events_without_eligible_alert':total-len(table),'coverage':len(table)/total if total else None,
        'evaluation_unit':'one event, assigned to incident of first same-server alert in [failure-24h, failure)',
        'unknown_predictions_included_in_accuracy':True,'component_accuracy':None,
        'limitation':'Server failure type is not ground truth for an exact switch/rack/datacenter component. Signature agreement is evaluated only on alert-covered failure events; mixed incidents can contain several true types.'}
    (Path(report_dir)/'root_cause_signature_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    labels=SIGNATURES+['UNKNOWN'];matrix=confusion_matrix(table.true_signature,table.inferred_signature,labels=labels) if len(table) else np.zeros((6,6),dtype=int)
    fig,ax=plt.subplots(figsize=(11,8));ConfusionMatrixDisplay(matrix,display_labels=labels).plot(ax=ax,cmap='Blues',colorbar=False,xticks_rotation=35,values_format='d')
    ax.set_title(f'Root-cause signature agreement: {len(table)} alert-covered events\nNot infrastructure component accuracy');fig.tight_layout();fig.savefig(Path(figure_dir)/'root_cause_signature_confusion_matrix.png',dpi=150);plt.close(fig)
    return summary


def clustering_analysis(hybrid,truth,report_dir,window_hours=3,repeats=100):
    """Matched-time healthy null; diagnostic only after root-cause inference."""
    metadata=truth[['server_id','timestamp','rack_id','switch_id','datacenter_id','failure_within_24h']].copy()
    metadata['timestamp']=pd.to_datetime(metadata.timestamp,utc=True)
    keys=hybrid.loc[hybrid.operational_alert,['server_id','timestamp']].copy();keys['timestamp']=pd.to_datetime(keys.timestamp,utc=True)
    alerts=keys.merge(metadata,on=['server_id','timestamp'],validate='one_to_one')
    def concentration(frame,column):
        frame=frame.copy();frame['bucket']=frame.timestamp.dt.floor(f'{window_hours}h')
        frame=frame.drop_duplicates(['bucket','server_id'])
        total=frame.groupby('bucket').size();denominator=(total*(total-1)/2).sum()
        group=frame.groupby(['bucket',column]).size();numerator=(group*(group-1)/2).sum()
        return float(numerator/denominator) if denominator else 0.
    columns=['switch_id','rack_id','datacenter_id']
    observed={column:concentration(alerts,column) for column in columns}
    healthy=metadata[metadata.failure_within_24h.eq(0)]
    pools={time:group.index.to_numpy() for time,group in healthy.groupby('timestamp')}
    counts=alerts.groupby('timestamp').size();rng=np.random.default_rng(42)
    null={column:[] for column in columns};replacement=0
    for _ in range(repeats):
        selected=[]
        for time,count in counts.items():
            pool=pools.get(time,np.array([],dtype=int))
            if not len(pool):raise ValueError('No healthy observations available for time-matched null.')
            replace=count>len(pool);replacement+=int(replace)
            selected.extend(rng.choice(pool,int(count),replace=replace))
        sample=metadata.loc[selected]
        for column in columns:null[column].append(concentration(sample,column))
    rows=[]
    for column in columns:
        values=np.asarray(null[column]);mean=float(values.mean());std=float(values.std(ddof=1))
        rows.append({'component_type':column.removesuffix('_id').upper(),'observed_same_component_pair_fraction':observed[column],
          'healthy_null_mean':mean,'healthy_null_std':std,'observed_minus_null':observed[column]-mean,
          'enrichment_ratio':observed[column]/mean if mean else np.nan,'z_score':(observed[column]-mean)/std if std else np.nan,
          'one_sided_empirical_p':(1+int((values>=observed[column]).sum()))/(repeats+1),
          'null_repeats':repeats,'seed':42,'time_buckets':f'{window_hours}h UTC','sampling':'same alert count per timestamp from healthy observations',
          'replacement_timestamp_draws':replacement,'note':'Descriptive association; no causal component ground truth; rack/switch memberships may be identical.'})
    table=pd.DataFrame(rows);table.to_csv(Path(report_dir)/'infrastructure_alert_clustering.csv',index=False)
    return table


def example_figures(details,graph,telemetry,hybrid,report_dir,figure_dir):
    if not details:
        (Path(report_dir)/'example_root_cause_incident.json').write_text('{"incident_available": false}\n');return None
    known=[d for d in details if d['suspected_component_type']!='UNKNOWN']
    example=sorted(known or details,key=lambda d:(-d['root_cause_confidence_score'],d['start_timestamp'],d['incident_id']))[0]
    (Path(report_dir)/'example_root_cause_incident.json').write_text(json.dumps(example,indent=2)+'\n')
    affected=set(example['affected_servers'])
    if example['suspected_component_type']!='UNKNOWN':
        component=node_id(example['suspected_component_type'],example['suspected_component'])
    else:component=node_id('SWITCH',example['dominant_switch'])
    neighbors=set(connected_servers(graph,component))-affected
    shown=sorted(affected)[:28]+sorted(neighbors)[:max(0,36-min(28,len(affected)))]
    nodes=set()
    for server in shown:
        key=node_id('SERVER',server);nodes.add(key);nodes.update(nx.ancestors(graph,key))
    sub=graph.subgraph(nodes);positions={}
    for level,kind in enumerate(['DATACENTER','RACK','SWITCH','SERVER']):
        layer=sorted(n for n in sub if sub.nodes[n]['node_type']==kind)
        for i,n in enumerate(layer):positions[n]=((i+.5)/len(layer),3-level)
    colors=['#cb633b' if sub.nodes[n]['node_type']=='SERVER' and sub.nodes[n]['component_id'] in affected else '#cbd5df' if sub.nodes[n]['node_type']=='SERVER' else '#427c94' for n in sub]
    fig,ax=plt.subplots(figsize=(15,8))
    nx.draw_networkx(sub,pos=positions,ax=ax,node_color=colors,node_size=460,font_size=6,
                     labels={n:sub.nodes[n]['component_id'] for n in sub},arrows=True,edge_color='#a3abb3')
    ax.set_title(f"{example['incident_id']}: infrastructure context\nOrange = affected; gray = unaffected neighbors; showing {len(shown)} servers (bounded display)")
    ax.axis('off');fig.tight_layout();fig.savefig(Path(figure_dir)/'example_incident_graph.png',dpi=170);plt.close(fig)
    start=pd.Timestamp(example['start_timestamp']);end=pd.Timestamp(example['end_timestamp'])
    servers=sorted(affected)[:4]
    frame=telemetry[telemetry.server_id.isin(servers)].copy();frame['timestamp']=pd.to_datetime(frame.timestamp,utc=True)
    frame=frame[frame.timestamp.ge(start-pd.Timedelta(hours=6))&frame.timestamp.le(end)]
    alerts=hybrid[hybrid.operational_alert&hybrid.server_id.isin(servers)].copy();alerts['timestamp']=pd.to_datetime(alerts.timestamp,utc=True)
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for ax,feature,label in zip(axes,['packet_loss_pct','network_latency_ms','temperature_c'],['Packet loss (%)','Latency (ms)','Temperature (°C)']):
        for server in servers:
            line=frame[frame.server_id==server].sort_values('timestamp')
            x=(line.timestamp-start).dt.total_seconds()/3600
            plotted=ax.plot(x,line[feature],label=server)[0]
            alert_times=set(alerts.loc[alerts.server_id==server,'timestamp'])
            marked=line.timestamp.isin(alert_times)
            ax.scatter(x[marked],line.loc[marked,feature],marker='x',s=65,color=plotted.get_color())
        ax.axvline(0,color='black',linestyle='--',label='Incident start');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,ncol=3);axes[-1].set_xlabel('Hours relative to incident start')
    fig.suptitle(f"{example['incident_id']}: up to four affected servers\n× = saved hybrid alert; no future failure timestamps used")
    fig.tight_layout();fig.savefig(Path(figure_dir)/'example_root_cause_timeline.png',dpi=150);plt.close(fig)
    return example
