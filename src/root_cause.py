"""Causal telemetry context and deterministic infrastructure hypotheses, not diagnosis."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.infrastructure_graph import TOPOLOGY,connected_servers,blast_radius,node_id

FEATURES=['packet_loss_pct','network_latency_ms','temperature_c','power_watts','cpu_utilization',
 'memory_utilization','disk_utilization','smart_error_count','fan_speed_rpm','gpu_utilization',
 'network_total_mb_s','disk_total_mb_s','power_change_6h','smart_error_change_12h']
FLOORS=np.array([.1,1.,1.,20.,5.,5.,5.,1.,100.,5.,5.,5.,20.,1.])
SIGNATURES=['NETWORK_LIKE','THERMAL_LIKE','POWER_LIKE','DISK_LIKE','MEMORY_COMPUTE_LIKE']
SIGNAL_COLUMNS=['failure_probability','anomaly_score','anomaly_percentile','hybrid_risk_level','operational_alert']
WEIGHTS={'shared_component_evidence':.35,'telemetry_signature_strength':.25,'temporal_coherence':.20,'anomaly_evidence':.10,'hybrid_risk_evidence':.10}
RANK={'LOW':0,'MEDIUM':1,'HIGH':2,'CRITICAL':3}
PRIORITIES=dict(zip(SIGNATURES,['NETWORK','THERMAL','POWER','STORAGE','MEMORY_COMPUTE']))
PROTOCOL={'window_hours':3,'history_hours':48,'minimum_history_observations':12,
 'weights':WEIGHTS,'minimum_signature_strength':.35,'minimum_signature_margin':.05,
 'minimum_candidate_score':.55,'high_confidence_threshold':.75,
 'candidate_coverage_minimum':.10,'datacenter_coverage_minimum':.50,
 'shared_evidence':'0.6 * positive membership lift above within-datacenter background + 0.4 * affected component coverage; single server uses 1 / incident affected count',
 'temporal_evidence':'1 - first-alert onset spread / incident window; a single server gets 0.5, not perfect synchronization',
 'interpretation':'Root-cause confidence score is heuristic evidence, NOT a calibrated causal probability.'}


def _past_statistics(frame,group_column):
    """Strict [t-48h,t) windows; no current row and no future samples."""
    med=np.full((len(frame),len(FEATURES)),np.nan);disp=med.copy()
    for _,group in frame.groupby(group_column,sort=False):
        group=group.sort_values('timestamp');positions=group.index.to_numpy()
        values=group.set_index('timestamp')[FEATURES]
        rolling=values.rolling('48h',closed='left',min_periods=12)
        med[positions]=rolling.median().to_numpy()
        spread=(rolling.quantile(.75)-rolling.quantile(.25)).to_numpy()/1.349
        disp[positions]=np.maximum(spread,FLOORS)
    return med,disp


def contextualize(telemetry,observations):
    """Explicit inference allowlist; even supplied labels/metadata cannot enter."""
    history=telemetry[['timestamp',*TOPOLOGY,*FEATURES]].copy()
    history['timestamp']=pd.to_datetime(history.timestamp,utc=True)
    history=history.sort_values(['timestamp','server_id']).reset_index(drop=True)
    if history.duplicated(['server_id','timestamp']).any():raise ValueError('Duplicate telemetry key.')
    keys=observations[['server_id','timestamp']].copy();keys['timestamp']=pd.to_datetime(keys.timestamp,utc=True)
    if keys.duplicated().any():raise ValueError('Duplicate inference observation key.')
    indexed=history.reset_index(names='_position')
    result=keys.merge(indexed,on=['server_id','timestamp'],how='left',validate='one_to_one')
    if result._position.isna().any():raise ValueError('Missing telemetry for hybrid observations.')
    result[SIGNAL_COLUMNS]=observations[SIGNAL_COLUMNS].reset_index(drop=True)
    positions=result._position.to_numpy(dtype=int)
    med,disp=_past_statistics(history,'server_id')
    values=result[FEATURES].to_numpy(dtype=float)
    for j,feature in enumerate(FEATURES):
        result['server_baseline_'+feature]=med[positions,j]
        result['server_z_'+feature]=np.nan_to_num((values[:,j]-med[positions,j])/disp[positions,j],nan=0.)
    result['baseline_available']=np.isfinite(med[positions]).all(axis=1)
    for level in ['rack','switch']:
        column=level+'_id'
        peers=history.groupby([column,'timestamp'],as_index=False,sort=True)[FEATURES].median().reset_index(drop=True)
        med,disp=_past_statistics(peers,column)
        peer_index=pd.MultiIndex.from_frame(peers[[column,'timestamp']])
        lookup=peer_index.get_indexer(pd.MultiIndex.from_frame(result[[column,'timestamp']]))
        if (lookup<0).any():raise ValueError('Missing infrastructure peer context.')
        for j,feature in enumerate(FEATURES):
            result[level+'_baseline_'+feature]=med[lookup,j]
            result[level+'_z_'+feature]=np.nan_to_num((values[:,j]-med[lookup,j])/disp[lookup,j],nan=0.)
    # Context weights are fixed before evaluation; missing baseline supplies no evidence.
    up={};absolute={}
    for feature in FEATURES:
        z=sum(weight*result[level+'_z_'+feature].to_numpy() for level,weight in [('server',.70),('rack',.15),('switch',.15)])
        up[feature]=np.clip(z/6,0,1);absolute[feature]=np.clip(np.abs(z)/6,0,1)
    result['NETWORK_LIKE']=.4*up['packet_loss_pct']+.4*up['network_latency_ms']+.2*absolute['network_total_mb_s']
    result['THERMAL_LIKE']=.5*up['temperature_c']+.25*absolute['fan_speed_rpm']+.25*np.sqrt((up['cpu_utilization']+up['gpu_utilization'])/2*up['power_watts'])
    result['POWER_LIKE']=.6*absolute['power_change_6h']+.4*absolute['power_watts']
    result['DISK_LIKE']=.4*np.maximum(up['smart_error_count'],up['smart_error_change_12h'])+.3*absolute['disk_total_mb_s']+.3*up['disk_utilization']
    result['MEMORY_COMPUTE_LIKE']=.5*up['memory_utilization']+.3*absolute['cpu_utilization']+.2*np.minimum(up['cpu_utilization'],up['memory_utilization'])
    return result.drop(columns='_position')


def group_incidents(context,window_hours=3):
    if not isinstance(window_hours,int) or window_hours<1:raise ValueError('Window must be a positive integer of hours.')
    alerts=context[context.operational_alert].copy()
    alerts['window_start']=alerts.timestamp.dt.floor(f'{window_hours}h')
    groups=[]
    for (start,dc),group in alerts.groupby(['window_start','datacenter_id'],sort=True):
        group=group.sort_values(['timestamp','server_id'])
        groups.append({'incident_id':f'INCIDENT_{len(groups)+1:04d}','window_start':start,
          'window_end':start+pd.Timedelta(hours=window_hours),'datacenter_id':dc,'rows':group,
          'start_timestamp':group.timestamp.min(),'end_timestamp':group.timestamp.max()})
    return groups


def signature_of(values):
    ordered=sorted(((name,float(values[name])) for name in SIGNATURES),key=lambda x:(-x[1],x[0]))
    strength=ordered[0][1]
    signature=ordered[0][0] if strength>=PROTOCOL['minimum_signature_strength'] and strength-ordered[1][1]>=PROTOCOL['minimum_signature_margin'] else 'UNKNOWN'
    return signature,strength,{name:value for name,value in ordered}


def candidates_for(incident,graph,window_hours=3):
    rows=incident['rows'];affected=set(rows.server_id);total=len(affected)
    dc=node_id('DATACENTER',incident['datacenter_id']);population=len(connected_servers(graph,dc))
    components={node_id(kind,str(value)) for kind,column in [('SERVER','server_id'),('SWITCH','switch_id'),('RACK','rack_id'),('DATACENTER','datacenter_id')] for value in rows[column].unique()}
    candidates=[]
    for component in sorted(components):
        attrs=graph.nodes[component];kind=attrs['node_type'];radius=blast_radius(graph,component,affected)
        subset=rows[rows.server_id.isin(radius['affected_server_ids'])]
        count=radius['affected_servers'];fraction=count/total;background=radius['connected_servers']/population
        coverage=count/radius['connected_servers']
        lift=max(0.,(fraction-background)/(1-background)) if background<1 else 0.
        shared=(.6*lift+.4*coverage) if kind!='SERVER' else fraction
        per_server=subset.groupby('server_id')[SIGNATURES].max()
        signature,strength,signature_values=signature_of(per_server.mean())
        onsets=subset.groupby('server_id').timestamp.min()
        temporal=max(0.,1-(onsets.max()-onsets.min()).total_seconds()/(window_hours*3600)) if len(onsets)>1 else .5
        anomaly=float(subset.groupby('server_id').anomaly_percentile.max().mean())
        severity=float(subset.assign(_risk=subset.hybrid_risk_level.map(RANK)/3).groupby('server_id')._risk.max().mean())
        evidence={'shared_component_evidence':shared,'telemetry_signature_strength':strength,
                  'temporal_coherence':temporal,'anomaly_evidence':anomaly,'hybrid_risk_evidence':severity}
        score=float(np.clip(sum(WEIGHTS[k]*v for k,v in evidence.items()),0,1))
        compatible=(kind=='SERVER' or kind=='SWITCH' and signature=='NETWORK_LIKE' or kind=='RACK' and signature in ['THERMAL_LIKE','POWER_LIKE'] or kind=='DATACENTER' and signature in ['NETWORK_LIKE','THERMAL_LIKE','POWER_LIKE'])
        enough=(kind=='SERVER' or count>=2 and coverage>=PROTOCOL['candidate_coverage_minimum'])
        if kind=='DATACENTER':enough=count>=6 and coverage>=PROTOCOL['datacenter_coverage_minimum']
        eligible=signature!='UNKNOWN' and compatible and enough and score>=PROTOCOL['minimum_candidate_score']
        reasons=[]
        if kind!='SERVER' and count>=2:reasons.append('SHARED_'+kind)
        if lift>0:reasons.append('ABOVE_BACKGROUND_MEMBERSHIP')
        if temporal>=2/3 and count>=2:reasons.append('SYNCHRONIZED_ALERT_ONSET')
        for feature,code in [('packet_loss_pct','PACKET_LOSS'),('network_latency_ms','LATENCY'),('temperature_c','TEMPERATURE'),('power_change_6h','POWER_FLUCTUATIONS'),('smart_error_change_12h','SMART_CHANGES')]:
            deviation=subset['server_z_'+feature]
            if feature in ['power_change_6h','smart_error_change_12h']:deviation=deviation.abs()
            elevated=subset.assign(_deviation=deviation).groupby('server_id')._deviation.max().ge(2)
            if elevated.mean()>=.5:reasons.append(('CORRELATED_' if count>=2 else 'ELEVATED_')+code)
        if signature!='UNKNOWN':reasons.append(signature)
        if not enough:reasons.append('INSUFFICIENT_COMPONENT_COVERAGE')
        if not eligible:reasons.append('HYPOTHESIS_NOT_CONFIRMED')
        candidates.append({'component_node':component,'component_id':attrs['component_id'],'component_type':kind,
            'root_cause_confidence_score':score,'signature':signature,'signature_strengths':signature_values,
            'shared_component_fraction':fraction,'expected_background_fraction_within_datacenter':background,
            'component_coverage':coverage,'membership_lift':lift,**radius,'evidence':evidence,
            'eligible':bool(eligible),'reason_codes':reasons or ['LOW_EVIDENCE']})
    return sorted(candidates,key=lambda c:(-c['root_cause_confidence_score'],c['component_node']))


def infer_incidents(context,graph,window_hours=3):
    details=[]
    for incident in group_incidents(context,window_hours):
        rows=incident['rows'];candidates=candidates_for(incident,graph,window_hours)
        eligible=[c for c in candidates if c['eligible']]
        best=eligible[0] if eligible else candidates[0]
        known=bool(eligible)
        severity=max(rows.hybrid_risk_level,key=lambda value:RANK[value])
        affected=sorted(rows.server_id.unique());racks=sorted(rows.rack_id.unique());switches=sorted(rows.switch_id.unique())
        shares={}
        for kind,column in [('switch','switch_id'),('rack','rack_id'),('datacenter','datacenter_id')]:
            counts=rows[['server_id',column]].drop_duplicates().groupby(column).size().sort_values(ascending=False)
            dominant=sorted(counts[counts==counts.max()].index)[0]
            shares['dominant_'+kind]=dominant;shares['shared_'+kind+'_fraction']=int(counts.max())/len(affected)
        signature=best['signature'] if known else 'UNKNOWN'
        priority=PRIORITIES.get(signature,'UNKNOWN')
        reasons=best['reason_codes'] if known else ['LOW_OR_AMBIGUOUS_EVIDENCE','ROOT_CAUSE_UNKNOWN']
        detail={'incident_id':incident['incident_id'],'start_timestamp':str(incident['start_timestamp']),
            'end_timestamp':str(incident['end_timestamp']),'window_start':str(incident['window_start']),
            'window_end_exclusive':str(incident['window_end']),
            'inference_as_of':str(incident['window_end']),'severity':severity,'maximum_risk_level':severity,
            'affected_servers':affected,'affected_racks':racks,'affected_switches':switches,
            'affected_server_count':len(affected),'alert_count':len(rows),**shares,
            'suspected_component':best['component_id'] if known else None,
            'suspected_component_type':best['component_type'] if known else 'UNKNOWN',
            'root_cause_level':best['component_type']+'_LEVEL' if known else 'UNKNOWN',
            'root_cause_confidence_score':best['root_cause_confidence_score'],
            'telemetry_signature':signature,'suspected_component_affected_servers':best['affected_servers'] if known else None,
            'connected_servers':best['connected_servers'] if known else None,
            'potential_additional_exposure':best['potential_additional_exposure'] if known else None,
            'inspection_priority':priority,'reason_codes':reasons,
            'major_incident':len(affected)>=5 or severity=='CRITICAL','candidates':candidates,
            'limitations':['Heuristic hypothesis, not a calibrated causal probability.','Rack and switch membership may be identical; signature compatibility is a heuristic, not proof.','Grouping by datacenter conditions the membership background; 100% datacenter sharing alone is not evidence.']}
        reference=rows.sort_values(['server_id','timestamp']).iloc[0]
        detail['baseline_example']={'server_id':reference.server_id,'timestamp':str(reference.timestamp),
            'history_window':'[timestamp - 48 hours, timestamp); minimum 12 past observations',
            'baseline_available':bool(reference.baseline_available),
            'telemetry':{feature:{'current':float(reference[feature]) if pd.notna(reference[feature]) else None,
                **{level+'_historical_median':float(reference[level+'_baseline_'+feature]) if pd.notna(reference[level+'_baseline_'+feature]) else None
                   for level in ['server','rack','switch']}} for feature in ['packet_loss_pct','network_latency_ms','temperature_c','power_watts','smart_error_count']}}
        detail['summary']=(f"{detail['incident_id']} — {severity}. {len(affected)} servers produced {len(rows)} alerts in this bounded window. "
            f"Suspected component: {detail['suspected_component'] or 'UNKNOWN'}; signature: {signature}; evidence score {best['root_cause_confidence_score']:.3f}. "
            f"Observed indicators: {', '.join(code.replace('_',' ').lower() for code in reasons)}. Potential connected-server radius: {detail['connected_servers'] if known else 'undetermined'}; "
            f"additional exposure: {detail['potential_additional_exposure'] if known else 'undetermined'}. Recommended inspection target: {priority}. This is not a confirmed diagnosis.")
        details.append(detail)
    return details


def save_incidents(details,report_dir):
    report_dir=Path(report_dir);report_dir.mkdir(parents=True,exist_ok=True)
    columns=['incident_id','start_timestamp','end_timestamp','severity','affected_server_count','alert_count',
      'dominant_switch','dominant_rack','shared_switch_fraction','shared_rack_fraction','shared_datacenter_fraction',
      'suspected_component','suspected_component_type','root_cause_level','root_cause_confidence_score','telemetry_signature',
      'suspected_component_affected_servers','connected_servers','potential_additional_exposure','inspection_priority','reason_codes']
    rows=[{key:'|'.join(d[key]) if key=='reason_codes' else d[key] for key in columns} for d in details]
    pd.DataFrame(rows,columns=columns).to_csv(report_dir/'root_cause_incidents.csv',index=False)
    with (report_dir/'root_cause_details.jsonl').open('w') as file:
        for detail in details:file.write(json.dumps(detail,allow_nan=False)+'\n')
    live=[]
    for d in details:
        live.append({'incident_id':d['incident_id'],'severity':d['severity'],'suspected_component':d['suspected_component'],
          'component_type':d['suspected_component_type'],'confidence_score':d['root_cause_confidence_score'],
          'signature':d['telemetry_signature'],'affected_servers':d['affected_server_count'],'blast_radius':d['connected_servers'],
          'additional_exposure':d['potential_additional_exposure'],'start_timestamp':d['start_timestamp'],
          'latest_timestamp':d['end_timestamp'],'inspection_priority':d['inspection_priority']})
    # Latest incident window per datacenter; historical dashboard snapshot, not live monitoring.
    if live:
        latest=max(d['window_start'] for d in details)
        live=[row for row,d in zip(live,details) if d['window_start']==latest]
    pd.DataFrame(live,columns=['incident_id','severity','suspected_component','component_type','confidence_score','signature','affected_servers','blast_radius','additional_exposure','start_timestamp','latest_timestamp','inspection_priority']).to_csv(report_dir/'root_cause_live_view.csv',index=False)
    (report_dir/'root_cause_incident_summaries.md').write_text('\n\n'.join(d['summary'] for d in details if d['major_incident'])+'\n')
