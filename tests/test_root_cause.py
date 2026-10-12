"""Graph correctness, strict past-only context, deterministic inference and reports."""
import hashlib
import json
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from src import infrastructure_graph as infra,root_cause as root,root_cause_evaluation as evaluation


@pytest.fixture
def telemetry():
    rows=[]
    for i in range(6):
        for h,time in enumerate(pd.date_range('2025-01-01',periods=80,freq='h',tz='UTC')):
            row={'server_id':f's{i}','datacenter_id':'d0' if i<4 else 'd1','rack_id':f'r{i//2}','switch_id':f'w{i//2}','timestamp':time}
            row.update({feature:1. for feature in root.FEATURES})
            row.update(temperature_c=50.,network_latency_ms=10.,packet_loss_pct=.1,fan_speed_rpm=3000.,power_watts=300.,cpu_utilization=40.,memory_utilization=40.,disk_utilization=40.)
            if i<2 and h>=60:
                row.update(packet_loss_pct=5.,network_latency_ms=60.,network_total_mb_s=100.)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(['timestamp','server_id']).reset_index(drop=True)


@pytest.fixture
def observations(telemetry):
    frame=telemetry[telemetry.timestamp.ge(pd.Timestamp('2025-01-03T12:00Z'))][['server_id','timestamp']].copy().reset_index(drop=True)
    frame['failure_probability']=.2;frame['anomaly_score']=8.;frame['anomaly_percentile']=.99
    frame['hybrid_risk_level']='CRITICAL';frame['operational_alert']=frame.server_id.isin(['s0','s1'])&frame.timestamp.dt.hour.isin([12,18,0,6])
    return frame


def test_graph_membership_artifact_and_blast_radius(telemetry,tmp_path):
    graph=infra.build_graph(telemetry)
    assert nx.is_directed_acyclic_graph(graph)
    for server in telemetry.server_id.unique():
        assert infra.node_id('SERVER',server) in graph
        assert graph.in_degree(infra.node_id('SERVER',server))==1
    summary=infra.save_graph(graph,tmp_path,tmp_path)
    assert summary['number_of_servers']==6 and summary['number_of_datacenters']==2
    assert summary['number_of_edges']==12
    reread=nx.read_graphml(tmp_path/'infrastructure_graph.graphml')
    assert set(reread.edges)==set(graph.edges)
    radius=infra.blast_radius(graph,'SWITCH:w0',{'s0','s4'})
    assert radius['affected_servers']==1 and radius['connected_servers']==2 and radius['potential_additional_exposure']==1


@pytest.mark.parametrize('column,value',[('rack_id','r1'),('switch_id','w1'),('datacenter_id','d1')])
def test_graph_rejects_conflicting_memberships(telemetry,column,value):
    wrong=telemetry.iloc[[0]].copy();wrong[column]=value
    with pytest.raises(ValueError,match='Conflicting'):
        infra.build_graph(pd.concat([telemetry,wrong]))


def test_baselines_exclude_current_and_future(telemetry,observations):
    full=root.contextualize(telemetry,observations)
    first=full[(full.server_id=='s0')&full.timestamp.eq(pd.Timestamp('2025-01-03T12:00Z'))].iloc[0]
    assert first.server_baseline_packet_loss_pct==pytest.approx(.1)
    assert first.switch_baseline_packet_loss_pct==pytest.approx(.1)
    cutoff=pd.Timestamp('2025-01-03T18:00Z')
    obs=observations[observations.timestamp.lt(cutoff)].reset_index(drop=True)
    prefix=root.contextualize(telemetry[telemetry.timestamp.lt(cutoff)],obs)
    pd.testing.assert_frame_equal(prefix,full[full.timestamp.lt(cutoff)].reset_index(drop=True))
    changed=telemetry.copy();changed.loc[changed.timestamp.ge(cutoff),root.FEATURES]=9999
    altered=root.contextualize(changed,observations)
    pd.testing.assert_frame_equal(altered[altered.timestamp.lt(cutoff)].reset_index(drop=True),prefix)


def test_metadata_cannot_affect_inference(telemetry,observations):
    graph=infra.build_graph(telemetry)
    first=root.infer_incidents(root.contextualize(telemetry,observations),graph)
    changed=telemetry.assign(failure_type='network_failure',actual_failure_timestamp='2099-01-01',failure_within_24h=1,failure_event_id='made-up')
    obs=observations.assign(failure_type='disk_failure',actual_failure_timestamp='2000-01-01')
    second=root.infer_incidents(root.contextualize(changed,obs),graph)
    assert first==second


def test_grouping_is_bounded_chronological_and_scores_deterministic(telemetry,observations):
    graph=infra.build_graph(telemetry);context=root.contextualize(telemetry,observations)
    incidents=root.group_incidents(context.sample(frac=1,random_state=5),3)
    starts=[g['window_start'] for g in incidents];assert starts==sorted(starts)
    for group in incidents:
        assert group['end_timestamp']-group['start_timestamp']<pd.Timedelta(hours=3)
        assert group['rows'].datacenter_id.nunique()==1
    details=root.infer_incidents(context,graph)
    assert details==root.infer_incidents(context.sample(frac=1,random_state=7),graph)
    for detail in details:
        assert detail['reason_codes']
        assert all(0<=c['root_cause_confidence_score']<=1 for c in detail['candidates'])
        if detail['suspected_component_type']!='UNKNOWN':
            assert detail['connected_servers']>=detail['suspected_component_affected_servers']


def test_unknown_on_weak_context(telemetry,observations):
    context=root.contextualize(telemetry,observations)
    context[root.SIGNATURES]=0.
    details=root.infer_incidents(context,infra.build_graph(telemetry))
    assert details and all(d['root_cause_level']=='UNKNOWN' for d in details)
    assert all(d['connected_servers'] is None for d in details)


def test_reports_and_signature_evaluation_are_post_inference(telemetry,observations,tmp_path):
    protected=tmp_path/'hybrid_policy.json';protected.write_text('{"frozen":true}')
    digest=hashlib.sha256(protected.read_bytes()).hexdigest()
    graph=infra.build_graph(telemetry);context=root.contextualize(telemetry,observations)
    details=root.infer_incidents(context,graph);root.save_incidents(details,tmp_path)
    assert (tmp_path/'root_cause_incidents.csv').exists() and (tmp_path/'root_cause_live_view.csv').exists()
    assert (tmp_path/'root_cause_details.jsonl').exists()
    assert len(pd.read_csv(tmp_path/'root_cause_incidents.csv'))==len(details)
    # Ground truth is introduced only after the same inference details are saved.
    truth=telemetry.copy();truth['failure_event_id']=None;truth['failure_type']='none'
    truth['failure_within_24h']=0;truth['actual_failure_timestamp']=pd.Series(pd.NaT,index=truth.index,dtype='datetime64[ns, UTC]')
    failure=pd.Timestamp('2025-01-04T00:00Z')
    mask=truth.server_id.eq('s0')&truth.timestamp.ge(failure-pd.Timedelta(hours=24))&truth.timestamp.lt(failure)
    truth.loc[mask,'actual_failure_timestamp']=failure
    truth.loc[mask,'failure_event_id']='event0';truth.loc[mask,'failure_type']='network_failure';truth.loc[mask,'failure_within_24h']=1
    figures=tmp_path/'figures';figures.mkdir()
    summary=evaluation.evaluate_signatures(details,observations,truth,tmp_path,figures)
    assert summary['evaluated_unique_failure_events']==1 and summary['component_accuracy'] is None
    assert pd.read_csv(tmp_path/'root_cause_signature_evaluation.csv').failure_event_id.is_unique
    assert hashlib.sha256(protected.read_bytes()).hexdigest()==digest
    assert evaluation.example_figures(details,graph,telemetry,observations,tmp_path,figures)
    assert (figures/'example_incident_graph.png').exists()
    assert (figures/'example_root_cause_timeline.png').exists()


def test_missing_history_supplies_no_abnormality(telemetry):
    obs=telemetry[telemetry.timestamp.eq(telemetry.timestamp.min())][['server_id','timestamp']].reset_index(drop=True)
    obs=obs.assign(failure_probability=.9,anomaly_score=10.,anomaly_percentile=1.,hybrid_risk_level='CRITICAL',operational_alert=True)
    context=root.contextualize(telemetry,obs)
    assert not context.baseline_available.any()
    assert context[root.SIGNATURES].eq(0).all().all()
    assert all(d['root_cause_level']=='UNKNOWN' for d in root.infer_incidents(context,infra.build_graph(telemetry)))
