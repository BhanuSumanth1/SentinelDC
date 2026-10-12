"""Hybrid policy causality, validation isolation, artifact protection and outputs."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from src import hybrid_risk as risk,hybrid_evaluation as evaluation
from src.alert_policy import AlertPolicy,apply_alert_policy
from src.anomaly_evaluation import apply_anomaly_policy


@pytest.fixture
def configs():
    return {'selected_threshold':.037,'emergency_threshold':.5},{'threshold':2.2,'extreme_threshold':10.}


@pytest.fixture
def frame():
    rows=[]
    for server in ['a','b','c']:
        failure=pd.Timestamp('2025-01-03',tz='UTC')
        for hour in range(72):
            timestamp=pd.Timestamp('2025-01-01',tz='UTC')+pd.Timedelta(hours=hour)
            positive=server!='c' and 0<(failure-timestamp).total_seconds()/3600<=24
            row={'timestamp':timestamp,'server_id':server,'rack_id':'rack','switch_id':'switch',
                 'failure_within_24h':int(positive),'failure_event_id':'e'+server if positive else None,
                 'actual_failure_timestamp':failure if positive else pd.NaT,'failure_type':'disk_failure' if positive else 'none'}
            rows.append(row)
    return pd.DataFrame(rows).sort_values(['timestamp','server_id']).reset_index(drop=True)


def signals(frame):
    rng=np.random.default_rng(7)
    positive=frame.failure_within_24h.to_numpy()==1
    return np.where(positive,.07,.015)+rng.uniform(0,.01,len(frame)),np.where(positive,7.,1.)+rng.uniform(0,1,len(frame))


def test_normalizer_uses_only_validation_and_ties():
    normalizer=risk.AnomalyPercentile().fit_validation([1,2,2,4])
    assert normalizer.fit_split_=='VALIDATION'
    before=normalizer.reference_.copy()
    np.testing.assert_array_equal(normalizer.transform([0,1,2,4,100]),[0,.25,.75,1,1])
    np.testing.assert_array_equal(normalizer.reference_,before)


def test_candidates_and_exact_frozen_baselines(configs,frame):
    sup,ano=configs;policies=risk.candidate_policies(sup,ano)
    assert len(policies)==15 and len({p['strategy'] for p in policies})==6
    assert policies[0]['strategy']=='SUPERVISED_ONLY'
    p,a=signals(frame);normalizer=risk.AnomalyPercentile().fit_validation(a)
    supervised=apply_alert_policy(frame,p,AlertPolicy(.037,.5))
    anomaly=apply_anomaly_policy(frame,a,2.2,10.)
    for policy,expected in zip(policies[:2],[supervised,anomaly]):
        actual,_,_=risk.apply_hybrid_policy(frame,p,a,normalizer,policy)
        np.testing.assert_array_equal(actual.operational_alert,expected.operational_alert)
        np.testing.assert_array_equal(actual.consecutive_high_observations,expected.consecutive_high_observations)


def test_scores_risks_reasons_and_metadata_independence(configs,frame):
    p,a=signals(frame);normalizer=risk.AnomalyPercentile().fit_validation(a)
    changed=frame.drop(columns=['failure_type','actual_failure_timestamp','failure_event_id','failure_within_24h','rack_id','switch_id'])
    for policy in risk.candidate_policies(*configs):
        output,score,_=risk.apply_hybrid_policy(frame,p,a,normalizer,policy)
        again,_,_=risk.apply_hybrid_policy(changed,p,a,normalizer,policy)
        pd.testing.assert_frame_equal(output,again)
        assert np.isfinite(score).all()
        assert set(output.hybrid_risk_level)<=set(risk.RISK_LEVELS)
        assert output.reason_code.str.len().gt(0).all()


def test_persistence_cooldown_and_prefix_invariance(configs):
    frame=pd.DataFrame({'server_id':'a','timestamp':pd.date_range('2025-01-01',periods=12,freq='h',tz='UTC')})
    p=np.full(12,.1);a=np.full(12,8.)
    normalizer=risk.AnomalyPercentile().fit_validation([0,4,8,9])
    for policy in risk.candidate_policies(*configs):
        full,_,_=risk.apply_hybrid_policy(frame,p,a,normalizer,policy)
        for end in [1,2,7,12]:
            prefix,_,_=risk.apply_hybrid_policy(frame.iloc[:end],p[:end],a[:end],normalizer,policy)
            assert prefix.operational_alert.tolist()==full.operational_alert.iloc[:end].tolist()
        times=full.loc[full.operational_alert,'timestamp']
        assert times.diff().dropna().ge(pd.Timedelta(hours=6)).all()
        reversed_rows,_,_=risk.apply_hybrid_policy(frame.iloc[::-1],p[::-1],a[::-1],normalizer,policy)
        assert reversed_rows.sort_values('timestamp').operational_alert.tolist()==full.operational_alert.tolist()


def test_extreme_anomaly_alone_does_not_bypass_gating(configs):
    frame=pd.DataFrame({'server_id':'a','timestamp':pd.date_range('2025-01-01',periods=3,freq='h',tz='UTC')})
    normalizer=risk.AnomalyPercentile().fit_validation([1,2,3])
    for policy in risk.candidate_policies(*configs):
        if policy['strategy'] not in ['ANOMALY_RESCUE','TWO_STAGE_GATED','STRICT_AGREEMENT']:continue
        output,_,_=risk.apply_hybrid_policy(frame,[.001]*3,[100]*3,normalizer,policy)
        assert not output.operational_alert.any()
        assert output.hybrid_risk_level.eq('HIGH').all()
        assert output.reason_code.str.contains('EXTREME_ANOMALY').all()


def test_utility_penalizes_false_alerts_and_rewards_coverage():
    base={'event_detection_rate':.8,'false_alerts_per_server_day':1.,'operational_precision':.1,'detected_ge_6h_pct':60.}
    assert evaluation.utility(base)==pytest.approx(.62)
    assert evaluation.utility({**base,'false_alerts_per_server_day':2.})<evaluation.utility(base)
    assert evaluation.utility({**base,'event_detection_rate':.9})>evaluation.utility(base)


def test_validation_selection_and_frozen_test_outputs(configs,frame,tmp_path,monkeypatch):
    models=tmp_path/'models';reports=tmp_path/'reports';figures=reports/'figures'
    models.mkdir();reports.mkdir();figures.mkdir()
    # Sentinels representing existing frozen models/reports must survive unchanged.
    protected=[]
    for directory,name in [(models,'best_failure_model.joblib'),(models,'anomaly_selected_detector.joblib'),(reports,'final_test_metrics.csv'),(reports,'anomaly_test_metrics.csv')]:
        path=directory/name;path.write_bytes(b'protected');protected.append(path)
    hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    p,a=signals(frame)
    observed=[];fit=risk.AnomalyPercentile.fit_validation
    def record(self,values):
        observed.append(np.asarray(values).copy());return fit(self,values)
    monkeypatch.setattr(risk.AnomalyPercentile,'fit_validation',record)
    frozen=evaluation.select_on_validation(frame,p,a,*configs,models,reports,figures)
    np.testing.assert_array_equal(observed[0],a)
    assert frozen['selection_split']=='VALIDATION'
    assert (models/'hybrid_policy.json').exists()
    # Perturb an independent TEST frame: selection cannot see it.
    test=frame.copy();test['timestamp']+=pd.Timedelta(days=10)
    test['actual_failure_timestamp']+=pd.Timedelta(days=10)
    repeat=evaluation.select_on_validation(frame,p,a,*configs,models,reports,figures)
    assert frozen==repeat
    test_p=p*.9;test_a=a*1.1
    sup=apply_alert_policy(test,test_p,AlertPolicy(.037,.5))
    ano=apply_anomaly_policy(test,test_a,2.2,10.)
    sup.to_csv(reports/'test_predictions_and_alerts.csv',index=False)
    ano.to_csv(reports/'anomaly_test_scores.csv',index=False)
    def forbidden(*args,**kwargs):pytest.fail('Final TEST attempted selection, normalization fitting or model inference')
    monkeypatch.setattr(evaluation,'select_on_validation',forbidden)
    monkeypatch.setattr(risk.AnomalyPercentile,'fit_validation',forbidden)
    monkeypatch.setattr(evaluation,'load_validation_signals',forbidden)
    applications=[];apply=evaluation.apply_hybrid_policy
    def record_apply(*args,**kwargs):
        applications.append(len(args[0]));return apply(*args,**kwargs)
    monkeypatch.setattr(evaluation,'apply_hybrid_policy',record_apply)
    metrics,overlap=evaluation.final_test(test,frozen,models,reports,figures)
    assert applications==[len(test)]
    assert sum(overlap.values())==metrics['total_failure_events']==2
    assert metrics['detected_failure_events']<=2
    events=pd.read_csv(reports/'hybrid_failure_event_analysis.csv')
    assert events.failure_event_id.is_unique
    assert metrics['true_operational_alerts']+metrics['false_operational_alerts']==metrics['total_operational_alerts']
    assert all(hashlib.sha256(path.read_bytes()).hexdigest()==digest for path,digest in hashes.items())
    for name in ['detector_overlap.csv','hybrid_live_view.csv','example_hybrid_risk_report.json','hybrid_test_metrics.csv','hybrid_high_risk_explanations.jsonl']:
        assert (reports/name).exists()
    assert len(pd.read_csv(reports/'hybrid_live_view.csv'))==3


def test_rescue_does_not_supply_a_supervised_persistence_reading(configs):
    policy=next(p for p in risk.candidate_policies(*configs) if p['candidate_id']=='rescue_0.50')
    frame=pd.DataFrame({'server_id':'a','timestamp':pd.date_range('2025-01-01',periods=8,freq='h',tz='UTC')})
    normalizer=risk.AnomalyPercentile().fit_validation([0,1,10])
    p=np.array([.02]*6+[.04,.04]);a=np.array([10.]*6+[0.,0.])
    result,_,_=risk.apply_hybrid_policy(frame,p,a,normalizer,policy)
    assert result.loc[result.operational_alert,'observation_position'].tolist()==[0,7]
    assert result.consecutive_high_observations.iloc[6]==1
