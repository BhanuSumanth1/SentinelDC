"""Anomaly healthy-TRAIN fitting, held-out decisions, score direction and alerts."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from src import anomaly_detection as detection, anomaly_evaluation as evaluation
from src.config import FEATURE_COLUMNS, TARGET
from src.train import temporal_split


@pytest.fixture(scope='module')
def data():
    # Explicit small orchestration fixture, never used for benchmark results.
    timestamps=pd.date_range('2025-01-01',periods=720,freq='h',tz='UTC')
    frames=[];rng=np.random.default_rng(5)
    for i in range(4):
        frame=pd.DataFrame(rng.normal(2,.3,(720,len(FEATURE_COLUMNS))),columns=FEATURE_COLUMNS)
        frame['timestamp']=timestamps;frame['server_id']=f's{i}'
        frame[TARGET]=0;frame['failure_event_id']=None
        frame['actual_failure_timestamp']=pd.Series(pd.NaT,index=frame.index,dtype='datetime64[ns, UTC]')
        frame['failure_type']='none'
        for j,hour in enumerate([200,550,690]):
            warning=(frame.timestamp>=timestamps[hour]-pd.Timedelta(hours=24))&(frame.timestamp<timestamps[hour])
            frame.loc[warning,TARGET]=1;frame.loc[warning,'failure_event_id']=f'e{i}_{j}'
            frame.loc[warning,'actual_failure_timestamp']=timestamps[hour]
            frame.loc[warning,'failure_type']='disk_failure'
            frame.loc[warning,FEATURE_COLUMNS[:8]]+=2
        frames.append(frame)
    return pd.concat(frames,ignore_index=True)


@pytest.fixture(scope='module')
def fitted(data,tmp_path_factory):
    directory=tmp_path_factory.mktemp('anomaly')
    training,validation,test,split=temporal_split(data)
    preprocessor,detectors,metadata=detection.fit_anomaly_models(training,directory)
    return directory,training,validation,test,preprocessor,detectors,metadata


def test_healthy_train_preprocessing_and_artifacts(fitted):
    directory,training,validation,test,preprocessor,detectors,metadata=fitted
    healthy=training[training[TARGET].eq(0)][FEATURE_COLUMNS]
    assert metadata['healthy_training_rows']==len(healthy)
    assert metadata['isolation_training_sample_rows']==len(healthy)
    np.testing.assert_allclose(preprocessor.named_steps['imputer'].statistics_,healthy.median())
    np.testing.assert_allclose(preprocessor.named_steps['scaler'].center_,healthy.median())
    assert set(healthy.index).isdisjoint(validation.index)
    assert set(healthy.index).isdisjoint(test.index)
    assert set(healthy.index).isdisjoint(training[training[TARGET].eq(1)].index)
    for filename in ['anomaly_preprocessor.joblib','isolation_forest.joblib','anomaly_robust_detector.joblib','anomaly_statistics.json','anomaly_training_metadata.json']:
        assert (directory/filename).exists()


def test_fitting_observes_only_healthy_train(data,tmp_path,monkeypatch):
    training,validation,test,_=temporal_split(data)
    seen=[]
    original=detection.Pipeline.fit_transform
    def record(self,features,*args,**kwargs):
        seen.append(features.index.tolist())
        return original(self,features,*args,**kwargs)
    monkeypatch.setattr(detection.Pipeline,'fit_transform',record)
    detection.fit_anomaly_models(training,tmp_path)
    assert seen==[training[training[TARGET].eq(0)].index.tolist()]
    assert set(seen[0]).isdisjoint(validation.index.union(test.index))


def test_positive_train_and_future_values_do_not_affect_fit(data,tmp_path):
    training,validation,test,_=temporal_split(data)
    original_pre,original_models,_=detection.fit_anomaly_models(training,tmp_path/'first')
    changed=data.copy()
    changed.loc[validation.index.union(test.index),FEATURE_COLUMNS]=1e8
    changed.loc[training.index[training[TARGET].eq(1)],FEATURE_COLUMNS]=-1e8
    changed_train,*_=temporal_split(changed)
    new_pre,new_models,_=detection.fit_anomaly_models(changed_train,tmp_path/'second')
    np.testing.assert_array_equal(original_pre.named_steps['scaler'].center_,new_pre.named_steps['scaler'].center_)
    probe=original_pre.transform(training[FEATURE_COLUMNS].iloc[:10]).astype(np.float32)
    for name in detection.DETECTOR_NAMES:
        np.testing.assert_array_equal(detection.anomaly_scores(name,original_models[name],probe),detection.anomaly_scores(name,new_models[name],probe))


def test_scores_finite_and_sign_consistent(fitted):
    _,_,_,_,preprocessor,detectors,_=fitted
    ordinary=np.full((1,len(FEATURE_COLUMNS)),2.)
    extreme=np.full((1,len(FEATURE_COLUMNS)),200.)
    transformed=preprocessor.transform(pd.DataFrame(np.r_[ordinary,extreme],columns=FEATURE_COLUMNS)).astype(np.float32)
    for name in detection.DETECTOR_NAMES:
        scores=detection.anomaly_scores(name,detectors[name],transformed)
        assert np.isfinite(scores).all() and scores[1]>scores[0]
    np.testing.assert_array_equal(detection.anomaly_scores('isolation_forest',detectors['isolation_forest'],transformed),-detectors['isolation_forest'].score_samples(transformed))


def test_robust_clipping_and_explanation():
    model=detection.RobustStatisticalDetector().fit(np.zeros((10,len(FEATURE_COLUMNS))))
    one=np.zeros((1,len(FEATURE_COLUMNS)));one[0,0]=1e12
    assert model.anomaly_score(one)[0]==2
    explanation=model.explain(one)
    assert sum(x['contribution_to_overall_score'] for x in explanation['largest_deviations'])==explanation['overall_anomaly_score']
    assert not explanation['score_is_failure_probability']


def test_metadata_excluded():
    forbidden={TARGET,'failure_type','failure_event_id','actual_failure_timestamp','server_id','rack_id','switch_id','datacenter_id','timestamp'}
    assert forbidden.isdisjoint(detection.ANOMALY_FEATURES)
    assert set(detection.ANOMALY_FEATURES)==set(FEATURE_COLUMNS)


def test_validation_only_selection_and_one_test_scoring(fitted,tmp_path,monkeypatch):
    directory,training,validation,test,preprocessor,detectors,_=fitted
    reports=tmp_path/'reports';reports.mkdir()
    selection=evaluation.select_on_validation(validation,preprocessor,detectors,directory,reports)
    # TEST perturbations cannot reach a function that only accepts validation.
    changed=test.copy();changed[FEATURE_COLUMNS]=999;changed[TARGET]=1-changed[TARGET]
    repeated=evaluation.select_on_validation(validation,preprocessor,detectors,directory,reports)
    assert repeated==selection
    assert selection['selection_split']=='VALIDATION'
    assert selection['extreme_threshold']>selection['threshold']
    assert (directory/'anomaly_threshold.json').exists()
    def forbidden(*args,**kwargs):
        pytest.fail('Final TEST attempted fitting or threshold selection')
    monkeypatch.setattr(evaluation,'select_on_validation',forbidden)
    monkeypatch.setattr(evaluation,'threshold_analysis',forbidden)
    monkeypatch.setattr(detection.RobustStatisticalDetector,'fit',forbidden)
    monkeypatch.setattr(detection.IsolationForest,'fit',forbidden)
    monkeypatch.setattr(detection.Pipeline,'fit',forbidden)
    calls=[];original=evaluation.anomaly_scores
    def record(name,detector,features):
        calls.append((name,len(features)))
        return original(name,detector,features)
    monkeypatch.setattr(evaluation,'anomaly_scores',record)
    metrics,operational=evaluation.final_test(test,selection,directory,reports,reports/'figures')
    assert calls==[(selection['selected_detector'],len(test))]
    assert np.isfinite(metrics['pr_auc'])
    assert operational['detected_failure_events']<=operational['total_failure_events']==4
    events=pd.read_csv(reports/'anomaly_failure_event_analysis.csv')
    assert events.failure_event_id.is_unique
    assert operational['true_anomaly_alerts']+operational['false_anomaly_alerts']==operational['total_anomaly_alerts']
    for filename in ['anomaly_validation_metrics.csv','anomaly_threshold_analysis.csv','anomaly_test_metrics.csv','anomaly_operational_metrics.csv','anomaly_detection_by_failure_type.csv','example_anomaly_explanation.json']:
        assert (reports/filename).exists()
    for filename in ['anomaly_score_distribution.png','anomaly_precision_recall_curve.png','anomaly_detection_by_failure_type.png','example_anomaly_timeline.png']:
        assert (reports/'figures'/filename).exists()


def test_anomaly_persistence_chronology_and_cooldown():
    frame=pd.DataFrame({'server_id':'a','timestamp':pd.date_range('2025-01-01',periods=10,freq='h',tz='UTC')})
    scores=np.full(10,5.)
    full=evaluation.apply_anomaly_policy(frame,scores,4.,9.)
    assert full.loc[full.operational_alert,'observation_position'].tolist()==[1,7]
    for end in range(1,11):
        prefix=evaluation.apply_anomaly_policy(frame.iloc[:end],scores[:end],4.,9.)
        assert prefix.operational_alert.tolist()==full.operational_alert.iloc[:end].tolist()
    order=[9,2,5,4,6,1,0,8,3,7]
    shuffled=evaluation.apply_anomaly_policy(frame.iloc[order],scores[order],4.,9.).sort_values('timestamp')
    assert shuffled.operational_alert.tolist()==full.operational_alert.tolist()
    extreme=evaluation.apply_anomaly_policy(frame,np.full(10,10.),4.,9.)
    assert extreme.loc[extreme.operational_alert,'observation_position'].tolist()==[0,6]


def test_anomaly_gaps_reset_persistence_and_thresholds_are_not_probabilities():
    frame=pd.DataFrame({'server_id':['a','a','a','b'],'timestamp':pd.to_datetime(['2025-01-01T00:00Z','2025-01-01T02:00Z','2025-01-01T03:00Z','2025-01-01T00:00Z'])})
    result=evaluation.apply_anomaly_policy(frame,[80]*4,50,100)
    assert result.operational_alert.tolist()==[False,False,True,False]
    assert 'calibrated_failure_probability' not in result


def test_threshold_counts_and_recall_rule():
    labels=np.array([0]*90+[1]*10);scores=np.linspace(-4,8,100)
    analysis,threshold=evaluation.threshold_analysis(labels,scores)
    chosen=analysis[analysis.selected].iloc[0]
    assert chosen.recall>=.6
    assert chosen.f2==analysis.loc[analysis.recall>=.6,'f2'].max()
    metrics=evaluation.row_metrics(labels,scores,threshold)
    for key in ['precision','recall','f1','f2','false_positive','false_negative']:
        assert chosen[key]==pytest.approx(metrics[key])
