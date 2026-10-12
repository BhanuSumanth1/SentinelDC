"""Operational validation selection, then frozen hybrid TEST replay and overlap."""
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import MODEL_DIR, REPORT_DIR, FIGURE_DIR, TARGET
from src.anomaly_detection import ANOMALY_FEATURES, anomaly_scores
from src.anomaly_evaluation import row_metrics
from src.event_evaluation import analyze_failure_events, select_timeline_event
from src.hybrid_risk import AnomalyPercentile, candidate_policies, apply_hybrid_policy

UTILITY_RULE = 'utility = event_detection_rate - 0.25 * false_alerts_per_server_day + 0.10 * operational_precision + 0.10 * detected_ge_6h_fraction_of_all_events. Undefined precision/early fraction contributes zero. Event detection already penalizes misses (1 - missed/events). No TEST-informed coefficient changes.'
SELECTION_RULE = 'Maximum validation utility; exact ties: supervised-only first, fewer false alerts, higher median lead time, candidate_id. No row metric selects the policy.'


def summarize_events(observations,scored):
    events,summary=analyze_failure_events(observations,scored.operational_alert.to_numpy(dtype=float),.5)
    events=events.drop(columns='max_model_risk_score_before_failure')
    flags=scored.operational_alert.to_numpy(dtype=bool)
    positive=observations[TARGET].to_numpy()==1
    # Validated metadata maps positive rows to a same-server event in [failure-24h,failure).
    true=int((flags&positive).sum());false=int((flags&~positive).sum());total=int(flags.sum())
    precision=true/total if total else np.nan;healthy=int((~positive).sum())
    metrics={**summary,'total_operational_alerts':total,'true_operational_alerts':true,
        'false_operational_alerts':false,'operational_precision':precision,
        'redundant_true_alerts':true-summary['detected_failure_events'],
        'false_alerts_per_1000_healthy_observations':1000*false/healthy if healthy else np.nan,
        'false_alerts_per_server_day':false/(len(observations)/24),
        'observed_server_days':len(observations)/24}
    return metrics,events


def utility(metrics):
    def number(value):
        return float(value) if pd.notna(value) else 0.
    return (number(metrics['event_detection_rate'])-.25*metrics['false_alerts_per_server_day']+
            .10*number(metrics['operational_precision'])+.10*number(metrics['detected_ge_6h_pct'])/100)


def load_validation_signals(validation,model_dir=MODEL_DIR):
    """Inference from frozen artifacts on VALIDATION only; no .fit anywhere."""
    model_dir=Path(model_dir)
    supervised=joblib.load(model_dir/'best_failure_model.joblib')
    p=supervised.predict_proba(validation[ANOMALY_FEATURES])[:,1]
    anomaly_config=json.loads((model_dir/'anomaly_threshold.json').read_text())
    preprocessor=joblib.load(model_dir/'anomaly_preprocessor.joblib')
    detector=joblib.load(model_dir/'anomaly_selected_detector.joblib')
    transformed=preprocessor.transform(validation[ANOMALY_FEATURES]).astype(np.float32)
    a=anomaly_scores(anomaly_config['selected_detector'],detector,transformed)
    return p,a


def select_on_validation(validation,probabilities,anomaly_values,supervised_config,anomaly_config,
                         model_dir=MODEL_DIR,report_dir=REPORT_DIR,figure_dir=FIGURE_DIR):
    """Accept only validation frame/signals. Freeze policy before loading TEST caches."""
    model_dir=Path(model_dir);report_dir=Path(report_dir);figure_dir=Path(figure_dir)
    for directory in [model_dir,report_dir,figure_dir]:directory.mkdir(parents=True,exist_ok=True)
    normalizer=AnomalyPercentile().fit_validation(anomaly_values)
    joblib.dump(normalizer,model_dir/'hybrid_anomaly_reference.joblib')
    candidates=candidate_policies(supervised_config,anomaly_config);rows=[]
    for candidate in candidates:
        scored,score,threshold=apply_hybrid_policy(validation,probabilities,anomaly_values,normalizer,candidate,explain=False)
        operational,_=summarize_events(validation,scored)
        row=row_metrics(validation[TARGET],score,threshold)
        rows.append({'candidate_id':candidate['candidate_id'],'strategy':candidate['strategy'],
            'split':'VALIDATION','utility':utility(operational),**operational,
            **{'row_'+key:value for key,value in row.items()},'parameters':json.dumps(candidate,sort_keys=True)})
    comparison=pd.DataFrame(rows)
    comparison['baseline_tie_priority']=comparison.strategy.ne('SUPERVISED_ONLY').astype(int)
    ranking=comparison.sort_values(['utility','baseline_tie_priority','false_operational_alerts','median_lead_time_hours','candidate_id'],ascending=[False,True,True,False,True],na_position='last')
    best=ranking.iloc[0];policy=next(c for c in candidates if c['candidate_id']==best.candidate_id)
    comparison['selected']=comparison.candidate_id.eq(best.candidate_id)
    comparison.to_csv(report_dir/'hybrid_validation_comparison.csv',index=False)
    def native(value):return None if pd.isna(value) else value.item() if hasattr(value,'item') else value
    validation_metrics={key:native(best[key]) for key in ['event_detection_rate','operational_precision','false_operational_alerts','false_alerts_per_server_day','median_lead_time_hours','detected_failure_events','missed_failure_events','utility']}
    frozen={'selected_strategy':policy['strategy'],'policy':policy,'selection_split':'VALIDATION',
        'validation_metrics':validation_metrics,'selection_rule':SELECTION_RULE,'utility_formula':UTILITY_RULE,
        'candidate_count':len(candidates),'strategy_family_count':len(set(c['strategy'] for c in candidates)),
        'normalization':{'fit_split':'VALIDATION','rows':len(validation),'start':str(validation.timestamp.min()),
            'end':str(validation.timestamp.max()),'method':'right empirical CDF; ties receive upper percentile; TEST cannot update reference'},
        'normalizer_sha256':hashlib.sha256((model_dir/'hybrid_anomaly_reference.joblib').read_bytes()).hexdigest(),
        'validation_caveat':'Frozen supervised calibration used an early subset of this validation cohort; these are development comparisons, not independent external estimates.',
        'risk_rule':'MEDIUM: p>=0.5*t or q>=0.90; HIGH: p>=t or q>=0.995 or operational alert; CRITICAL: p>=t AND q>=0.95. Risk alone never bypasses selected policy persistence/cooldown.'}
    (model_dir/'hybrid_policy.json').write_text(json.dumps(frozen,indent=2,allow_nan=False)+'\n')
    fig,ax=plt.subplots(figsize=(10,7))
    for strategy,group in comparison.groupby('strategy'):
        ax.scatter(group.false_alerts_per_server_day,group.event_detection_rate*100,label=strategy,s=50)
    ax.scatter([best.false_alerts_per_server_day],[best.event_detection_rate*100],marker='*',s=230,color='black',label='Selected')
    ax.set(xlabel='False operational alerts per server-day',ylabel='Failure events detected (%)',ylim=(0,105),title='VALIDATION: prespecified hybrid candidates')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figure_dir/'hybrid_policy_comparison.png',dpi=150);plt.close(fig)
    print(comparison[['candidate_id','utility','event_detection_rate','operational_precision','false_alerts_per_server_day','selected']].to_string(index=False))
    return frozen


def load_test_cache(test,report_dir=REPORT_DIR):
    """No underlying model inference: consume aligned, already-frozen TEST outputs."""
    report_dir=Path(report_dir)
    supervised=pd.read_csv(report_dir/'test_predictions_and_alerts.csv')
    anomaly=pd.read_csv(report_dir/'anomaly_test_scores.csv')
    expected=test[['server_id','timestamp']].reset_index(drop=True).copy()
    expected['timestamp']=pd.to_datetime(expected.timestamp,utc=True)
    for frame in [supervised,anomaly]:
        keys=frame[['server_id','timestamp']].copy();keys['timestamp']=pd.to_datetime(keys.timestamp,utc=True)
        if not keys.equals(expected):raise ValueError('Frozen TEST cache keys/order do not match chronological TEST.')
    return supervised.calibrated_failure_probability.to_numpy(),anomaly.anomaly_score.to_numpy(),supervised.operational_alert.to_numpy(),anomaly.operational_alert.to_numpy()


def detector_overlap(test,supervised_flags,anomaly_flags,report_dir,figure_dir):
    sup,_=analyze_failure_events(test,np.asarray(supervised_flags,dtype=float),.5)
    ano,_=analyze_failure_events(test,np.asarray(anomaly_flags,dtype=float),.5)
    overlap=sup[['failure_event_id','server_id','failure_type','actual_failure_timestamp','detected']].rename(columns={'detected':'supervised_detected'})
    overlap=overlap.merge(ano[['failure_event_id','detected']].rename(columns={'detected':'anomaly_detected'}),on='failure_event_id',validate='one_to_one')
    s=overlap.supervised_detected;a=overlap.anomaly_detected
    overlap['category']=np.select([s&a,s&~a,~s&a],['BOTH','SUPERVISED_ONLY','ANOMALY_ONLY'],default='NEITHER')
    overlap.to_csv(Path(report_dir)/'detector_overlap.csv',index=False)
    counts=overlap.category.value_counts().reindex(['BOTH','SUPERVISED_ONLY','ANOMALY_ONLY','NEITHER'],fill_value=0)
    fig,ax=plt.subplots(figsize=(8,5));bars=ax.bar(counts.index,counts.values,color=['#277f90','#4468ad','#c98332','#89949c'])
    ax.bar_label(bars);ax.set(ylabel='Unique TEST failure events',title='Frozen detector overlap (analysis after hybrid selection)');ax.tick_params(axis='x',rotation=10)
    fig.tight_layout();fig.savefig(Path(figure_dir)/'detector_overlap.png',dpi=150);plt.close(fig)
    return {key:int(value) for key,value in counts.items()}


def final_test(test,frozen,model_dir=MODEL_DIR,report_dir=REPORT_DIR,figure_dir=FIGURE_DIR):
    model_dir=Path(model_dir);report_dir=Path(report_dir);figure_dir=Path(figure_dir)
    if json.loads((model_dir/'hybrid_policy.json').read_text())!=frozen:raise ValueError('Hybrid policy differs from frozen validation choice.')
    if hashlib.sha256((model_dir/'hybrid_anomaly_reference.joblib').read_bytes()).hexdigest()!=frozen['normalizer_sha256']:raise ValueError('Frozen anomaly normalization changed.')
    normalizer=joblib.load(model_dir/'hybrid_anomaly_reference.joblib')
    p,a,sup_flags,ano_flags=load_test_cache(test,report_dir)
    # One application of the selected policy. No candidate comparison on TEST.
    scored,score,threshold=apply_hybrid_policy(test,p,a,normalizer,frozen['policy'])
    metrics,events=summarize_events(test,scored)
    metrics={'selected_strategy':frozen['selected_strategy'],**metrics,**{'row_'+k:v for k,v in row_metrics(test[TARGET],score,threshold).items()}}
    pd.DataFrame([metrics]).to_csv(report_dir/'hybrid_test_metrics.csv',index=False)
    events.to_csv(report_dir/'hybrid_failure_event_analysis.csv',index=False)
    scored.to_csv(report_dir/'hybrid_test_scores.csv',index=False)
    # Metadata is joined for display only, after every score and alert is frozen.
    display=scored.copy()
    for column in ['rack_id','switch_id']:
        display[column]=test[column].to_numpy() if column in test else None
    latest=display.sort_values(['timestamp','server_id']).groupby('server_id',sort=True).tail(1)
    latest[['server_id','rack_id','switch_id','timestamp','failure_probability','anomaly_score','anomaly_percentile','hybrid_risk_level','operational_alert','reason_code']].to_csv(report_dir/'hybrid_live_view.csv',index=False)
    # Structured evidence for EVERY HIGH/CRITICAL row, without outcome metadata.
    with (report_dir/'hybrid_high_risk_explanations.jsonl').open('w') as stream:
        for row in scored[scored.hybrid_risk_level.isin(['HIGH','CRITICAL'])].itertuples():
            stream.write(json.dumps({'server_id':row.server_id,'timestamp':str(row.timestamp),
                'supervised_failure_probability':row.failure_probability,'anomaly_score':row.anomaly_score,
                'anomaly_percentile':row.anomaly_percentile,'hybrid_score':row.hybrid_score,
                'risk_level':row.hybrid_risk_level,'operational_alert':row.operational_alert,
                'reason_codes':row.reason_code.split('|')})+'\n')
    overlap=detector_overlap(test,sup_flags,ano_flags,report_dir,figure_dir)
    save_demo(test,scored,events,frozen,report_dir,figure_dir)
    print(json.dumps({'metrics':metrics,'overlap':overlap},indent=2))
    return metrics,overlap


def save_demo(test,scored,events,frozen,report_dir,figure_dir):
    # Representative deterministic sample, not chosen by labels or score quality.
    rng=np.random.default_rng(42);indices=np.sort(rng.choice(len(scored),min(5000,len(scored)),replace=False))
    fig,ax=plt.subplots(figsize=(8,5));ax.scatter(scored.failure_probability.iloc[indices],scored.anomaly_percentile.iloc[indices],s=5,alpha=.25)
    ax.set(xlabel='Calibrated failure probability',ylabel='Validation-reference anomaly percentile',ylim=(0,1.02),title='TEST: representative frozen-signal sample')
    fig.tight_layout();fig.savefig(figure_dir/'supervised_vs_anomaly_scores.png',dpi=150);plt.close(fig)
    detected=events[events.detected]
    if not detected.empty:
        event=select_timeline_event(test,detected)
        chosen_time=pd.Timestamp(event.first_alert_timestamp)
    elif not events.empty:
        event=events.sort_values(['actual_failure_timestamp','failure_event_id']).iloc[0]
        window=test[(test.server_id==event.server_id)&test[TARGET].eq(1)&test.failure_event_id.eq(event.failure_event_id)]
        chosen_time=pd.Timestamp(window.timestamp.min())
    else:
        (report_dir/'example_hybrid_risk_report.json').write_text(json.dumps({'example_available':False,'reason':'No TEST failure events'})+'\n')
        fig,ax=plt.subplots();ax.text(.5,.5,'No TEST event available',ha='center');fig.savefig(figure_dir/'example_hybrid_timeline.png');plt.close(fig);return
    times=pd.to_datetime(scored.timestamp,utc=True);failure=pd.Timestamp(event.actual_failure_timestamp)
    row=scored.loc[(scored.server_id==event.server_id)&times.eq(chosen_time)].iloc[0]
    report={'server_id':event.server_id,'timestamp':str(chosen_time),'calibrated_failure_probability':float(row.failure_probability),
        'supervised_threshold':frozen['policy']['supervised_threshold'],'anomaly_score':float(row.anomaly_score),
        'anomaly_percentile':float(row.anomaly_percentile),'hybrid_score':float(row.hybrid_score),
        'risk_level':row.hybrid_risk_level,'reason_codes':row.reason_code.split('|'),
        'first_alert_timestamp':str(event.first_alert_timestamp) if bool(event.detected) else None,
        'actual_failure_timestamp':str(failure),'warning_lead_time_hours':float(event.lead_time_hours) if bool(event.detected) else None,
        'failure_type':event.failure_type,'detected':bool(event.detected),
        'selection_rule':'earliest detected event with 48h TEST context, otherwise earliest detected; if none, earliest actual event (miss clearly labeled)',
        'outcome_fields_are_reporting_metadata_only':True}
    (report_dir/'example_hybrid_risk_report.json').write_text(json.dumps(report,indent=2)+'\n')
    mask=(scored.server_id==event.server_id)&times.ge(failure-pd.Timedelta(hours=48))&times.le(failure+pd.Timedelta(hours=3))
    window=scored[mask].sort_values('timestamp');hours=(pd.to_datetime(window.timestamp,utc=True)-failure).dt.total_seconds()/3600
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    axes[0].plot(hours,window.failure_probability);axes[0].axhline(frozen['policy']['supervised_threshold'],color='gray',linestyle=':');axes[0].set_ylabel('Failure probability')
    axes[1].plot(hours,window.anomaly_percentile,color='#c98332');axes[1].set(ylabel='Anomaly percentile',ylim=(0,1.05))
    axes[2].step(hours,window.hybrid_risk_level.map({'LOW':0,'MEDIUM':1,'HIGH':2,'CRITICAL':3}),where='post',color='#537656')
    axes[2].set(yticks=[0,1,2,3],yticklabels=['LOW','MEDIUM','HIGH','CRITICAL'],ylabel='Hybrid risk',xlabel='Hours relative to actual failure',ylim=(-.2,3.2))
    for ax in axes:
        ax.axvline(0,color='#b83232',label='Actual failure')
        if bool(event.detected):ax.axvline((pd.Timestamp(event.first_alert_timestamp)-failure).total_seconds()/3600,color='#a87919',linestyle='--',label='First hybrid alert')
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8);fig.suptitle(f'TEST demonstration: {event.server_id} · {event.failure_type}');fig.tight_layout();fig.savefig(figure_dir/'example_hybrid_timeline.png',dpi=150);plt.close(fig)
