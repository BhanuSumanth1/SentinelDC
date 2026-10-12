"""Validation-only anomaly decisions and one frozen selected-detector TEST pass."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

from src.config import MODEL_DIR, REPORT_DIR, FIGURE_DIR, TARGET
from src.anomaly_detection import ANOMALY_FEATURES, DETECTOR_NAMES, anomaly_scores
from src.alert_policy import AlertPolicy, apply_alert_policy, evaluate_operational_alerts
from src.event_evaluation import select_timeline_event

MODEL_RULE = 'Highest complete-validation average precision; exact ties: ROC-AUC, then detector name.'
THRESHOLD_RULE = 'Among validation thresholds with recall >= 0.60 choose highest F2; ties minimize false positives, then maximize threshold. If none qualifies, highest F2 with the same tie rules. F2 penalizes excessive false positives; no external alert-budget standard is claimed.'
EXTREME_RULE = 'Maximum of validation score 99.9th percentile and next representable score above selected threshold; no TEST scores used.'


def row_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int); positive = np.asarray(scores) >= threshold
    tp = int((positive & (labels == 1)).sum()); fp = int((positive & (labels == 0)).sum())
    fn = int((~positive & (labels == 1)).sum()); tn = int((~positive & (labels == 0)).sum())
    return {'pr_auc': float(average_precision_score(labels, scores)), 'roc_auc': float(roc_auc_score(labels, scores)),
        'precision': tp / (tp + fp) if tp + fp else 0.0, 'recall': tp / (tp + fn) if tp + fn else 0.0,
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        'f2': 5 * tp / (5 * tp + fp + 4 * fn) if 5 * tp + fp + 4 * fn else 0.0,
        'true_positive': tp, 'false_positive': fp, 'true_negative': tn, 'false_negative': fn,
        'false_alerts_per_1000_healthy_observations': 1000 * fp / (tn + fp) if tn + fp else np.nan,
        'threshold': float(threshold)}


def threshold_analysis(labels, scores):
    """1,001 validation score quantiles, plus a no-alert endpoint, deduplicated."""
    labels = np.asarray(labels, dtype=int); scores = np.asarray(scores, dtype=float)
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError('Validation needs both classes and finite scores.')
    thresholds = np.unique(np.r_[np.quantile(scores, np.linspace(0, 1, 1001)), np.nextafter(scores.max(), np.inf)])
    order = np.argsort(scores); cumulative = np.r_[0, np.cumsum(labels[order])]
    at = np.searchsorted(scores[order], thresholds, side='left')
    tp = cumulative[-1] - cumulative[at]; fp = len(labels) - at - tp; fn = cumulative[-1] - tp
    def divide(a, b):
        return np.divide(a, b, out=np.zeros(len(a), dtype=float), where=b > 0)
    result = pd.DataFrame({'threshold': thresholds, 'precision': divide(tp, tp + fp), 'recall': tp / cumulative[-1],
        'f1': divide(2*tp, 2*tp+fp+fn), 'f2': divide(5*tp, 5*tp+fp+4*fn),
        'false_positive': fp, 'false_negative': fn,
        'false_alerts_per_1000_healthy_observations': 1000*fp/(len(labels)-cumulative[-1])})
    result['preferred_recall_met'] = result.recall >= .60
    eligible = result[result.preferred_recall_met]
    if eligible.empty:
        eligible = result
    best = eligible.sort_values(['f2','false_positive','threshold'], ascending=[False, True, False]).iloc[0]
    result['selected'] = result.threshold.eq(best.threshold)
    return result, float(best.threshold)


def select_on_validation(validation, preprocessor, detectors, model_dir=MODEL_DIR, report_dir=REPORT_DIR):
    """No TEST argument and no fitting calls. Complete validation is scored."""
    model_dir=Path(model_dir); report_dir=Path(report_dir)
    model_dir.mkdir(parents=True,exist_ok=True); report_dir.mkdir(parents=True,exist_ok=True)
    transformed = preprocessor.transform(validation[ANOMALY_FEATURES]).astype(np.float32)
    rows, analyses, all_scores = [], [], {}
    for name in DETECTOR_NAMES:
        scores = anomaly_scores(name, detectors[name], transformed); all_scores[name] = scores
        analysis, threshold = threshold_analysis(validation[TARGET], scores)
        analysis.insert(0,'detector',name); analyses.append(analysis)
        rows.append({'detector': name, 'split': 'validation', **row_metrics(validation[TARGET], scores, threshold)})
    comparison = pd.DataFrame(rows).sort_values(['pr_auc','roc_auc','detector'], ascending=[False,False,True])
    name = comparison.iloc[0].detector
    comparison['selected'] = comparison.detector.eq(name)
    comparison.to_csv(report_dir/'anomaly_validation_metrics.csv',index=False)
    thresholds = pd.concat(analyses,ignore_index=True)
    thresholds['selected_detector'] = thresholds.detector.eq(name)
    thresholds.to_csv(report_dir/'anomaly_threshold_analysis.csv',index=False)
    row = comparison.iloc[0]; threshold = float(row.threshold)
    extreme = float(max(np.quantile(all_scores[name],.999),np.nextafter(threshold,np.inf)))
    selection = {'selected_detector':name, 'threshold':threshold, 'extreme_threshold':extreme,
        'validation_pr_auc':float(row.pr_auc), 'validation_precision':float(row.precision),
        'validation_recall':float(row.recall), 'validation_f2':float(row.f2),
        'preferred_recall_met':bool(row.recall>=.60), 'selection_split':'VALIDATION',
        'model_selection_rule':MODEL_RULE, 'selection_rule':THRESHOLD_RULE, 'extreme_rule':EXTREME_RULE,
        'persistence_hours':2, 'cooldown_hours':6.0, 'comparison_operator':'>=',
        'score_is_failure_probability':False,
        'validation_start':str(validation.timestamp.min()), 'validation_end':str(validation.timestamp.max())}
    joblib.dump(detectors[name],model_dir/'anomaly_selected_detector.joblib')
    selection['selected_model_sha256'] = hashlib.sha256((model_dir/'anomaly_selected_detector.joblib').read_bytes()).hexdigest()
    selection['preprocessor_sha256'] = hashlib.sha256((model_dir/'anomaly_preprocessor.joblib').read_bytes()).hexdigest()
    (model_dir/'anomaly_threshold.json').write_text(json.dumps(selection,indent=2)+'\n')
    print(comparison.to_string(index=False));print(json.dumps(selection,indent=2))
    return selection


def apply_anomaly_policy(observations, scores, threshold, extreme, cooldown_hours=6.0):
    """Reuse immutable persistence logic via three eligibility categories.

    0/.6/1 are internal below/high/extreme categories, NOT probabilities or
    transformed anomaly scores. Only comparison predicates enter the adapter.
    Original continuous anomaly scores are retained in every output row.
    """
    scores=np.asarray(scores,dtype=float)
    if scores.shape != (len(observations),) or not np.isfinite(scores).all() or not np.isfinite([threshold,extreme]).all() or extreme<=threshold:
        raise ValueError('Require finite per-row scores and an extreme threshold above the operating threshold.')
    categories=np.where(scores>=extreme,1.0,np.where(scores>=threshold,.6,0.0))
    output=apply_alert_policy(observations,categories,AlertPolicy(.5,.9,2,cooldown_hours))
    output=output.drop(columns='calibrated_failure_probability')
    output['anomaly_score']=scores
    return output


def final_test(test, selection, model_dir=MODEL_DIR, report_dir=REPORT_DIR, figure_dir=FIGURE_DIR):
    """Score only the saved selected detector once. No fit or threshold search."""
    model_dir=Path(model_dir);report_dir=Path(report_dir);figure_dir=Path(figure_dir)
    report_dir.mkdir(parents=True,exist_ok=True);figure_dir.mkdir(parents=True,exist_ok=True)
    if selection!=json.loads((model_dir/'anomaly_threshold.json').read_text()):
        raise ValueError('Anomaly policy differs from frozen validation selection.')
    for filename,key in [('anomaly_selected_detector.joblib','selected_model_sha256'),('anomaly_preprocessor.joblib','preprocessor_sha256')]:
        if hashlib.sha256((model_dir/filename).read_bytes()).hexdigest()!=selection[key]:
            raise ValueError('Frozen anomaly artifacts changed after validation selection.')
    preprocessor=joblib.load(model_dir/'anomaly_preprocessor.joblib')
    detector=joblib.load(model_dir/'anomaly_selected_detector.joblib')
    transformed=preprocessor.transform(test[ANOMALY_FEATURES]).astype(np.float32)
    scores=anomaly_scores(selection['selected_detector'],detector,transformed)
    metrics={'detector':selection['selected_detector'],'split':'final_test',**row_metrics(test[TARGET],scores,selection['threshold'])}
    pd.DataFrame([metrics]).to_csv(report_dir/'anomaly_test_metrics.csv',index=False)
    scored=apply_anomaly_policy(test,scores,selection['threshold'],selection['extreme_threshold'],selection['cooldown_hours'])
    scored.to_csv(report_dir/'anomaly_test_scores.csv',index=False)
    # Existing event matcher writes only into a temporary directory. Never
    # overwrite a frozen supervised report, figure or model.
    with tempfile.TemporaryDirectory(prefix='sentineldc-anomaly-events-') as temp:
        directory=Path(temp);figures=directory/'figures';figures.mkdir()
        operational=evaluate_operational_alerts(test,scored,directory,figures)
        for source,target in [('failure_event_analysis.csv','anomaly_failure_event_analysis.csv'),
                              ('operational_alerts.csv','anomaly_alerts.csv')]:
            shutil.copyfile(directory/source,report_dir/target)
    operational.update({'total_anomaly_alerts':operational['total_operational_alerts'],
        'true_anomaly_alerts':operational['true_operational_alerts'],
        'false_anomaly_alerts':operational['false_operational_alerts'],
        'operational_anomaly_precision':operational['operational_alert_precision']})
    pd.DataFrame([operational]).to_csv(report_dir/'anomaly_operational_metrics.csv',index=False)
    events=pd.read_csv(report_dir/'anomaly_failure_event_analysis.csv')
    types=events.groupby('failure_type').detected.agg(['sum','count']).reset_index().rename(columns={'sum':'detected','count':'actual'})
    types['missed']=types.actual-types.detected;types['event_detection_rate']=types.detected/types.actual
    types.to_csv(report_dir/'anomaly_detection_by_failure_type.csv',index=False)
    example_position=save_figures(test,scores,events,types,selection,figure_dir)
    robust=joblib.load(model_dir/'anomaly_robust_detector.joblib')
    explanation=robust.explain(transformed[example_position])
    row=test.iloc[example_position]
    explanation.update({'server_id':str(row.server_id),'timestamp':str(row.timestamp),
        'selection_basis':'first alert of earliest detected event with available history; otherwise highest selected-detector score, earliest row tie',
        'note':'This explains the robust statistical baseline only; it is not an Isolation Forest attribution or a hardware-failure diagnosis.'})
    (report_dir/'example_anomaly_explanation.json').write_text(json.dumps(explanation,indent=2)+'\n')
    return metrics,operational


def save_figures(test,scores,events,types,selection,figure_dir):
    fig,ax=plt.subplots(figsize=(8,5))
    bins=np.linspace(min(scores),max(scores)+1e-12,51)
    for label,mask,color in [('Healthy',test[TARGET].eq(0),'#287e9a'),('Pre-failure',test[TARGET].eq(1),'#c45c35')]:
        ax.hist(scores[mask],bins=bins,density=True,alpha=.55,label=label,color=color)
    ax.axvline(selection['threshold'],color='black',linestyle='--',label='Operating threshold')
    ax.set(xlabel='Anomaly score (not probability)',ylabel='Within-class density',title='FINAL TEST: anomaly score distributions')
    ax.legend();fig.tight_layout();fig.savefig(figure_dir/'anomaly_score_distribution.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));p,r,_=precision_recall_curve(test[TARGET],scores)
    ax.plot(r,p,label=f"AP {average_precision_score(test[TARGET],scores):.4f}")
    ax.axhline(test[TARGET].mean(),color='gray',linestyle='--',label='TEST prevalence')
    ax.set(xlabel='Recall',ylabel='Precision',title='FINAL TEST: anomaly precision–recall',xlim=(0,1),ylim=(0,1))
    ax.legend();fig.tight_layout();fig.savefig(figure_dir/'anomaly_precision_recall_curve.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    bars=ax.bar(types.failure_type,100*types.event_detection_rate,color='#287e9a')
    for bar,row in zip(bars,types.itertuples()):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{row.detected}/{row.actual}',ha='center')
    ax.set(ylabel='Events detected (%)',ylim=(0,115),title='FINAL TEST: anomaly detection by failure type');ax.tick_params(axis='x',rotation=15)
    fig.tight_layout();fig.savefig(figure_dir/'anomaly_detection_by_failure_type.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));detected=events[events.detected].copy()
    if detected.empty:
        position=int(np.argmax(scores));ax.text(.5,.5,'No detected events; timeline unavailable',ha='center',transform=ax.transAxes)
    else:
        detected['actual_failure_timestamp']=pd.to_datetime(detected.actual_failure_timestamp,utc=True)
        example=select_timeline_event(test,detected)
        failure=example.actual_failure_timestamp;first=pd.to_datetime(example.first_alert_timestamp,utc=True)
        times=pd.to_datetime(test.timestamp,utc=True)
        mask=test.server_id.eq(example.server_id)&times.ge(failure-pd.Timedelta(hours=48))&times.le(failure+pd.Timedelta(hours=3))
        positions=np.flatnonzero(mask.to_numpy());order=np.argsort(times.iloc[positions].to_numpy());positions=positions[order]
        hours=(times.iloc[positions]-failure).dt.total_seconds()/3600
        ax.plot(hours,scores[positions],color='#287e9a',label='Anomaly score')
        ax.axhline(selection['threshold'],color='gray',linestyle='--',label='Operating threshold')
        ax.axhline(selection['extreme_threshold'],color='purple',linestyle=':',label='Extreme threshold')
        ax.axvline((first-failure).total_seconds()/3600,color='#ce8c25',linestyle='--',label='First anomaly alert')
        ax.axvline(0,color='#b83232',label='Actual failure')
        ax.set(title=f'FINAL TEST: {example.server_id} · {example.failure_type}',xlabel='Hours relative to failure',ylabel='Anomaly score')
        ax.legend(fontsize=8)
        position=int(np.flatnonzero((test.server_id.eq(example.server_id)&times.eq(first)).to_numpy())[0])
    fig.tight_layout();fig.savefig(figure_dir/'example_anomaly_timeline.png',dpi=150);plt.close(fig)
    return position
