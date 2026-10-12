"""Frozen-signal hybrid policies. No estimator fitting and no outcome inputs."""
import numpy as np
import pandas as pd
from src.alert_policy import AlertPolicy, apply_alert_policy

RISK_LEVELS = ('LOW','MEDIUM','HIGH','CRITICAL')


class AnomalyPercentile:
    """Right empirical CDF of complete VALIDATION scores, including ties."""
    def fit_validation(self, scores):
        values=np.asarray(scores,dtype=float)
        if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
            raise ValueError('Normalization needs nonempty finite validation scores.')
        self.reference_=np.sort(values)
        self.fit_split_='VALIDATION'
        return self

    def transform(self,scores):
        values=np.asarray(scores,dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Anomaly scores must be finite.')
        return np.searchsorted(self.reference_,values,side='right')/len(self.reference_)


def candidate_policies(supervised,anomaly):
    base={'supervised_threshold':supervised['selected_threshold'],
          'supervised_emergency':supervised['emergency_threshold'],
          'anomaly_threshold':anomaly['threshold'],'anomaly_extreme':anomaly['extreme_threshold'],
          'persistence_hours':2,'cooldown_hours':6.0,
          'severity_boundaries':{'medium_supervised_fraction':.5,'unusual_anomaly_percentile':.90,
              'strong_anomaly_percentile':.95,'extreme_anomaly_percentile':.995}}
    rows=[]
    def add(strategy,name,**parameters):
        rows.append({**base,'strategy':strategy,'candidate_id':name,**parameters})
    add('SUPERVISED_ONLY','supervised_only')
    add('ANOMALY_ONLY','anomaly_only')
    for level in [.75,.90,.95]:
        add('STRICT_AGREEMENT',f'agreement_{level:.2f}',anomaly_level=level)
    for fraction in [.50,.75]:
        add('ANOMALY_RESCUE',f'rescue_{fraction:.2f}',evidence_fraction=fraction,extreme_percentile=.995)
    for weight in [.75,.90]:
        for threshold in [.40,.60]:
            add('WEIGHTED_HYBRID',f'weighted_{weight:.2f}_{threshold:.2f}',
                supervised_weight=weight,anomaly_weight=1-weight,hybrid_threshold=threshold,
                supervised_normalization='min(probability / (2 * supervised_threshold), 1)')
    for fraction in [.50,.75]:
        for level in [.90,.95]:
            add('TWO_STAGE_GATED',f'gated_{fraction:.2f}_{level:.2f}',evidence_fraction=fraction,anomaly_level=level,extreme_percentile=.995)
    return rows


def policy_signals(probabilities,anomaly_scores,percentiles,policy):
    p=np.asarray(probabilities,dtype=float);a=np.asarray(anomaly_scores,dtype=float);q=np.asarray(percentiles,dtype=float)
    if p.ndim!=1 or a.shape!=p.shape or q.shape!=p.shape or not np.isfinite(np.r_[p,a,q]).all() or not ((p>=0)&(p<=1)&(q>=0)&(q<=1)).all():
        raise ValueError('Require equally sized finite probabilities, anomaly scores and percentiles.')
    t=policy['supervised_threshold'];emergency=p>=policy['supervised_emergency']
    strategy=policy['strategy'];rescue=np.zeros(len(p),dtype=bool)
    if strategy=='SUPERVISED_ONLY':
        score=p;threshold=t;high=p>=t
    elif strategy=='ANOMALY_ONLY':
        score=a;threshold=policy['anomaly_threshold'];high=a>=threshold;emergency=a>=policy['anomaly_extreme']
    elif strategy=='STRICT_AGREEMENT':
        score=np.minimum(p/t,q/policy['anomaly_level']);threshold=1.;high=score>=1
        emergency=emergency&(q>=policy['anomaly_level'])
    elif strategy=='ANOMALY_RESCUE':
        evidence=t*policy['evidence_fraction']
        rescue=(p>=evidence)&(q>=policy['extreme_percentile'])
        score=np.maximum(p/t,np.minimum(p/evidence,q/policy['extreme_percentile']))
        threshold=1.;high=p>=t;emergency=emergency|rescue
    elif strategy=='WEIGHTED_HYBRID':
        score=policy['supervised_weight']*np.minimum(p/(2*t),1)+policy['anomaly_weight']*q
        threshold=policy['hybrid_threshold'];high=score>=threshold;emergency=emergency&(q>=.95)
    elif strategy=='TWO_STAGE_GATED':
        evidence=t*policy['evidence_fraction']
        score=np.minimum(p/evidence,q/policy['anomaly_level']);threshold=1.;high=score>=1
        emergency=((p>=t)&(q>=policy['extreme_percentile']))|(emergency&(q>=.90))
    else:
        raise ValueError('Unknown hybrid strategy.')
    return score,threshold,high,emergency,rescue


def apply_hybrid_policy(observations,probabilities,anomaly_scores,normalizer,policy,explain=True):
    """Use server/time only; separate instantaneous risk from emitted alerts."""
    p=np.asarray(probabilities,dtype=float);a=np.asarray(anomaly_scores,dtype=float);q=normalizer.transform(a)
    score,threshold,high,emergency,rescue=policy_signals(p,a,q,policy)
    # Category codes preserve exact eligibility predicates. They are never probabilities.
    categories=np.where(emergency,1.,np.where(high,.6,0.))
    if policy['strategy']=='ANOMALY_RESCUE':
        # A low-evidence rescue must not count as a supervised high reading.
        # Compute normal supervised eligibility without cooldown, then merge
        # instant rescues and apply one shared six-hour cooldown.
        base=apply_alert_policy(observations,p,AlertPolicy(policy['supervised_threshold'],
            policy['supervised_emergency'],policy['persistence_hours'],0.0))
        eligible=base.operational_alert.to_numpy()|rescue
        result=apply_alert_policy(observations,eligible.astype(float),AlertPolicy(.5,.9,2,policy['cooldown_hours']))
        result['consecutive_high_observations']=base.consecutive_high_observations.to_numpy()
        emitted=result.operational_alert.to_numpy()
        result.loc[emitted,'alert_reason']=np.where(rescue[emitted],'emergency',base.alert_reason.to_numpy()[emitted])
    else:
        result=apply_alert_policy(observations,categories,AlertPolicy(.5,.9,policy['persistence_hours'],policy['cooldown_hours']))
    result=result.drop(columns='calibrated_failure_probability')
    result['failure_probability']=p;result['anomaly_score']=a;result['anomaly_percentile']=q
    result['hybrid_score']=score;result['raw_positive_prediction']=high|emergency
    bounds=policy['severity_boundaries'];t=policy['supervised_threshold']
    supervised=p>=t;unusual=q>=bounds['unusual_anomaly_percentile']
    extreme=q>=bounds['extreme_anomaly_percentile'];agreement=supervised&(q>=bounds['strong_anomaly_percentile'])
    medium=(p>=t*bounds['medium_supervised_fraction'])|unusual
    strong=supervised|extreme|result.operational_alert.to_numpy()
    critical=agreement
    risk=np.full(len(p),'LOW',dtype=object);risk[medium]='MEDIUM';risk[strong]='HIGH';risk[critical]='CRITICAL'
    result['hybrid_risk_level']=risk
    if explain:
        reasons=[]
        persistent=result.consecutive_high_observations.to_numpy()>=policy['persistence_hours']
        for i in range(len(p)):
            codes=[]
            if supervised[i]: codes.append('SUPERVISED_FAILURE_SIGNAL')
            elif p[i]>=t*bounds['medium_supervised_fraction']: codes.append('SECONDARY_SUPERVISED_EVIDENCE')
            if extreme[i]: codes.append('EXTREME_ANOMALY')
            elif unusual[i]: codes.append('UNUSUAL_ANOMALY')
            if agreement[i]: codes.append('MULTI_SIGNAL_AGREEMENT')
            if persistent[i]: codes.append('PERSISTENT_ABNORMALITY')
            if rescue[i]: codes.append('ANOMALY_RESCUE')
            if result.operational_alert.iloc[i]: codes.append('OPERATIONAL_ALERT')
            if result.alert_reason.iloc[i]=='cooldown': codes.append('COOLDOWN_SUPPRESSION')
            reasons.append('|'.join(codes or ['NORMAL_OPERATION']))
        result['reason_code']=reasons
    return result,score,threshold
