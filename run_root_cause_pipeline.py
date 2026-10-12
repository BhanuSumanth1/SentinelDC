"""Infrastructure hypothesis generation from frozen telemetry and hybrid outputs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from src.config import ROOT,MODEL_DIR,REPORT_DIR,FIGURE_DIR,PROCESSED_PATH
from src.infrastructure_graph import build_graph,save_graph,TOPOLOGY
from src.root_cause import FEATURES,SIGNAL_COLUMNS,PROTOCOL,contextualize,infer_incidents,save_incidents
from src.root_cause_evaluation import evaluate_signatures,clustering_analysis,example_figures


def check_integrity():
    expected=json.loads((REPORT_DIR/'root_cause_protected_artifacts.json').read_text())
    changed=[name for name,digest in expected.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    if changed:raise ValueError(f'Frozen detection benchmark changed: {changed}')
    return len(expected)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--window-hours',type=int,default=3)
    args=parser.parse_args(argv)
    if args.window_hours<1:raise ValueError('Incident window must be positive.')
    protected=check_integrity()
    source_files=[ROOT/'src/infrastructure_graph.py',ROOT/'src/root_cause.py',ROOT/'src/root_cause_evaluation.py',Path(__file__).resolve()]
    protocol={**PROTOCOL,'window_hours':args.window_hours,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        'labels_used_only_after_inference':True,'clustering_null_repeats':100,'random_seed':42}
    (REPORT_DIR/'root_cause_prespecified_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    # Load inference allowlist only. No outcome or generator catalog is available here.
    telemetry=pd.read_csv(PROCESSED_PATH,usecols=['timestamp',*TOPOLOGY,*FEATURES])
    telemetry['timestamp']=pd.to_datetime(telemetry.timestamp,utc=True)
    split=json.loads((REPORT_DIR/'split_metadata.json').read_text())
    topology=telemetry[telemetry.timestamp.le(pd.Timestamp(split['training']['end']))][TOPOLOGY]
    graph=build_graph(topology)
    # Validate static membership across the entire observed period, without using outcomes.
    observed=telemetry[TOPOLOGY].drop_duplicates().sort_values(TOPOLOGY).reset_index(drop=True)
    training_topology=topology.drop_duplicates().sort_values(TOPOLOGY).reset_index(drop=True)
    if not observed.equals(training_topology):raise ValueError('Topology changes require time-aware graph snapshots.')
    summary=save_graph(graph,MODEL_DIR,REPORT_DIR);print('Infrastructure:',summary,flush=True)
    hybrid=pd.read_csv(REPORT_DIR/'hybrid_test_scores.csv',usecols=['server_id','timestamp',*SIGNAL_COLUMNS])
    hybrid['timestamp']=pd.to_datetime(hybrid.timestamp,utc=True)
    print('Computing strictly historical telemetry baselines...',flush=True)
    context=contextualize(telemetry,hybrid)
    print('Grouping bounded infrastructure incidents and ranking hypotheses...',flush=True)
    details=infer_incidents(context,graph,args.window_hours);save_incidents(details,REPORT_DIR)
    context.loc[context.operational_alert].to_csv(REPORT_DIR/'root_cause_alert_evidence.csv',index=False)
    print('Inference saved. Only now loading failure metadata for signature evaluation...',flush=True)
    truth=pd.read_csv(PROCESSED_PATH,usecols=['server_id','timestamp',*TOPOLOGY[:3],'failure_event_id','failure_type','actual_failure_timestamp','failure_within_24h'])
    truth['timestamp']=pd.to_datetime(truth.timestamp,utc=True)
    truth=truth[truth.timestamp.ge(pd.Timestamp(split['test']['start']))&truth.timestamp.le(pd.Timestamp(split['test']['end']))]
    evaluation=evaluate_signatures(details,hybrid,truth,REPORT_DIR,FIGURE_DIR,args.window_hours)
    clustering_analysis(hybrid,truth,REPORT_DIR,args.window_hours)
    example=example_figures(details,graph,telemetry,hybrid,REPORT_DIR,FIGURE_DIR)
    check_integrity()
    counts={'incidents_identified':len(details),'high_confidence_incidents':sum(d['suspected_component_type']!='UNKNOWN' and d['root_cause_confidence_score']>=.75 for d in details),
      'unknown_low_evidence_incidents':sum(d['suspected_component_type']=='UNKNOWN' for d in details),
      'most_common_signature':pd.Series([d['telemetry_signature'] for d in details]).value_counts().index[0] if details else 'UNKNOWN',
      'signature_evaluation':evaluation,'example_incident_id':example['incident_id'] if example else None,
      'protected_file_count':protected,'frozen_supervised_artifacts_unchanged':True,'frozen_anomaly_artifacts_unchanged':True,
      'frozen_hybrid_artifacts_unchanged':True,'dataset_hashes_unchanged':True,'underlying_detector_training_or_inference':False}
    (REPORT_DIR/'root_cause_integrity_check.json').write_text(json.dumps(counts,indent=2)+'\n')
    print(json.dumps(counts,indent=2));print('Root-cause pipeline completed without modifying frozen detectors or results.')


if __name__=='__main__':main()
