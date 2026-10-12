# SentinelDC

Failure prediction and operational alert evaluation on a **hardened synthetic data-center simulation**. This benchmark does not represent a production data center. Root-cause analysis remains future work.

## Local setup: Apple Silicon / Python 3.11

```bash
python3.11 -m venv .venv
source .venv/bin/activate
brew install libomp  # macOS OpenMP dependency, if not installed
python -m pip install -r requirements.txt
pytest -q
python run_pipeline.py
python -m src.predict --input example_with_history.json --history example_history.csv
```

Use native arm64 Python. Native macOS wheels and OpenMP dependencies are documented by [XGBoost](https://xgboost.readthedocs.io/en/stable/install.html) and [LightGBM](https://github.com/lightgbm-org/LightGBM/blob/main/python-package/README.rst). In this local workspace the existing Python runtime reuses scikit-learn's native `libomp.dylib` through a local library symlink. No system library was replaced and no duplicate-runtime suppression flag was used. This workspace environment is not included in the project archive; fresh installations should install `libomp` normally.

To verify a completed run without retraining, reselection, calibration fitting, or TEST inference:

```bash
python run_pipeline.py --verify-only
```

This mode checks frozen input hashes, split metadata, validation decisions and source integrity, then replays cached TEST probabilities through the unchanged alert policy. It compares raw and operational metrics with the saved reports and verifies that all data, model and report files remain byte-for-byte unchanged. Temporary comparison files are removed automatically. The ordinary command still runs the full training pipeline.

Run from the project directory. Existing raw and processed data are reused; only missing files are generated/preprocessed. The pipeline verifies hardened input hashes in `reports/advanced_input_integrity.json`. The generator, feature values, allowlist, and outer chronological splits are unchanged.

Five models train sequentially with bounded tree counts and four estimator workers by default. Set `SENTINELDC_N_JOBS=-1` for all available estimator workers, or a smaller positive count to reduce resource use. Numerical-library threads are bounded before imports. There is no hyperparameter sweep, GPU dependency, or SMOTE. Timings are saved in `reports/training_runtime.csv`.

## Dataset

The `heterogeneous-v2` simulator uses seed 42 and produces 500 servers × 30 days × hourly observations by default. Timestamped incidents are explicit; telemetry warnings remain labeled 1 exactly when a failure occurs in **(t, t + 24h]**, even when a fault's degradation starts earlier or later. A simulated 24-hour future tail makes end-of-data outcome labels known. Repairs reset uptime at the event; at most one fault per server remains a simplification.

Every server has persistent normal CPU, memory, temperature, latency, packet loss, power, disk activity, and fan characteristics. Noisy daily workloads and correlated CPU/IO/thermal/power behavior vary around those baselines. Recovering CPU, memory, thermal, network, disk, SMART, power, and fan disturbances also occur without failures. Shared rack heat/power and switch congestion can affect multiple servers without causing a fault.

Failure onset is type-dependent and varies from 4–72 hours. Severity, slope, intermittent recovery, and affected signals vary by event. Disk failures mainly affect SMART/disk behavior, overheating affects thermal/fan/load behavior, network faults affect packet loss/latency/throughput, memory faults affect memory and CPU stability, and power faults produce fluctuations. There is no universal packet-loss increase for every fault type. Signals overlap healthy conditions, and short-onset faults can have normal-looking observations during much of their positive-label window.

Incident probability is sampled per server rather than deleting observations or fixing a realized positive-row count. The benchmark targets approximately 1–3% positive telemetry rows, with natural variation. This deliberately accelerates incident incidence; it is not an industry fleet failure-rate estimate. Parameters were specified before final model evaluation, without optimizing them to a desired score.

| Group | Columns |
| --- | --- |
| Time and topology | timestamp, datacenter_id, rack_id, server_id, switch_id |
| Utilization (%) | cpu_utilization, memory_utilization, disk_utilization, gpu_utilization |
| Thermal and power | temperature_c, fan_speed_rpm, power_watts |
| Disk throughput (MB/s) | disk_read_mb_s, disk_write_mb_s |
| Network | network_in_mb_s, network_out_mb_s, packet_loss_pct, network_latency_ms |
| Health | smart_error_count, uptime_days |
| Labels and event metadata | failure_within_24h, failure_type, failure_event_id, actual_failure_timestamp |

`failure_event_id` and `actual_failure_timestamp` are shared across an incident's warning observations and null on unrelated rows. Unrelated rows have `failure_type = none`. SMART counts represent hourly errors rather than lifetime cumulative counts, so a SMART-change feature may legitimately be negative.

Four separate simulation catalogs are written beside the raw CSV: `server_profiles.csv`, `failure_event_catalog.csv`, `healthy_transients.csv`, and `shared_conditions.csv`. They expose baseline distributions, fault onset/severity, recovering disturbances, and infrastructure episodes for simulator verification only. They are never joined into model inputs. The main telemetry schema is unchanged.

## Features and leakage prevention

All 15 numeric telemetry columns are used. Nine same-observation features are generated: `hour_of_day`, `day_of_week`, `cpu_memory_interaction`, `temperature_deviation`, `network_total_mb_s`, `disk_total_mb_s`, `error_rate_indicator`, `high_temperature_flag`, and `high_packet_loss_flag`.

Eight additional trend features bring the total to **32**: `temperature_rolling_mean_6h`, `temperature_change_6h`, `packet_loss_rolling_mean_6h`, `latency_rolling_mean_6h`, `smart_error_change_12h`, `cpu_rolling_mean_6h`, `memory_rolling_mean_6h`, and `power_change_6h`.

Rolling means use the same server's `(t−6h, t]` interval, including the current observation. Differences compare against the exact timestamp six or twelve hours earlier, not six or twelve rows earlier. Missing history yields missing differences handled by the training-fitted imputer. No centered windows, future backfill, or cross-server history is used. Observed validation/test history may inform later feature rows as it would online; no model is refitted on that history.

The temperature reference is a fixed 55°C; flags use 75°C and 1% packet loss. CPU/memory interaction is their product divided by 100. The error indicator is `log1p(SMART errors) + packet loss / 5`, a unitless engineering heuristic.

- Identifiers, timestamp itself, target, failure type, failure event ID, and actual failure timestamp are excluded from model inputs using an explicit allowlist.
- Exact duplicate rows are removed. Missing telemetry is filled only from the same server's past reading, at most two hours old. Remaining gaps use training-fitted medians inside each saved model pipeline. No future backfill or whole-dataset median is used.
- Unique timestamps define 70% training, 15% validation, and 15% test windows; every server at a timestamp stays together. The final 24 hours of training and validation are purged so their future outcome windows cannot cross the next boundary. These percentages describe windows before purging. With defaults: training January 1 00:00–January 20 23:00 (240,000 rows); validation January 22 00:00–January 25 11:00 (42,000 rows); test January 26 12:00–January 30 23:00 (54,000 rows), all UTC. A total of 24,000 rows are purged.
- Random splitting would mix neighboring observations from the same degradation episode and let later behavior influence an earlier-time assessment. Chronological splitting simulates learning from the past and assessing later observations.
- Training fits model parameters, imputers, and scaling. Validation chooses the model, fits its probability calibrator, and selects its alert policy. Test measures the frozen choices; no alternative models or thresholds are tried on test, and no base estimator is refitted on validation.
- Imputers and Logistic Regression scaling fit on training data only. No SMOTE is used. HistGradientBoosting's random early-stopping split is disabled.
- Prediction uses the same causal cleanup and feature builder. With no history, rolling means reduce to current readings and unavailable lag differences use training medians; this cold-start mode is not equivalent to the history-rich evaluation. Supply prior observations for realistic inference.

## Models and calibration

| Model | Imbalance handling | Bounds |
| --- | --- | --- |
| LogisticRegression | TRAIN-fitted balanced weights | Standardized inputs; 600 iterations |
| RandomForestClassifier | TRAIN-fitted balanced weights | 100 trees; depth 14; minimum leaf 8 |
| HistGradientBoostingClassifier | TRAIN-fitted balanced weights | 120 iterations; 15 leaves; no random early stopping |
| XGBClassifier | TRAIN negatives / TRAIN positives | 250 trees; depth 5; CPU histogram bins 128 |
| LGBMClassifier | TRAIN negatives / TRAIN positives | 250 trees; depth 6; 31 leaves; histogram bins 127 |

Base estimators, imputers and scalers fit TRAIN only. Exact training counts and positive-class weight are saved in `models/training_class_weights.json`. Neither validation nor test prevalence enters those weights.

The highest full-validation PR-AUC selects the base model. PR-AUC means non-interpolated Average Precision. Exact ties prefer F2 at the common comparison threshold 0.50, then model name. All five comparisons include PR-AUC, ROC-AUC, precision, recall, F1 and F2 and explicitly identify their uncalibrated scores. The comparison threshold is not the operating policy.

Unweighted Platt scaling of logit scores calibrates the selected model. The first third of validation timestamps fits only the calibrator. A 24-hour purge follows, then the remaining validation timestamps assess calibration and select the policy: 28 calibration hours, 24 purged hours, and 32 policy-selection hours in this benchmark. Both subwindows must contain both classes. The base estimator is never refitted. Model choice uses all validation, so these development assessments are not independent model-selection holdouts.

Saved model outputs are called **calibrated failure probability**. Calibration adjusts the score scale distorted by class weighting; it does not establish reliability on real hardware or under distribution drift. Brier scores before/after calibration and reliability curves are recorded on later validation and final TEST, not the calibration fit rows. A monotonic sigmoid preserves ranking, except for numerical saturation.

## Operational Alerting

A raw prediction is one decision per telemetry observation. An operational alert is a notification produced by a separate chronological per-server policy. Near-100% recall with approximately 2% precision is operationally poor: roughly 98% of positive rows are false positives, so event coverage alone can conceal an overwhelming workload.

The calibrated threshold search evaluates 0.010 through 0.990 in 0.001 steps (981 thresholds), using the later policy-validation window only. It records precision, recall, F1, F2, false positives, false negatives and false positive rows per 1,000 healthy observations. These threshold-table counts are raw decisions before persistence and cooldown.

Preferred engineering constraints are recall ≥0.80 and precision ≥0.20. Choose highest F2 among thresholds meeting both. If none qualifies, choose highest F2 overall and explicitly record that the constraints were not met. Ties prefer precision, recall, then higher threshold. These initial engineering choices are not industry standards. F2 favors recall while penalizing poor precision; it does not guarantee acceptable alert workload.

An alert requires two consecutive hourly probabilities at or above the selected threshold, **or** one observation at or above the emergency threshold. Missing hours or low scores reset persistence. The emergency threshold is the maximum of 0.50, the policy-validation probability 99.9th percentile, and the selected threshold plus 0.05, capped at 0.999999. Its value and validation exceedance count are saved. This conservative exception may fire rarely or never; no test scores select it.

A six-hour per-server cooldown suppresses repeated notifications, including emergencies, while scores continue to be recorded. At exactly six hours another qualifying alert is allowed. Configure `SENTINELDC_COOLDOWN_HOURS` before the pipeline; do not adjust it after inspecting TEST. Persistence filters isolated spikes and cooldown limits repeats. TEST starts with empty alert state; prior-window alerts are not imported. The policy reads server/time/score only, never outcome metadata. Single-record inference reports a raw decision and cannot claim that a persisted alert occurred.

A true operational alert must occur on the same server in `[failure − 24 hours, failure)`. At-failure alerts are too late. Repeated true alerts after cooldown remain part of workload and precision denominators and are explicitly reported as redundant true alerts. **Each event is detected at most once**, with its first eligible operational alert determining lead time.

The event cohort includes every incident with a TEST warning row, including partial windows and simulator-known outcomes after the final telemetry timestamp. Real data would require outcome follow-up through that tail. Early detection percentages divide by **all actual events**, including misses. Mean/median lead time condition on detected events; no detections mean undefined lead time. False alerts per server-day divide by observed hourly rows / 24; unobserved hours do not add exposure. False alerts per 1,000 healthy observations use the healthy TEST-row count.

The model, calibrator, thresholds, persistence and cooldown are saved before final TEST inference. One base-model prediction call yields cached raw scores; calibration, raw metrics, operational alerts and plots reuse them. TEST is never used for selection or fitting. This cohort was inspected in previous project stages: “untouched” describes this stage's code isolation, not a fresh independent benchmark. No post-test retuning is performed.

## Outputs

| Artifact | Contents |
| --- | --- |
| `reports/advanced_validation_model_metrics.csv` | Five-model validation comparison |
| `reports/advanced_threshold_analysis.csv` | Fine calibrated threshold sweep |
| `models/decision_threshold.json` | Frozen policy, calibration windows, validation metrics, constraint status |
| `models/best_failure_model.joblib` | TRAIN pipeline and validation-fitted calibrator |
| `models/training_class_weights.json` | TRAIN class counts and positive-class weight |
| `reports/final_test_metrics.csv` | Raw observation metrics, F2, Brier, confusion counts |
| `reports/raw_prediction_metrics.csv` | Row-level counts, separate from incidents |
| `reports/operational_alert_metrics.csv` | Operational alerts, unique events, burden and lead times |
| `reports/operational_alerts.csv` | Alert reason, matched event and first-alert flag |
| `reports/test_predictions_and_alerts.csv` | Every TEST probability, raw decision, streak and cooldown status |
| `reports/failure_event_analysis.csv` | One row per actual event and first operational alert |
| `reports/failure_lead_time_summary.csv` | Event coverage, early detection and partial windows |
| `reports/calibration_metrics.csv` | Validation/TEST Brier before and after calibration |
| `reports/figures/reliability_curve.png` | Final TEST reliability curves |
| `reports/figures/validation_reliability_curve.png` | Later validation reliability curves |
| `reports/advanced_feature_importance.csv` | Selected model's feature importance |
| `reports/figures/advanced_feature_importance.png` | Top 20 features |
| `reports/figures/` | Confusion matrix, ROC/PR, threshold, lead times and timeline |
| `reports/hardened_reference/` | Prior hardened results: 55 tests, RF threshold 0.10 |
| `reports/baseline_reference/` | Earlier easy-simulation results and audit |
| `reports/advanced_input_integrity.json` | Unchanged generator and data SHA-256 snapshot |

The compatibility file `reports/operational_metrics.csv` explicitly counts raw telemetry observations; use `operational_alert_metrics.csv` for notifications. Historical hardened realism and single-feature reports describe the previous model run. In that run uptime was the strongest single feature (ROC-AUC 0.7925), row precision was 0.0230, and all 43 events received some raw warning. That was not evidence of a useful alert policy.

Selected trees use built-in importance when available. Logistic regression uses absolute standardized coefficients. HistGradientBoosting has no public built-in importance; if selected it uses validation-only permutation AP decrease. Importances are associative, affected by correlation, and not causal. No SHAP is used.

## Limitations and future work

The simulation simplifies reality, including one incident per server and immediate repairs that create an uptime relationship. Shared servers across chronological splits measure later observations of known infrastructure, not unseen-server generalization. Calibration has few independent incidents, correlated rows and possible drift. No confidence intervals, operational cost estimates, causal diagnosis or production-readiness claims are provided. Stronger algorithms may still perform poorly; the policy can miss engineering constraints or produce too many alerts.

No Kafka, Spark, FastAPI, Streamlit, Docker, AWS, Neo4j, GNN, LLM, SHAP or autoencoder is added. These remain future work, alongside real datasets, rolling backtests and independent external assessment.

## Verified advanced run

Both `pytest -q` (**70 passed**) and `python run_pipeline.py` completed successfully with native Python 3.11 arm64, XGBoost 3.2.0, and LightGBM 4.7.0. The generator and all existing dataset-file hashes remained unchanged. Five model fits took approximately 21 seconds in this local environment (not a hardware-wide performance guarantee).

Validation selected LightGBM (full-validation PR-AUC 0.0832), followed by Platt calibration. The threshold is **0.037**, emergency threshold **0.50**, persistence **two consecutive hourly observations**, and cooldown **six hours**. Neither preferred operating constraint was met at the selected fallback: policy-validation recall was 0.5404 and precision 0.0344. The emergency threshold had no exceedances in policy-validation.

| Final TEST measure | Value |
| --- | --- |
| PR-AUC / ROC-AUC | 0.1046 / 0.7430 |
| Recall / precision | 0.5573 / 0.0315 |
| F1 / F2 | 0.0596 / 0.1284 |
| Brier score, raw / calibrated | 0.058329 / 0.015029 |
| Confusion matrix TN / FP / FN / TP | 38,802 / 14,360 / 371 / 467 |
| Operational alerts, total / true / false | 2,667 / 87 / 2,580 |
| Operational alert precision | 3.2621% |
| Redundant true alerts after cooldown | 51 |
| Events, actual / detected / missed | 43 / 36 / 7 |
| Event detection rate | 83.72% |
| False alerts per 1,000 healthy observations | 48.5309 |
| False alerts per server-day | 1.1467 |
| Median / mean lead time, detected events | 20.00 / 17.83 hours |
| Events detected ≥12h / ≥6h / ≥3h | 31 / 32 / 33 |
| Early detection percentages, all 43 events | 72.09% / 74.42% / 76.74% |

**This remains operationally poor.** Adding stronger algorithms did not establish stronger final ranking: TEST PR-AUC is below the previous RF run's 0.1377. The operational alert burden is still over one false alert per server-day, and seven events are missed. Calibration improved Brier score, but does not solve weak discrimination. The previous 35,291 false positive rows and the current 2,580 false operational alerts are different counting units and must not be presented as a like-for-like model improvement. No choices were retuned after inspecting these TEST results.

## Unsupervised Anomaly Detection

The anomaly subsystem is independent of the frozen supervised benchmark. A supervised classifier learns relationships associated with its available failure labels; it cannot be assumed to recognize every unseen failure pattern. Anomaly detection instead learns a reference for normal behavior and flags deviations. This can provide complementary evidence, but an anomaly is **not** automatically a hardware failure: legitimate workload spikes, maintenance, and changing operating conditions can also be unusual.

Run the separate subsystem with:

```bash
pytest -q
python run_anomaly_pipeline.py
```

It does not call or retrain the supervised pipeline. A manifest in `reports/anomaly_frozen_supervised_hashes.json` checks all pre-existing dataset, model, report, and scientific source files before and after the run. Outputs have separate anomaly filenames. The generator, original supervised threshold, calibration and saved TEST metrics remain frozen.

Both detectors use the same 32 leakage-safe numeric features, including causal historical trends. Server, rack, switch and datacenter identifiers, raw timestamp, target and failure metadata are excluded from inputs. IDs/time remain reporting metadata only. Every fit uses the existing chronological TRAIN split filtered to `failure_within_24h == 0`. Here “healthy” means no labeled failure within the next 24 hours; it does not guarantee perfect hardware health, and may include earlier degradation and legitimate extreme behavior.

Median imputation and `RobustScaler` fit all healthy TRAIN observations only. Scaling uses training medians and interquartile ranges rather than means and standard deviations. The fitted preprocessing pipeline is saved as `models/anomaly_preprocessor.joblib`. Neither validation nor TEST changes its fitted statistics.

Two fixed, bounded detectors are compared:

- **Robust Statistical Detector:** estimate per-feature median, MAD and IQR in scaled units. Dispersion is the maximum of `1.4826 × MAD`, `IQR / 1.349`, and a fixed floor of 0.1 scaled units. Absolute standardized deviations are clipped at 10; the overall score is the mean of the five largest clipped deviations. Thus one extreme feature contributes at most 2 score units. The floor handles constant and binary features. This fixed heuristic is transparent, not a fitted failure-probability model.
- **Isolation Forest:** 200 trees, up to 4,096 samples per tree, one fixed-seed uniform sample of up to 50,000 healthy TRAIN observations without replacement, all features, and parallel CPU fitting. The complete healthy training count and sampled count are saved. No hyperparameter search is performed. The score is **negative `score_samples`**, so larger values mean more anomalous. The estimator's internal `contamination="auto"` offset, `predict` decision and `decision_function` threshold are not used for alerts.

An anomaly score measures unusualness under its detector, not a calibrated failure probability. The two detectors have different numerical score scales and must not share thresholds. The statistical detector's feature explanations are not Isolation Forest attributions, do not use an LLM or SHAP, and do not establish causes.

All validation rows are scored by both detectors. Highest validation Average Precision (reported as PR-AUC) selects the detector; exact ties prefer ROC-AUC, then detector name. Each detector's threshold sweep uses 1,001 validation-score quantiles plus an endpoint above the maximum score, with duplicates removed. It records precision, recall, F1, F2, false positives, false negatives and false positive rows per 1,000 healthy observations. Among thresholds with recall ≥0.60, highest F2 wins; ties prefer fewer false positives, then a higher threshold. If none qualifies, highest F2 is used and the missed recall preference is recorded. F2 penalizes excessive false positives, but there is no empirically established alert budget or industry-standard threshold here.

An operational anomaly alert requires two consecutive hourly scores at or above the selected threshold, **or** one score at or above the extreme threshold. The extreme threshold is the larger of the complete-validation 99.9th score percentile and the next representable value above the operating threshold. A six-hour cooldown suppresses subsequent notifications for that server, including extreme readings. Missing hours or a low reading reset persistence. All processing uses current/past observations only and starts with empty state in TEST. A three-category internal adapter reuses the existing persistence/cooldown implementation; its numeric category codes are never reported as probabilities. Original continuous anomaly scores are retained.

The detector, preprocessing and complete policy are saved and fingerprinted before one selected-detector TEST scoring pass. Labels in validation select thresholds and assess failure association; no validation labels fit the detectors. TEST labels are retrospective evaluation only. The known TEST cohort has already been examined during supervised development, so this is not a fresh independent external benchmark. Anomaly TEST results are not used to retune anything.

Raw positive rows and operational notifications remain distinct. True alerts must occur on the same server in `[actual failure − 24h, actual failure)`. Each event contributes at most one detection and uses its first eligible anomaly alert for lead time. Repeated true alerts after cooldown still count as operator workload and are reported as redundant; they do not increase event coverage. The cohort retains partial warning windows and simulator-known future-tail outcomes. Median/mean lead time condition on detections; early-detection percentages divide by all actual TEST events, including misses. A warning at the actual failure timestamp is too late.

| Anomaly artifact | Contents |
| --- | --- |
| `models/anomaly_preprocessor.joblib` | Healthy-TRAIN imputation and RobustScaler |
| `models/anomaly_robust_detector.joblib`, `models/anomaly_statistics.json` | Robust reference statistics and aggregation settings |
| `models/isolation_forest.joblib` | Fixed-seed healthy-TRAIN Isolation Forest |
| `models/anomaly_training_metadata.json` | Healthy counts, sample details, features and fit provenance |
| `models/anomaly_selected_detector.joblib` | Validation-selected detector |
| `models/anomaly_threshold.json` | Operating/extreme thresholds, policy, rules and validation metrics |
| `reports/anomaly_validation_metrics.csv` | Complete validation comparison |
| `reports/anomaly_threshold_analysis.csv` | Both detectors' validation threshold sweeps |
| `reports/anomaly_test_metrics.csv` | Selected detector's final raw metrics including F2 |
| `reports/anomaly_operational_metrics.csv` | Alert workload, event coverage and early-warning summaries |
| `reports/anomaly_test_scores.csv`, `reports/anomaly_alerts.csv` | Cached scores, decisions and notification matching |
| `reports/anomaly_failure_event_analysis.csv` | One row per actual failure event |
| `reports/anomaly_detection_by_failure_type.csv` | Event detection for each observed failure type |
| `reports/example_anomaly_explanation.json` | Robust contributions at a deterministic example observation |
| `reports/anomaly_frozen_protocol.json` | Prespecified source hashes, splits and frozen validation policy |
| `reports/anomaly_integrity_verification.json` | Preservation checks and measured runtime |
| `reports/figures/anomaly_score_distribution.png` | Healthy/pre-failure within-class score distributions |
| `reports/figures/anomaly_precision_recall_curve.png` | Selected detector's TEST PR curve |
| `reports/figures/anomaly_detection_by_failure_type.png` | Detected/actual events by failure type |
| `reports/figures/example_anomaly_timeline.png` | Scores, both thresholds, first alert and actual failure |

The example chooses the earliest detected event with complete available history, or the earliest detected event if only partial history is available. The explanation uses its first alert. If nothing is detected, the timeline states that and the explanation uses the highest selected-detector score, breaking ties by row order. It always explains the robust statistical baseline, even if Isolation Forest is selected. This is a diagnostic example, not evidence that the two systems complement each other successfully. Poor discrimination or high false-alert burden is reported without changing the simulation.

## Hybrid Failure Intelligence

The hybrid engine combines the **frozen** supervised failure probability and anomaly score without refitting either detector. Supervised ML recognizes associations in its labeled failure examples; anomaly detection identifies unusual behavior under a learned normal reference. Neither guarantees recognition of every real fault. This remains a hardened synthetic telemetry benchmark.

A naive OR of the two alert streams would import the anomaly detector's large false-alert burden. Instead, SentinelDC compares a small, prespecified set of validation policies, including gated combinations and both original baselines. A hybrid is not automatically preferred: supervised-only can remain selected if it has the best operational utility. The final selected strategy is recorded in `models/hybrid_policy.json`.

```bash
pytest -q
python run_hybrid_pipeline.py
```

Only frozen-model inference on VALIDATION is needed because full validation score caches were not previously saved. No model, scaler, calibrator, or existing detector threshold is refitted or changed. TEST probabilities, anomaly scores and baseline alert flags come from the existing aligned caches. TEST overlap is inspected only after the hybrid policy has been saved.

**Three different quantities:** calibrated failure probability estimates a labeled 24-hour outcome under the synthetic benchmark; anomaly percentile describes relative unusualness; hybrid operational risk is an engineering state used for triage. Neither an anomaly percentile nor a hybrid score is a failure probability.

Anomaly normalization is the right empirical CDF of all VALIDATION anomaly scores, saved in `models/hybrid_anomaly_reference.joblib`. Values below the reference minimum map to 0 and above its maximum to 1. Ties receive their upper percentile: clipped scores can share percentile 1, so an extreme percentile does not guarantee a particular alert frequency. TEST never updates this reference. A weighted policy uses `min(failure_probability / (2 × supervised_threshold), 1)` as its supervised signal; raw probabilities and raw anomaly scores are never directly averaged.

Fifteen candidate configurations span six families:

| Strategy | Candidate behavior |
| --- | --- |
| SUPERVISED_ONLY | Exact existing supervised thresholds, two-hour persistence and six-hour cooldown |
| ANOMALY_ONLY | Exact existing anomaly thresholds, persistence and cooldown |
| STRICT_AGREEMENT | Supervised probability ≥ its threshold AND anomaly percentile ≥0.75, 0.90 or 0.95; supervised emergency also requires that anomaly level |
| ANOMALY_RESCUE | Normal supervised eligibility OR percentile ≥0.995 with probability ≥0.50 or 0.75 times the supervised threshold; instant rescues share the six-hour cooldown |
| WEIGHTED_HYBRID | Supervised weights 0.75 or 0.90, anomaly weight equal to the remainder, hybrid thresholds 0.40 or 0.60; emergency requires supervised emergency and percentile ≥0.95 |
| TWO_STAGE_GATED | Probability ≥0.50 or 0.75 times the supervised threshold, then anomaly percentile ≥0.90 or 0.95; instant exception requires normal supervised signal plus percentile ≥0.995, or supervised emergency plus percentile ≥0.90 |

Regular signals need two consecutive hourly observations; gaps and low readings reset persistence. Rescue observations below the normal supervised threshold do not count toward the supervised persistence streak. A single shared six-hour cooldown controls each candidate's operational notifications, including emergency/rescue alerts. State begins empty for each evaluation cohort. Future readings and outcome metadata are never inputs to policy decisions.

The prespecified validation utility is:

```text
utility = event_detection_rate
          − 0.25 × false_operational_alerts_per_server_day
          + 0.10 × operational_precision
          + 0.10 × fraction_of_all_events_detected_at_least_6_hours_early
```

Rates and precision are fractions, not percentages. Undefined precision or early-warning fractions contribute zero. Event detection already penalizes misses through `1 − missed_events / actual_events`; misses are not penalized twice. One additional false alert per server-day costs 0.25 utility. These fixed engineering weights express a tradeoff and are not industry standards or fitted costs. No coefficient, threshold grid or weight is adjusted after viewing TEST. Highest utility wins; exact ties favor supervised-only, then fewer false alerts, higher median lead time, and candidate ID. Row-level PR-AUC, precision, recall, F1 and F2 are recorded but do not choose the policy.

All policies use common deterministic risk boundaries anchored to the frozen supervised threshold `t`: MEDIUM if probability ≥0.5t or anomaly percentile ≥0.90; HIGH if probability ≥t, percentile ≥0.995, or an operational alert is emitted; CRITICAL if probability ≥t AND percentile ≥0.95; otherwise LOW. Higher states take precedence. An extreme anomaly alone can be HIGH without generating a gated failure alert. Risk labels do not bypass persistence or cooldown. CRITICAL describes multi-signal evidence, not a confirmed failure. Every HIGH/CRITICAL row receives structured, deterministic reason codes; no LLM or SHAP is used.

The complete validation cohort supports operational comparison. The existing supervised calibrator was fitted on an early part of that cohort, so this is development selection evidence rather than a completely independent calibration/model assessment. The TEST cohort was also inspected in earlier stages; current code isolation does not make it a new external benchmark. The hybrid policy and normalization hashes are frozen before cached TEST signals are loaded for final replay.

Failure-event matching retains the established `[failure − 24h, failure)` horizon and TEST cohort, including partial windows and simulator-known future-tail outcomes. One event is counted once, at its first eligible alert. Repeated true alerts after cooldown remain operator workload and are reported separately from unique detections. Early-detection fractions use all actual events; lead-time means and medians condition on detected events. False alerts per server-day use observed hourly rows divided by 24.

| Hybrid artifact | Contents |
| --- | --- |
| `models/hybrid_policy.json` | Frozen selected strategy, parameters, severity boundaries, utility and validation results |
| `models/hybrid_anomaly_reference.joblib` | Validation-only anomaly percentile reference |
| `reports/hybrid_validation_comparison.csv` | All 15 candidates' operational utility and row metrics |
| `reports/hybrid_test_metrics.csv` | Frozen selected policy's TEST operational and row metrics |
| `reports/hybrid_failure_event_analysis.csv` | Unique TEST events and first hybrid alerts |
| `reports/detector_overlap.csv` | Each TEST event classified as both, supervised-only, anomaly-only or neither |
| `reports/hybrid_test_scores.csv` | Full chronological scores, risk states, reasons and alert decisions |
| `reports/hybrid_high_risk_explanations.jsonl` | Structured evidence for every HIGH/CRITICAL row |
| `reports/example_hybrid_risk_report.json` | A deterministic failure example; outcome metadata is reporting-only |
| `reports/hybrid_live_view.csv` | Latest available observation per server, with topology and reasons; a historical snapshot, not a live feed |
| `reports/hybrid_integrity_check.json` | Protected baseline hashes, split/selection provenance and freeze checks |
| `reports/hybrid_prespecified_protocol.json` | Utility and source hashes recorded before validation selection |
| `reports/hybrid_protected_artifacts.json` | Hashes of all pre-existing protected data, models, reports and sources |
| `reports/figures/hybrid_policy_comparison.png` | Validation detection versus false-alert burden |
| `reports/figures/detector_overlap.png` | Unique TEST-event overlap |
| `reports/figures/supervised_vs_anomaly_scores.png` | Fixed-seed representative TEST signal sample |
| `reports/figures/example_hybrid_timeline.png` | Probability, anomaly percentile, risk, first alert and failure |

The demonstration chooses the earliest detected event with sufficient TEST history, otherwise the earliest detected event. If no event is detected, it clearly shows the earliest missed event. It is not chosen to maximize performance. Detector overlap and baseline comparisons are retrospective descriptions only; they cannot justify post-TEST policy tuning. No claim of improvement is made unless the measured tradeoff supports it.

## Root-Cause Intelligence

Prediction identifies risky machines, anomaly detection finds unusual behavior, and hybrid intelligence combines those signals. Root-cause intelligence adds infrastructure dependencies and historical telemetry context to generate **investigation hypotheses**. It neither predicts failure again nor provides guaranteed causal diagnosis. All three detection layers, their thresholds and policies, the generator, dataset, and previous TEST results remain unchanged.

```bash
python -m pip install -r requirements-root-cause.txt
pytest -q
python run_root_cause_pipeline.py
```

NetworkX is the only new dependency. Its requirements are in a separate file so existing frozen dependency manifests remain valid. The pipeline reads saved hybrid outputs; it performs no underlying detector fitting or inference. Use `--window-hours 3` to set the incident window before inference. Window/scoring choices must not be changed to improve TEST signature accuracy.

The typed directed graph is `DATACENTER → RACK → SWITCH → SERVER`. Membership comes from TRAIN topology metadata and is checked for consistency across the observed period. Every non-datacenter node must have one parent, server membership must be unique, and cross-datacenter conflicts fail validation. The graph is saved in `models/infrastructure_graph.graphml`. This dataset has one switch per rack with identical server memberships, so topology alone cannot distinguish rack and switch causes.

**Inference inputs** are an explicit allowlist: topology, timestamps, saved hybrid signals, and numeric telemetry/trends. Failure labels, event IDs, failure types, and actual failure timestamps are excluded. The simulator's failure/shared-condition catalogs are not inference inputs. Outcome metadata is loaded only after the incident reports have been generated and saved.

Operational alerts are grouped by datacenter within fixed UTC three-hour windows. Windows cannot chain across the full TEST period. Each incident is a closed-window hypothesis, available at the exclusive window end; start/end timestamps describe actual included alerts. Results within an unfinished window are provisional. Server histories use the strictly past interval `[t−48h, t)`, with at least 12 observations. Current readings and all future telemetry are excluded from every baseline. Switch/rack context first takes the peer median at each historical timestamp, then applies the same past-only window. Missing history contributes no abnormality evidence.

Per-feature deviation is `(current − past median) / max(past IQR / 1.349, dispersion floor)`. Fixed floors prevent constant signals from producing infinite evidence: packet loss 0.1 percentage points, latency/temperature 1 unit, power/change 20 W, CPU/GPU/memory/disk utilization 5 percentage points, SMART count/change 1, fan speed 100 RPM, and throughput totals 5 MB/s. Server, rack and switch deviations are combined with weights 0.70/0.15/0.15, then divided by 6 and clipped to `[0,1]` (positive or absolute deviation depending on the signature). These values are engineering heuristics, not calibrated probabilities.

Deterministic evidence signatures are:

- NETWORK_LIKE: 40% elevated packet loss, 40% elevated latency, 20% absolute total network-throughput deviation.
- THERMAL_LIKE: 50% elevated temperature, 25% fan-speed deviation, 25% joint CPU/GPU-load and power elevation (geometric combination).
- POWER_LIKE: 60% unusual six-hour power change, 40% power-level deviation.
- DISK_LIKE: 40% elevated SMART count or twelve-hour increase, 30% disk throughput deviation, 30% elevated disk utilization.
- MEMORY_COMPUTE_LIKE: 50% elevated memory, 30% CPU deviation, 20% joint CPU/memory pressure.

For a candidate component, signature strength is the mean across affected servers of each server's maximum signature evidence within the incident. A signature requires strength ≥0.35 and a lead of ≥0.05 over the next signature; otherwise it is UNKNOWN. This aggregation treats servers equally rather than rewarding repeated alerts from one server. Historical baselines can adapt to sustained abnormalities and legitimate workloads can mimic faults; these are limitations of the heuristic.

Shared-component fraction is compared against that component's expected membership **within the incident's datacenter**. Normalized positive membership lift is `max(0, (affected_fraction − background_fraction) / (1 − background_fraction))`; it is zero when background fraction is one. Shared evidence is `0.6 × lift + 0.4 × affected component coverage`. A server candidate uses its fraction of incident servers. Thus simply grouping by datacenter does not create artificial datacenter enrichment.

Candidate confidence is fixed before evaluation:

```text
0.35 × shared-component evidence
+ 0.25 × telemetry-signature strength
+ 0.20 × temporal coherence
+ 0.10 × mean per-server maximum anomaly percentile
+ 0.10 × mean per-server maximum hybrid severity / 3
```

Temporal coherence is `1 − first-alert onset spread / window duration`, clipped to `[0,1]`; a single server gets 0.5 rather than perfect synchronization. LOW/MEDIUM/HIGH/CRITICAL map to 0/1/2/3. Scores are bounded to `[0,1]` and called **root-cause confidence scores**, not statistically calibrated causal probabilities.

A component hypothesis needs score ≥0.55, a non-ambiguous signature, and compatible evidence. Shared candidates need at least two affected servers and 10% coverage; a datacenter needs at least six servers and 50% coverage. Switch hypotheses require a network signature; rack hypotheses require thermal or power evidence; datacenter hypotheses allow network/thermal/power evidence; server hypotheses allow any signature. Those compatibility priors are explicitly heuristic. The highest-scoring eligible candidate wins, with component ID breaking exact ties. Scores ≥0.75 on eligible hypotheses are labeled high-confidence; this is an engineering label only. When nothing qualifies, component, level and signature are UNKNOWN. Its displayed score is the strongest unconfirmed candidate's evidence, not confidence in a diagnosis; blast radius is left undefined rather than invented.

**Blast radius** is graph membership: all servers connected beneath the suspected component. Currently affected component members are counted separately from the incident's total affected servers. Potential additional exposure is connected membership minus currently affected membership. Exposure means dependency, not a prediction that those servers will fail. Disk and memory/compute signatures are not promoted to switch/rack diagnoses simply because servers share a location.

Every incident gets deterministic reason codes, candidate evidence, membership fractions, baseline examples, and a human-readable summary. Major incidents (at least five affected servers or CRITICAL severity) are included in a Markdown summary. Inspection priorities are NETWORK, THERMAL, POWER, STORAGE, MEMORY_COMPUTE or UNKNOWN as appropriate. Outputs recommend **inspection targets**, not component replacement or automated maintenance.

**Evaluation limitation:** the synthetic dataset has server failure types but no verified exact shared-component failure labels. Signature accuracy compares the inferred incident signature with the true type of each alert-covered failure event, once per event at its first hybrid alert inside `[failure−24h, failure)`. UNKNOWN predictions count as incorrect. Events without an eligible alert are reported separately; the conditional accuracy must be read alongside coverage. A grouped incident can contain multiple true failure types. No infrastructure-component accuracy is reported.

A separate post-inference clustering diagnostic compares same-switch/rack/datacenter server-pair concentration in three-hour windows with 100 fixed-seed healthy-observation samples. Each sample matches the number of alerts at each timestamp; pairs are counted once per unique server per time window. The report includes the null mean, standard deviation, enrichment and empirical upper-tail p-value. These descriptive associations neither prove causality nor tune inference. Identical rack/switch membership can produce identical clustering statistics.

| Root-cause artifact | Contents |
| --- | --- |
| `models/infrastructure_graph.graphml` | Typed dependency graph |
| `reports/infrastructure_graph_summary.json` | Node/edge counts and topology limitations |
| `reports/root_cause_incidents.csv` | One row per bounded incident and selected hypothesis |
| `reports/root_cause_details.jsonl` | All candidates, signatures, explanations and baseline examples |
| `reports/root_cause_alert_evidence.csv` | Past-only contextual evidence for saved hybrid alert rows |
| `reports/root_cause_incident_summaries.md` | Human-readable major-incident summaries |
| `reports/root_cause_live_view.csv` | Latest incident window snapshot; historical, not live monitoring |
| `reports/root_cause_signature_evaluation.csv` | One signature comparison per alert-covered failure event |
| `reports/root_cause_signature_summary.json` | Conditional accuracy, coverage and unavailable component accuracy |
| `reports/infrastructure_alert_clustering.csv` | Time-matched healthy-null clustering comparison |
| `reports/example_root_cause_incident.json` | Highest-scoring eligible hypothesis, earliest timestamp tie-break; strongest unconfirmed candidate if none qualifies |
| `reports/figures/root_cause_signature_confusion_matrix.png` | Signature agreement, including UNKNOWN |
| `reports/figures/example_incident_graph.png` | Bounded graph context, highlighting affected servers |
| `reports/figures/example_root_cause_timeline.png` | Up to four affected servers, incident start and saved hybrid alerts |
| `reports/root_cause_prespecified_protocol.json` | Fixed rules and source hashes saved before inference/evaluation |
| `reports/root_cause_integrity_check.json` | Completion summary and frozen-artifact checks |

The graph example shows at most 36 server nodes and includes affected servers and unaffected neighbors; omitted nodes do not change full-graph blast-radius counts. The timeline ends at the incident's last observation and uses no future failure timestamp to construct inference. Because the upstream alert precision is low, many grouped incidents may be non-failure workload behavior. UNKNOWN and poor signature accuracy are scientifically valid outcomes; neither is corrected by tuning on TEST.

## Model Explainability

SHAP describes how features contribute to the frozen supervised model's prediction. In an operational setting, these contributions help an operator inspect why a server received elevated model risk rather than treating a score as a diagnosis. Run `.venv/bin/python run_explainability_pipeline.py` after installing `requirements.txt`. This command does not train or select any model, threshold, alert policy, or root-cause rule.

The saved `CalibratedFailureModel` contains a TRAIN-fitted preprocessing pipeline and LightGBM classifier, followed by validation-fitted Platt calibration. The explainability subsystem uses exactly the saved feature ordering and median imputation, with no preprocessing fitting. `shap.TreeExplainer` explains the underlying LightGBM positive-class **raw log-odds**, using `tree_path_dependent` references from the frozen trees' training path counts. SHAP base value plus feature contributions reproduces the raw model margin. SHAP values are **not changes in calibrated probability**. The final displayed failure probability is obtained separately from the saved calibration wrapper and checked against saved TEST probabilities. The calibration layer is not represented as a tree. See the [TreeExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).

Global analysis uses a deterministic 2,000-row TEST sample, sampled proportionally within healthy and failure-window classes with seed 42. Labels are used only to preserve approximate prevalence in this post-hoc sample. It is not used for model selection, calibration, thresholds or tuning. Mean absolute SHAP ranks feature **influence in either direction**, not just risk increases. Beeswarm colors show feature values; position shows signed influence on the model margin. The direction summary compares high/low quartile groups for the top 20 features. When quartile cutoffs tie, the high group uses strictly greater values; absent groups are reported as unavailable. These are descriptive associations, sensitive to correlated features and interactions, not hard-coded engineering expectations.

The subsystem explains a deterministic sample of up to 100 observations per HIGH/CRITICAL hybrid level, every distinct saved supervised-or-hybrid operational warning row, and the exact latest snapshot used by `hybrid_live_view.csv`. All selected rows are deduplicated and computed once in batches of 512. The explanation cache is saved **before** reading failure-event reports. Event reporting then joins each detected event to the earliest warning across the supervised and hybrid systems, once per event. Actual failure timestamps and failure types never enter model preprocessing or SHAP. The example is the earliest first warning with HIGH/CRITICAL hybrid risk, breaking ties by event ID; its waterfall explains the frozen model, not the known failure outcome.

Local reports separate risk-increasing drivers from risk-decreasing factors and use deterministic display names. Feature values are the actual preprocessed model inputs (imputed when missing). The combined intelligence report reuses saved hybrid signals and root-cause hypotheses, and explains anomaly deviations with the saved robust statistics without refitting. Root-cause hypotheses are available at their closed incident window's `inference_as_of`, which may be later than the selected first warning. A null association means no applicable saved incident. Weak or UNKNOWN hypotheses remain explicitly unconfirmed; dependency blast radius does not predict additional failures.

**SHAP explains how the model arrived at its prediction. SHAP does not prove that a feature caused a hardware failure.** Root-cause intelligence proposes infrastructure hypotheses; neither model explanation nor infrastructure association establishes causation. Root-cause signature inference remains experimental: the frozen diagnostic found 0/41 correct signatures (0%), with 36 UNKNOWN and five incorrect non-UNKNOWN event classifications; 158/180 incidents were UNKNOWN/low-evidence. The 25 racks and 25 switches have identical memberships in pairs, so exact rack-versus-switch causal attribution is not identifiable from this topology. Those results and rules are unchanged during this stage. This remains a hardened synthetic data-center benchmark, not evidence of production effectiveness.

Artifacts: `shap_global_importance.csv`, `shap_feature_direction_summary.csv`, `shap_local_explanations.jsonl`, `failure_event_explanations.csv`, `explainability_live_view.csv`, and `example_sentinel_intelligence_report.json` live in `reports/`. Global bar, beeswarm and example waterfall figures live in `reports/figures/`. `shap_global_sample.csv` records sample keys and class mix; `shap_explanation_cache.npz` and its keys CSV provide a reusable numeric cache. `explainability_protected_artifacts.json` captures the pre-stage data, models, source, tests and reports; `explainability_integrity_check.json` verifies them after completion.

The explicitly requested `shap>=0.46,<0.49` addition to `requirements.txt` is an authorized dependency change. Older manifests are preserved verbatim and may still contain the previous requirements.txt hash; that dependency-only difference is not a model or benchmark change. The new explainability integrity check protects the frozen detection and root-cause artifacts independently of that requirements change. Python 3.11 arm64 uses SHAP wheels; no source-built OpenMP dependency was added for this stage.

In the initial frozen-model explanation run, 2,000 global observations included 31 failure-window rows; 7,571 unique rows supplied all outputs, including 200 local explanations, 41 detected-event explanations, and the 500-server snapshot. Uptime dominated mean absolute SHAP influence. Its high-value group had mean SHAP +0.879 versus −4.986 in the low-value group. Average packet loss (6h) had +0.047 versus −0.040, and average temperature (6h) had +0.111 versus −0.184. These latter associations are directionally consistent with elevated operational stress, but do not validate causation. Day of week also ranked highly, illustrating that the model uses temporal associations as well as telemetry; this is not evidence that weekday causes failure. No model or rule was changed after inspecting these findings. The selected example (`server_0038`) retains UNKNOWN/low-evidence root-cause status.
