# Hardened dataset realism report

Generator: `heterogeneous-v2`, seed 42. This is a synthetic portfolio benchmark, not validated data-center reliability physics. Generator parameters were specified before evaluating the regenerated test data; they were not adjusted to attain a score, AUC cutoff, or missed-event count.

## Generated dataset

| Quantity | Measured value |
| --- | --- |
| Telemetry rows | 360,000 |
| Servers | 500 |
| Actual incidents in the full dataset | 300 |
| Failure-positive observations | 1.9656% |
| Server-local recovering disturbances | 12077 |
| Disturbances entirely outside any 24h warning window | 11772 |
| Shared rack/switch condition episodes | 395 |
| TEST incidents | 43 |

A healthy transient is one finite server-local disturbance whose every observed row has label 0; it is counted once, not once per telemetry row. Other exogenous disturbances may overlap an unrelated fault warning window. Shared episodes are counted separately, once per affected rack or switch, rather than once per server. Catalog counts identify injected mechanisms, not model alarms or proof that all samples crossed a fixed abnormality threshold. No difficult observations were removed to adjust prevalence.

Incident incidence is deliberately accelerated: achieving about 1–3% positive hourly rows with 24-hour labels implies far more incidents per server-month than many real fleets. At most one incident per server and immediate simulated repair remain simplifying assumptions.

## Changes from the original benchmark

- Persistent, heterogeneous normal CPU, memory, temperature, latency, packet loss, power, disk activity, and fan profiles prevent a universal absolute baseline. Normal temperature spans 37.1–76.0°C in this run.
- Recovering CPU/memory/thermal/network/disk/SMART/power/fan disturbances create non-failure lookalikes. Shared switch congestion and rack thermal/power changes affect neighboring servers without determining their fault labels.
- Fault onset is drawn by type from a bounded lognormal distribution (4–72h); event strength, slope, affected channels, correlated noise, and intermittent recovery vary independently. A short-onset failure can have many normal-looking positive-label rows. Some weak or sparse signatures can be hard to detect.
- Fault-specific mechanisms replace the universal packet-loss/latency/SMART ramp. Disk faults need not disturb networking, and network faults need not disturb temperature. Current CPU and disk load influence temperature/power/fan through noisy nonlinear relationships.
- Eight past-only trends supplement the 24 original inputs. Means use `(t−6h, t]`; differences use the same server's exact timestamp at t−6h or t−12h. Missing lag observations remain NaN for a training-fitted imputer. Identifiers and all outcome metadata remain excluded.

Observed onset variation by type:

| Failure type | Events | Min onset h | Median onset h | Max onset h | Unique onsets |
| --- | --- | --- | --- | --- | --- |
| disk_failure | 81 | 10 | 37.0 | 72 | 44 |
| memory_failure | 44 | 11 | 29.0 | 69 | 28 |
| network_failure | 50 | 4 | 13.0 | 67 | 21 |
| overheating | 69 | 5 | 24.0 | 72 | 39 |
| power_failure | 56 | 4 | 7.5 | 31 | 16 |

The target remains **an actual event in (t, t + 24h]**, independent of when degradation begins. The four simulation catalogs in `data/raw/` are provenance only: server profiles, failure onset/severity, benign episodes, and shared conditions are never joined into model features.

## Frozen final test results

Train/validation/test retain chronological 70/15/15 timestamp windows and a 24h purge before each new partition. All estimators, imputers, and scaling fit on train only. Model selection and operating threshold selection use validation only. The selected model is scored once on test; event analysis reuses those scores. The single-feature diagnostic computes rankings directly from held-out features without fitting alternative models or changing decisions.

| Metric | Measured value |
| --- | --- |
| Selected model | random_forest |
| Selected threshold | 0.10 |
| Validation PR-AUC | 0.080437 |
| Final TEST PR-AUC | 0.137729 |
| Final TEST ROC-AUC | 0.736976 |
| Final TEST recall | 0.992840 |
| Final TEST precision | 0.023032 |
| Final TEST F1 | 0.045020 |
| Strongest individual feature | uptime_days |
| Raw-direction single-feature ROC-AUC | 0.792474 |
| Direction-adjusted single-feature strength | 0.792474 |
| Detected / missed TEST incidents | 43 / 0 |
| TEST event detection rate | 100.00% |
| Median warning lead time | 24.00 hours |
| Mean warning lead time | 22.35 hours |
| Lead time p25 / p75 | 24.00 / 24.00 hours |

**Operational interpretation:** the frozen rule selected threshold 0.10 and generated 35,291 false-positive test rows, or 663.84 false alerts per 1,000 healthy observations. Precision is 2.30%. Event detection and lead time must not be read as useful operational success without this alert burden. The pre-existing recall constraint can favor a very low threshold when benign/failure distributions overlap; that rule was preserved rather than changed after inspecting test performance.

The model test ROC-AUC (0.7370) does not exceed the strongest single-feature direction-adjusted ROC-AUC (0.7925). Thus a benefit from combining features and trends is not automatically established by adding them. Model Average Precision (0.1377) should instead be compared with the test positive prevalence (0.0155) for a no-skill ranking reference. No score was forced downward or upward.

`uptime_days` can retain a structural shortcut: after a simulated repair the single-incident-per-server model cannot fail again in this realization. This is not future-data leakage, but can make low post-repair uptime unusually predictive of no subsequent incident. Addressing recurrent faults or removing that simplification requires a separately assessed future change.

Single-feature realism warning (strength ≥0.90): **none**. These flags diagnose predictability, not automatically leakage. The model ROC-AUC and strongest individual ROC-AUC are comparable ranking summaries; model PR-AUC is a different metric and must not be compared numerically to a single-feature ROC-AUC. No raw-only versus trend-feature ablation was run, so these results do not isolate the causal contribution of trends.

The original benchmark's strongest individual feature was packet loss (ROC-AUC 0.9770), final PR-AUC 0.9497, event detection 100%, and median lead time 24h. Those measurements came from a different, easier data realization and are historical references, not scores to match. The original audit and source snapshots are preserved under `reports/baseline_reference/`. Latest results must come from the current CSVs and this report.

Event detection credits at least one positive warning in the event's available test `[failure−24h, failure)` window. Lead-time statistics condition on detected incidents; missed events have no artificial zero lead time. Partial windows and simulated future-tail outcomes remain flagged in the incident CSV. A median at 24h, if observed, is still limited by the scoring horizon. The test cohort has 18 partial warning windows and 12 failures after the last test observation with simulator-known outcomes.

## Remaining scientific limits

This revised generator uses plausible mechanisms rather than an empirically calibrated physical reliability model. All servers still recur across time partitions, scenarios come from a known simulator, and real production generalization remains unproven. Threshold rules prioritize recall and can produce substantial false alerts; incident detection must be interpreted with row precision and alert burden. Scores remain uncalibrated. Trend means are not server-normalized residuals, and sudden failures may lack enough precursor evidence. A genuinely independent future dataset and event-level alert-cost study are needed before operational claims.
