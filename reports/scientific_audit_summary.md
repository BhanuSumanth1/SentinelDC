> **Historical easy-benchmark audit.** This report predates heterogeneous-v2 and is not an audit of the hardened data. Original evidence and source snapshots are preserved in [baseline_reference](baseline_reference/). See [hardened_dataset_realism.md](hardened_dataset_realism.md) for current measured results. Other original audit CSVs, except the regenerated single-feature/overlap files, describe the earlier dataset.

# SentinelDC scientific leakage and realism audit

**Verdict: TOO EASY — GENERATOR SHOULD BE HARDENED**

The unusually strong results are reproducible under the current synthetic task and evaluation rules. No direct target leakage or future-observation leakage was found in the inspected implementation. The benchmark is nevertheless too easy to support realistic failure-prediction claims: three individual features exceed 0.97 ROC-AUC, generic fault signatures are shared across failure types, and the incident scorer credits any alert in a fixed 24-hour window.

This was a diagnostic-only audit. No model was fit, selected, or tuned; no threshold, generator, data, or existing benchmark report was modified. SHA-256 checks confirm all **56 original project files** remained byte-identical. **47 tests passed.** The model was loaded for frozen inference because per-row test probabilities had not been persisted. Its test PR-AUC and all per-event lead times/risk maxima were checked against existing artifacts.

## Category assessment

| Category | Status | Evidence |
| --- | --- | --- |
| Target leakage | PASS | All 24 inputs and fitted schemas exclude target, failure type, event ID, and actual failure timestamp. Metadata removal leaves engineered features identical. |
| Future-data leakage | PASS | Reviewed feature dependencies use current readings and past-only filling. Prefix truncation and future-value perturbation leave earlier features unchanged; saved imputer/scaler statistics match training data. |
| Temporal split | PASS | 70/15/15 timestamp windows, 24-hour purges at both boundaries, no shared incident IDs. All 500 servers recur in train and test; this is known-server temporal evaluation. |
| Single-feature dominance | WARNING | Packet loss AUC 0.977003; error indicator and latency also exceed 0.97. This is strong synthetic predictability, not automatically leakage. |
| Failure-window realism | WARNING | All 300 incidents start their positive label window at 24h globally; 43/63 alert on their first test-positive row. The median is capped by the evaluation horizon. |
| Healthy/failure distribution overlap | WARNING | No feature meets the declared near-perfect separation rule, but latency/error-indicator overlap is only about 13%; persistent non-failure stress is underrepresented. |
| Failure-event diversity | WARNING | No exact or flagged near copies; however, all incidents use one 48h ramp family, narrow severity scaling, shared signatures, and at most one fault per server. |
| Event-level evaluation | WARNING | Mathematically consistent, but any-warning detection offers repeated opportunities. Partial windows and known future-tail outcomes are included; 100% synthetic detection is not production evidence. |

## Scope and leakage evidence

[leakage_audit.txt](leakage_audit.txt) records **32 PASS checks** and 0 failures. It lists all raw inputs, every engineered-feature dependency, persisted schemas, training-only transformations, chronological boundaries, and runtime perturbation checks. These checks and source inspection provide evidence, not a universal proof against all possible shortcuts.

The explicit input allowlist is in `src/config.py:29`; training uses `train[FEATURE_COLUMNS]` in `src/train.py:92`. The nine engineered features are calendar fields, same-row sums/products, fixed-threshold flags, a fixed temperature reference, and a current SMART/packet-loss indicator. No rolling calculation, backward fill, future timestamp, or future status is used. Causal filling is in `src/preprocess.py:43–45`.

The generator necessarily knows the latent event schedule when synthesizing its precursor telemetry. That is backward construction of a synthetic joint distribution, not an observed feature being computed from a future production outcome. It is still the main realism concern: event timing governs a reliably increasing stress signature. Uptime resets at/after the simulated failure (`src/generate_data.py:92–93`), rather than exposing a countdown before it.

Source review confirms model/threshold selection receives validation only (`src/evaluate.py:84–98`) and final inference uses the persisted selected model and threshold (`src/evaluate.py:157–158`). Split prevalence and schema-validation checks read labels for reporting/consistency, not for fitting. The existing isolation tests remain passing.

**Historical holdout limitation:** the same synthetic dataset was used for earlier baseline development and previous model comparison, and its test outcomes have now been inspected repeatedly. Current computational isolation passes; this does not make the test partition a fresh, statistically independent final benchmark. These post-hoc diagnostics must not be presented as additional prospective performance. Future hardening should be assessed on a newly frozen realization or independent real dataset, not optimized to this test set.

## Single-feature predictability: test only

All 54,000 test rows were used, with no validation/training rows and no missing values for the 24 inputs. The raw feature value is the score; `predictive_strength = max(ROC-AUC, 1 − ROC-AUC)` accounts for inverse association without fitting a classifier. Flags at strength ≥0.90 request realism review; they do not diagnose leakage. The direction is selected descriptively using test labels and must not be deployed as a tuned rule.

| Feature | ROC-AUC | Predictive strength | Review flag |
| --- | --- | --- | --- |
| packet_loss_pct | 0.977003 | 0.977003 | YES |
| error_rate_indicator | 0.974326 | 0.974326 | YES |
| network_latency_ms | 0.973396 | 0.973396 | YES |
| smart_error_count | 0.890306 | 0.890306 | NO |
| temperature_c | 0.793919 | 0.793919 | NO |
| temperature_deviation | 0.793919 | 0.793919 | NO |
| fan_speed_rpm | 0.783873 | 0.783873 | NO |
| uptime_days | 0.755090 | 0.755090 | NO |
| disk_utilization | 0.720262 | 0.720262 | NO |
| cpu_memory_interaction | 0.686403 | 0.686403 | NO |

Full results: [single_feature_auc.csv](single_feature_auc.csv). The engineered error indicator includes packet loss, so these are not three independent sources of evidence. Temperature and temperature deviation also have identical rankings by construction. No sampling uncertainty interval is asserted: nearby rows and repeated incident observations are correlated.

## Why the median warning is exactly 24 hours

The label is defined as `0 < actual_failure_timestamp − timestamp <= 24h`. That is a correct definition of this prediction task, not evidence of leakage. Every one of the 300 simulated incidents has its first global labeled row exactly 24h before failure. A model cannot receive more than 24h lead-time credit under the incident scorer, even if it fired earlier.

Among the 63 test incidents:

- 58 have their first test-positive label exactly 24h before the event; five start later because the beginning of their warning window precedes test.
- 43 (68.25%) alert on that first test-positive row, including the five left-truncated windows.
- 38 (60.32%) have exactly 24h model lead time. The remaining 20 incidents alert later than their first eligible label.
- Median label-window lead time is 24.00h; median credited model lead time is 24.00h.
- The frozen model also emits 198 alerts in 1192 available test observations 24–48h before these failures (16.61%). They are outside the target horizon and correctly count as row false positives. They account for 198/219 = 90.41% of all test false alerts.

The classifier is therefore not merely switching on when the label switches at 24h. It responds to a precursor already developing earlier. Restricting credit to the label window creates the lead-time ceiling; strong precursors make that ceiling easy to reach.

| Lead hours | First test-positive label: incidents | First credited model alert: incidents |
| --- | --- | --- |
| 6 | 1 | 1 |
| 7 | 1 | 1 |
| 14 | 1 | 1 |
| 18 | 0 | 1 |
| 19 | 0 | 2 |
| 20 | 0 | 1 |
| 21 | 1 | 5 |
| 22 | 0 | 3 |
| 23 | 1 | 10 |
| 24 | 58 | 38 |

[failure_window_audit.csv](failure_window_audit.csv) uses test-only labels and alerts for the requested lead-time fields, with additional global-label fields for boundary context. [all_event_window_audit.csv](all_event_window_audit.csv) checks all 300 global incidents. Alerts are counted at the saved 0.90 threshold without changing it.

## Degradation realism

[failure_degradation_analysis.csv](failure_degradation_analysis.csv) analyzes the **63 test incidents**, retrospectively retrieving same-server history from the existing raw data up to 72h before each event. Some descriptive history precedes the test boundary; `context_rows_before_test`, observation counts, and represented-event counts make that explicit. No historical context is used to augment test predictions or alter fitting. Bins use `lower < hours_before_failure <= upper`, giving nonoverlapping hourly observations. Statistics are observation-weighted; partial tails change the event composition of later bins.

The source starts every fault precursor at 48h and scales the same linear envelope by a uniform factor in [0.7, 1.25] (`src/generate_data.py:61–62`). Each type receives the same packet-loss, latency, and SMART stress components (`:74–76`), plus its type-specific signature. There is no telemetry assignment branch at 24h; that boundary only controls labels/metadata. Adjacent-hour diagnostics confirm noisy variation rather than a universal sudden change at 24h.

Mean packet loss (%) by failure type illustrates both a gradual trend and a universal shortcut:

| Failure type | 48-72h | 24-48h | 12-24h | 6-12h | 0-6h |
| --- | --- | --- | --- | --- | --- |
| disk_failure | 0.091 | 0.221 | 0.384 | 0.483 | 0.576 |
| memory_failure | 0.089 | 0.219 | 0.420 | 0.499 | 0.532 |
| network_failure | 0.099 | 0.680 | 2.559 | 4.088 | 5.309 |
| overheating | 0.091 | 0.219 | 0.390 | 0.481 | 0.565 |
| power_failure | 0.076 | 0.199 | 0.387 | 0.486 | 0.530 |

For disk failures, mean SMART errors rise from 0.109 (48–72h) through 2.160, 8.437, 13.760 to 17.375 (0–6h). Overheating mean temperature progresses from 46.17°C to 54.36, 66.75, 73.26 and 75.83°C. Power-failure mean power rises from 242.13W to 288.77, 351.29, 388.31 and 427.51W. Standard deviations and quartiles in the CSV show noise. Not every signal is monotonic for every type; workload contributes variability.

At 25h versus 24h, mean packet loss is 0.310 versus 0.321 for disk failures and 0.380 versus 0.328 for overheating: the latter decreases despite the label changing. [failure_boundary_diagnostics.csv](failure_boundary_diagnostics.csv) includes 26, 25, 24, 23, and 22h for all eight signals and five types. This is evidence against a universal discontinuity at exactly 24h, not evidence of calibrated real-world physics.

The problem is consistency and contrast: every event has a similar onset/duration and generic stress, while non-failing operations rarely sustain comparable packet loss or SMART activity. There are no abrupt/unheralded failures, recovery-without-failure precursor episodes, or recurrent faults. About 300 of 500 servers experience an incident during the generated horizon (including the future tail); 1.96% positive rows should not be mistaken for realistic fleet incident incidence. Power is computed from base CPU before later fault CPU changes, another simplified physical relationship rather than target leakage.

## Healthy/failure overlap and hard negatives

Here “healthy” means label 0: no failure in the next 24h. It can still include degradation 24–48h before a failure, earlier/later incident history, or benign stress. We separately inspect servers with no incident anywhere in the simulated dataset; this retrospective grouping is diagnostic only.

The overlap coefficient sums the minimum of two normalized marginal histograms. It ranges from 0 (no overlap) to 1 (identical bin masses). Continuous features use 25/50/100 equal-width pooled-range bins to expose bin sensitivity; low-cardinality features use exact value frequencies. A near-separation flag is set for 50-bin overlap <0.10, best single-cut balanced accuracy ≥0.95, or predictive strength ≥0.99. The single-cut metric is descriptive only; its threshold is never saved or applied to the model. No feature meets this strict rule and none has disjoint observed ranges, so the distributions are not perfectly separable.

Nevertheless, the strongest features have limited overlap:

| Feature | 25-bin overlap | 50-bin overlap | 100-bin overlap | Diagnostic best balanced accuracy |
| --- | --- | --- | --- | --- |
| network_latency_ms | 0.274 | 0.132 | 0.132 | 0.941 |
| error_rate_indicator | 0.186 | 0.134 | 0.104 | 0.910 |
| packet_loss_pct | 0.214 | 0.179 | 0.116 | 0.943 |
| smart_error_count | 0.289 | 0.289 | 0.289 | 0.855 |
| uptime_days | 0.416 | 0.349 | 0.271 | 0.769 |

Healthy abnormal observations on the test partition:

| Signal ≥ threshold | No failure next 24h: count / rows | Never-failing servers: count / rows |
| --- | --- | --- |
| cpu_utilization ≥ 90 | 47 / 52,681 | 26 / 21,600 |
| temperature_c ≥ 75 | 3 / 52,681 | 1 / 21,600 |
| packet_loss_pct ≥ 1 | 78 / 52,681 | 3 / 21,600 |
| smart_error_count ≥ 5 | 63 / 52,681 | 0 / 21,600 |
| smart_error_count ≥ 1 | 6,568 / 52,681 | 2,441 / 21,600 |

Thus healthy abnormalities exist, but severe temperature excursions, packet-loss episodes, and larger SMART counts are scarce on never-failing servers. Most model false alarms are already on genuine fault trajectories just beyond the chosen 24h horizon. Files: [healthy_failure_overlap.csv](healthy_failure_overlap.csv), [healthy_abnormal_observations.csv](healthy_abnormal_observations.csv).

## Duplicate and structure checks

| Check | Result |
| --- | --- |
| exact duplicate rows | 0 |
| duplicate server timestamp keys | 0 |
| duplicate numeric telemetry vectors | 0 |
| full 24h events compared | 288 |
| partial events excluded | 12 |
| event pairs compared | 41328 |
| identical event pairs | 0 |
| near duplicate pairs | 0 |
| maximum shape correlation | 0.976540 |
| minimum standardized shape rmse | 0.993034 |
| repeated exact server feature traces | 0 |

Full 24-hour event windows were aligned by hours before failure across eight signals. Exact matches compare raw arrays. Shape comparison subtracts each event's per-signal mean and divides by the corresponding healthy-test standard deviation; pairwise normalized dot products then measure correlation. Near-copy flags use correlation ≥0.995 **or** standardized shape RMSE ≤0.25. These explicit heuristic thresholds are not a formal generative-diversity test. Partial global warning windows were excluded, not padded. [event_pattern_similarity.csv](event_pattern_similarity.csv) records every compared pair.

SHA-256 fingerprints also compared entire same-length server traces for each of the 15 raw signals, with no repeated exact traces. Source review confirms one RNG seeded once outside the server loop, rather than restarting the same random stream for each event. This rules out observed exact copying, not every possible stochastic dependence. The maximum shape correlation of 0.9765 still reflects strong shared mechanisms. Exact-copy checks pass; mechanistic diversity warrants WARNING because noise is laid over the same fixed family of ramps.

## Temporal and incident evaluation checks

| Partition | Start UTC | End UTC | Rows | Positive fraction |
| --- | --- | --- | --- | --- |
| training | 2025-01-01T00:00:00+00:00 | 2025-01-20T23:00:00+00:00 | 240000 | 1.8062% |
| validation | 2025-01-22T00:00:00+00:00 | 2025-01-25T11:00:00+00:00 | 42000 | 2.2571% |
| test | 2025-01-26T12:00:00+00:00 | 2025-01-30T23:00:00+00:00 | 54000 | 2.4426% |

Both boundaries purge 24h of potential label overlap. There are no common timestamps or event IDs across retained partitions. **500/500 test servers also occur in training.** This tests later behavior on known infrastructure, not transfer to unseen hardware or data centers. Identifiers are excluded, but shared server operating distributions may still make prediction easier.

The event calculations match saved outputs: 63/63 incidents detected; credited lead times are nonnegative and no more than 24h. A single correct warning suffices, so 105 false-negative rows can coexist with zero missed incidents. Repeated correlated observations offer multiple detection opportunities; no independence assumption such as `1 − (1 − recall)^24` is justified.

The cohort includes all event IDs with any positive test row: 46 have complete warning windows, 17 are partial, and 13 events occur after the last test observation with outcomes known only because they are simulated. These flags are retained in the event CSV. The 100% point estimate has not been externally validated. The horizon-capped lead time is the first **eligible** test warning, not necessarily the first warning the model ever issued; the 198 earlier out-of-horizon alerts demonstrate the distinction.

## Reproduction and preserved state

From the existing project root with its Python 3.11 environment active:

```bash
python -m src.scientific_audit
pytest -q
```

The audit command regenerates numeric diagnostic reports and `leakage_audit.txt`; this Markdown document is the reviewed interpretation of this run. Do not run `run_pipeline.py` for this diagnostic stage because that regenerates benchmark artifacts. Test output is saved separately as [scientific_audit_tests.txt](scientific_audit_tests.txt); the original test and pipeline reports remain unchanged. [audit_integrity_manifest.json](audit_integrity_manifest.json) records byte-identical SHA-256 hashes for the original files. [scientific_audit_console.txt](scientific_audit_console.txt) contains the top 10 and both requested lead-time distributions.

## Final classification

**TOO EASY — GENERATOR SHOULD BE HARDENED**

There is no evidence of current computational target/future-data leakage, nor of copied trajectories. The strong scores instead come from clean shared precursors, a fixed onset/ramp family, scarce sustained benign lookalikes, and an event metric that rewards any warning and caps lead time at 24h. The software remains useful as a portfolio workflow demonstration, but the synthetic benchmark is insufficiently challenging for a claim of realistic failure forecasting. Generator hardening and fresh independent evaluation are recommended for a later authorized stage; neither was performed in this audit.
