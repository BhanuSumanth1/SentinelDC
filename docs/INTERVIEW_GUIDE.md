# Interview guide

**Why PR-AUC?** Failure-window observations are rare. Precision-recall evaluation highlights positive-class retrieval and false positives; PR-AUC should be compared with prevalence and supplemented by event and alert metrics.

**Why chronological splitting?** Random splitting can mix adjacent telemetry and overlapping outcome windows. Forward splits with 24-hour purges better simulate deployment and avoid outcome-window overlap.

**Why not accuracy?** Predicting healthy almost everywhere can score highly on accuracy while missing failures. Recall, precision, PR-AUC, lead time and alert burden expose that failure.

**Why anomaly detection?** Healthy-behavior modeling can detect unusual telemetry patterns without needing labels for every failure mode. Unusualness is not automatically failure and can increase false alarms.

**Why a hybrid model?** Supervised and anomaly signals detect overlapping but different events. Validation selected the combination; frozen TEST shows greater coverage, but precision remains low. Hybrid is a trade-off, not an unqualified improvement.

**Why calibration?** A ranking score is not necessarily a meaningful probability. Validation-fitted Platt calibration adjusts the LightGBM score to observed prevalence; Brier score and reliability curves assess calibration. Calibration does not guarantee production validity.

**What is alert fatigue?** Frequent low-value warnings burden operators and reduce trust. Persistence suppresses single-reading noise and cooldown reduces duplicate incidents; neither fixes poor model precision on its own.

**What does SHAP mean?** It attributes a model prediction relative to a reference output across input features. Here tree-path-dependent SHAP values sum with a base value to the frozen LightGBM raw log-odds. Displayed calibrated probability is separate.

**Why isn't SHAP causal?** It explains learned associations and model behavior, including correlations and simulation artifacts. It does not estimate physical interventions or rule out confounding.

**Why is root-cause output only a hypothesis?** There are no exact shared-component failure labels. Rack and switch memberships coincide. Deterministic signatures are weak: 0/41 correct with 36 UNKNOWN. The module provides inspectable evidence rather than confirmed diagnoses.

**What would you change with real telemetry?** Establish label provenance and timestamps; validate across time/sites; monitor drift; measure operator alert budgets and intervention cost; improve component-level ground truth and feedback. Tune only with proper validation, never the held-out test outcomes.

**What is actually deployed?** A local read-only FastAPI/Streamlit demo and Docker configuration over saved research artifacts. No production deployment, streaming ingestion, online predictions, public authentication or real-company telemetry is claimed.

**How are results protected?** Existing artifacts are SHA-256 fingerprinted before serving work. Tests and final integrity checks ensure UI changes do not modify data, policies, model objects or scientific outputs.

**How is serving lightweight?** It loads CSV/JSON/GraphML and figures, validates joins, and caches reads. Serving dependencies exclude model training libraries. Containers mount artifacts read-only and never run training at startup.
