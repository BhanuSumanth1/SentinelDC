# Resume bullets — synthetic research prototype

- Built SentinelDC, a reproducible failure-intelligence research prototype over 360,000 hourly observations from 500 simulated servers, combining calibrated LightGBM, healthy-behavior anomaly detection and validation-selected hybrid alert policies with chronological evaluation.
- Evaluated a frozen hybrid system detecting 41/43 TEST failure events with a 21-hour median warning among detected events; documented the 2.83% operational precision and false-alert burden, adding SHAP explanations and graph-based infrastructure hypotheses without claiming causal diagnosis.
- Developed a read-only Streamlit dashboard and FastAPI service over frozen research artifacts, with Docker Compose packaging, interactive telemetry views, validated report joins and regression/integrity checks; maintained explicit separation between synthetic benchmark results and production claims.

Use “Docker Compose packaging,” not “production container deployment,” unless independently deployed and measured. Do not claim real data-center savings, uptime improvements or accurate root-cause identification. Final test count and runtime verification status are recorded in `reports/final_project_integrity.json`.
