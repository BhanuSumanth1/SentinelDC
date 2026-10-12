# Five-minute SentinelDC demo

## Before the call

From the project root activate `.venv`, run `./scripts/run_all.sh`, and open http://localhost:8501 and http://localhost:8000/docs. Alternatively use `docker compose up --build` with Docker running. Confirm `/health` and `/summary` respond. Use the saved bundle; do not train or regenerate artifacts. Snapshot dates are historical.

## 0:00–0:45 — Executive dashboard

“SentinelDC is a research prototype on 360,000 synthetic observations from 500 servers. This is a read-only view of frozen results.” Show topology KPIs and 41/43 hybrid event detection with 21-hour median warning among detected events. Immediately qualify that event coverage is not alert precision. Show the combined intelligence panel and its UNKNOWN root-cause field.

## 0:45–1:35 — Inspect a high-risk server

Open Live Server Monitor, choose HIGH or CRITICAL, then inspect a server. Explain calibrated failure probability versus anomaly percentile. Show saved threshold, hybrid reasons and top SHAP drivers. Switch telemetry from temperature to packet loss and point out saved alert markers. State that no model runs when changing filters and the snapshot is not live ingestion.

## 1:35–2:20 — Explain model behavior

Open Model Explainability. Show global SHAP importance and the beeswarm. Uptime has the largest influence; explain that this may reflect simulated temporal associations. Inspect one saved local explanation and distinguish positive/negative log-odds contributions from calibrated probability. “SHAP explains the model, not what physically caused failure.”

## 2:20–3:10 — Failure-event detection

Open Failure Intelligence, filter detected events and inspect first-alert timing. Show supervised/anomaly overlap to explain complementary evidence. Note that event matching counts each failure once, whereas operational precision concerns alerts. Some failure windows are partial or end beyond TEST telemetry; the report preserves these flags.

## 3:10–4:00 — Infrastructure hypothesis

Open Incident / Root-Cause Explorer and select an incident. Show affected servers, reason codes, heuristic confidence and blast radius. UNKNOWN is an honest result. Fixed example figures are labeled with their own incident ID. “One switch per rack creates attribution ambiguity; component causality is not labeled.”

## 4:00–5:00 — Limitations and engineering

Open Model Performance. Show approximately 2.83% hybrid operational precision and 3,780 false operational alerts alongside high event coverage. Root-cause signature accuracy is 0/41, with 36 UNKNOWN. Explain chronological splitting, calibration and frozen evaluation. Finish with `/docs`: read-only FastAPI, validated shared loaders and Docker packaging—not a deployed production monitoring product.

Stop local services with Ctrl-C. For Compose, run `docker compose down`. Never replace weak benchmark results for a demo. Capture real screenshots into the README's documented filenames if desired.
