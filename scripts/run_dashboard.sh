#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/streamlit run dashboard/app.py --server.address=127.0.0.1 --server.port=8501
