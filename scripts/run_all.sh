#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
api_pid=''
cleanup() {
  if [ -n "$api_pid" ]; then
    kill "$api_pid" 2>/dev/null || true
    wait "$api_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 &
api_pid=$!
.venv/bin/streamlit run dashboard/app.py --server.address=127.0.0.1 --server.port=8501
