#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../backend"
export ARGUS_CALIBRATION_PATH="../data/evaluation_validation/calibration.json"
if [ -x .venv/bin/uvicorn ]; then exec .venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000; fi
exec uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
