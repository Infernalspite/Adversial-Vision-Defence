#!/usr/bin/env bash
set -euo pipefail
python3 -m venv backend/.venv
backend/.venv/bin/python -m pip install --upgrade pip
backend/.venv/bin/pip install -r backend/requirements.txt
npm --prefix frontend install
