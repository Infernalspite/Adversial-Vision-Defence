#!/usr/bin/env bash
set -euo pipefail
npm --prefix "$(dirname "$0")/../frontend" run dev -- --host 0.0.0.0
