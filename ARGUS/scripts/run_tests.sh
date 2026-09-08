#!/usr/bin/env bash
set -euo pipefail
(cd backend && pytest)
(cd frontend && npm run test:run)
