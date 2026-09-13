#!/usr/bin/env bash

set -euo pipefail

echo "Starting Home Optimizer..."

exec uvicorn web.server:app \
    --app-dir src \
    --host 0.0.0.0 \
    --port 8099 \
    --workers 1 \
    --reload
