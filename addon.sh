#!/usr/bin/with-contenv bashio

set -euo pipefail

bashio::log.info "Starting Home Optimizer..."

exec uvicorn web.server:app \
    --app-dir src \
    --host 0.0.0.0 \
    --port 8099 \
    --workers 1
