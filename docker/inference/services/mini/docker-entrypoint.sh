#!/usr/bin/env bash
set -euo pipefail

export VITOOM_SERVICE_GROUP=mini
export VITOOM_SUPERVISOR_CONF=/app/docker/inference/services/mini/supervisord.conf

exec /app/docker/inference/common/docker-entrypoint.sh "$@"
