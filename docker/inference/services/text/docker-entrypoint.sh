#!/usr/bin/env bash
set -euo pipefail

export VITOOM_SERVICE_GROUP=text
export VITOOM_SUPERVISOR_CONF=/app/docker/inference/services/text/supervisord.conf

exec /app/docker/inference/common/docker-entrypoint.sh "$@"
