#!/usr/bin/env sh
set -eu

copy_if_missing() {
  src="$1"
  dst="$2"
  if [ ! -e "$dst" ] && [ -e "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    cp -R "$src" "$dst"
  fi
}

relax_host_mount_permissions() {
  chmod -R a+rwX \
    /app/resources/data \
    /app/resources/outputs \
    /app/logs \
    2>/dev/null || true
  chmod a+rw /app/resources/data/vitoom.db /app/resources/data/vitoom.db-* 2>/dev/null || true
}

mkdir -p \
  /app/config \
  /app/inference/config \
  /app/resources/data \
  /app/resources/outputs \
  /app/resources/models \
  /app/resources/elasticsearch/data \
  /app/logs \
  /app/logs/elasticsearch

copy_if_missing /app/config.defaults/default.yaml /app/config/default.yaml
copy_if_missing /app/config.defaults/agent_tools.yaml /app/config/agent_tools.yaml
copy_if_missing /app/config.defaults/model_catalog_meta.yaml /app/config/model_catalog_meta.yaml
copy_if_missing /app/config.defaults/tts_speakers.json /app/config/tts_speakers.json
copy_if_missing /app/config.defaults/agents /app/config/agents
copy_if_missing /app/resources.defaults/models/multilingual-e5-small-onnx /app/resources/models/multilingual-e5-small-onnx

relax_host_mount_permissions

start_elasticsearch() {
  mkdir -p /app/resources/elasticsearch/data /app/logs/elasticsearch
  chown -R elasticsearch:elasticsearch /app/resources/elasticsearch /app/logs/elasticsearch

  es_pid=""
  if ! curl -fsS http://127.0.0.1:9200/ >/dev/null 2>&1; then
    ES_JAVA_OPTS="${ELASTICSEARCH_JAVA_OPTS:--Xms512m -Xmx512m}" \
      gosu elasticsearch /opt/elasticsearch/bin/elasticsearch \
      > /app/logs/elasticsearch/stdout.log \
      2> /app/logs/elasticsearch/stderr.log &
    es_pid=$!
  fi

  attempt=0
  while [ "$attempt" -lt 60 ]; do
    if curl -fsS http://127.0.0.1:9200/ >/dev/null 2>&1; then
      return 0
    fi
    if [ -n "$es_pid" ] && ! kill -0 "$es_pid" 2>/dev/null; then
      echo "Elasticsearch exited; see /app/logs/elasticsearch/stderr.log" >&2
      tail -n 30 /app/logs/elasticsearch/stderr.log >&2 || true
      return 1
    fi
    attempt=$((attempt + 1))
    sleep 1
  done
  echo "Elasticsearch not ready on :9200" >&2
  return 1
}

start_elasticsearch

python - <<'PY'
from backend.database.db import init_db
from backend.database.migrations import migrate

init_db()
migrate()
PY

relax_host_mount_permissions

python - <<'PY'
from backend.services.agent import settings as agent_settings
from backend.services.agent.tools.builtin.knowledge_base_core.es_client import KnowledgeBaseEsClient
from backend.services.agent.tools.builtin.knowledge_base_core.ingest import ensure_indices

if not agent_settings.is_knowledge_base_enabled():
    raise SystemExit(0)

client = KnowledgeBaseEsClient(
    url=agent_settings.get_knowledge_base_es_url(),
    username=agent_settings.get_knowledge_base_es_username(),
    password=agent_settings.get_knowledge_base_es_password(),
    timeout=agent_settings.get_knowledge_base_request_timeout_seconds(),
)
ensure_indices(
    client,
    document_index=agent_settings.get_knowledge_base_document_index(),
    chunk_index=agent_settings.get_knowledge_base_chunk_index(),
    dims=agent_settings.get_knowledge_base_embedding_dims(),
)
PY

exec "$@"
