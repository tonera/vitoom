#!/usr/bin/env bash
set -euo pipefail

cd /app

: "${VITOOM_BACKEND_URL:=http://host.docker.internal:8888}"
: "${VITOOM_WS_URL:=}"
: "${VITOOM_MODELS_DIR:=resources/models}"
: "${VITOOM_WEIGHTS_DIR:=resources/weights}"
: "${VITOOM_LORAS_DIR:=resources/loras}"
: "${VITOOM_OUTPUTS_DIR:=resources/outputs}"
: "${VITOOM_PIPELINE_CACHE_TTL_SECONDS:=0}"
: "${VITOOM_CIVITAI_TOKEN:=}"
: "${VITOOM_OVERWRITE_CONFIG:=0}"
: "${VITOOM_INFERENCE_UPLOAD_AUTH_SECRET:=}"
: "${VITOOM_SUPERVISOR_URL:=}"
: "${VITOOM_SERVICE_GROUP:=visual}"
: "${VITOOM_SUPERVISOR_CONF:=/app/docker/inference/services/${VITOOM_SERVICE_GROUP}/supervisord.conf}"

if [ -z "$VITOOM_WS_URL" ]; then
  _ws_base="$VITOOM_BACKEND_URL"
  case "$_ws_base" in
    http://*) VITOOM_WS_URL="ws://${_ws_base#http://}" ;;
    https://*) VITOOM_WS_URL="wss://${_ws_base#https://}" ;;
    *) VITOOM_WS_URL="$_ws_base" ;;
  esac
fi

mkdir -p /app/inference/config /app/logs/supervisor /app/resources

# Compile/JIT cache mount points (compose bind-mount targets); ensure dirs exist at startup so users need not mkdir manually.
case "${VITOOM_SERVICE_GROUP:-}" in
  visual)
    mkdir -p /root/.triton/cache
    ;;
  text)
    mkdir -p /root/.cache/flashinfer /root/.cache/vllm
    ;;
  audio)
    mkdir -p /root/.triton/cache /root/.cache/vllm /root/.cache/torch
    ;;
esac

RUNTIME_CONFIG_FILES=" inference.yaml image.yaml video.yaml text.yaml translate.yaml qwen_asr.yaml qwen_tts.yaml mini.yaml download.yaml "

seed_config_defaults() {
  local src="/app/inference/config.defaults"
  local dst="/app/inference/config"
  if [ ! -d "$src" ]; then
    return 0
  fi
  local path name
  for path in "$src"/*; do
    [ -e "$path" ] || continue
    name=$(basename "$path")
    case "$RUNTIME_CONFIG_FILES" in
      *" ${name} "*) continue ;;
    esac
    if [ -d "$path" ]; then
      cp -Rn "$path" "$dst/"
    elif [ ! -e "$dst/$name" ]; then
      cp -n "$path" "$dst/$name"
    fi
  done
}

sync_inference_env_urls() {
  python - <<'PY'
import os
import re
from pathlib import Path

path = Path("/app/inference/config/inference.yaml")
backend = os.environ.get("VITOOM_BACKEND_URL", "").strip()
ws = os.environ.get("VITOOM_WS_URL", "").strip()
if not path.is_file() or not backend or not ws:
    raise SystemExit(0)

text = path.read_text(encoding="utf-8")


def replace_key(content: str, key: str, value: str) -> str:
    line = f'{key}: "{value}"'
    pattern = rf"^{re.escape(key)}:.*$"
    if re.search(pattern, content, flags=re.MULTILINE):
        return re.sub(pattern, line, content, count=1, flags=re.MULTILINE)
    return content.rstrip() + "\n" + line + "\n"


updated = replace_key(text, "api_base_url", backend)
updated = replace_key(updated, "ws_url", ws)
if updated != text:
    path.write_text(updated, encoding="utf-8")
PY
}

for resource_subdir in models weights loras outputs; do
  resource_path="/app/resources/${resource_subdir}"
  if [ -e "$resource_path" ] && [ ! -d "$resource_path" ]; then
    echo "ERROR: ${resource_path} exists but is not a directory." >&2
    echo "Check VITOOM_RESOURCES_DIR on the host. It must point to the resources root containing models/, weights/, loras/, outputs/." >&2
    exit 1
  fi
  mkdir -p "$resource_path"
done

write_config() {
  local path="$1"
  local content="$2"

  if [ -f "$path" ] && [ "$VITOOM_OVERWRITE_CONFIG" != "1" ]; then
    return 0
  fi

  printf '%s\n' "$content" > "$path"
}

# On first text.yaml generation: default 4B text model needs ~14GiB; derive vLLM utilization from total GPU VRAM.
compute_text_gpu_memory_utilization() {
  python - <<'PY'
import subprocess

limit_gib = 14.0
fallback = 0.92
max_util = 0.95


def format_util(total_gib: float) -> str:
    if total_gib <= 0:
        return str(fallback)
    util = min(max_util, limit_gib / total_gib)
    text = f"{util:.4f}".rstrip("0").rstrip(".")
    return text if text else str(fallback)


try:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    first = proc.stdout.strip().splitlines()[0].strip()
    if first.isdigit():
        print(format_util(int(first) / 1024.0))
        raise SystemExit(0)
except Exception:
    pass

try:
    import torch

    if torch.cuda.is_available():
        total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(format_util(total_gib))
        raise SystemExit(0)
except Exception:
    pass

print(fallback)
PY
}

if [ -z "$VITOOM_INFERENCE_UPLOAD_AUTH_SECRET" ] && [ -f /app/config/default.yaml ]; then
  VITOOM_INFERENCE_UPLOAD_AUTH_SECRET="$(
    python - <<'PY'
from pathlib import Path
import re

text = Path("/app/config/default.yaml").read_text(encoding="utf-8")
in_inference = False
for line in text.splitlines():
    if re.match(r"^inference:\s*(?:#.*)?$", line):
        in_inference = True
        continue
    if in_inference and line and not line.startswith((" ", "\t", "#")):
        break
    if in_inference:
        match = re.match(r'^\s+upload_auth_secret:\s*["\']?([^"\'#]*)["\']?', line)
        if match:
            print(match.group(1).strip())
            break
PY
  )"
fi

write_config "/app/inference/config/inference.yaml" "$(cat <<EOF
# Global inference configuration.
# api_base_url/ws_url come from VITOOM_BACKEND_URL / VITOOM_WS_URL in .env.
# storage.server.auth.secret comes from VITOOM_INFERENCE_UPLOAD_AUTH_SECRET.
models_dir: "${VITOOM_MODELS_DIR}"
weights_dir: "${VITOOM_WEIGHTS_DIR}"
loras_dir: "${VITOOM_LORAS_DIR}"
outputs_dir: "${VITOOM_OUTPUTS_DIR}"
pipeline_cache_ttl_seconds: ${VITOOM_PIPELINE_CACHE_TTL_SECONDS}

storage:
  default: "server"
  server:
    upload_path: "/api/inference/upload"
    timeout_seconds: 60
    auth:
      secret: "${VITOOM_INFERENCE_UPLOAD_AUTH_SECRET}"
  s3:
    endpoint: null
    region: null
    bucket: ""
    access_key_id: ""
    secret_access_key: ""
    public_base_url: null
  oss:
    endpoint: ""
    bucket: ""
    access_key_id: ""
    access_key_secret: ""
    public_base_url: null

api_base_url: "${VITOOM_BACKEND_URL}"
ws_url: "${VITOOM_WS_URL}"
# supervisor_url is set per service in {service_id}.yaml (overrides this file) to avoid port clashes when multiple inference containers share a directory.

transport:
  ingresses:
    - type: "ws"
  egresses:
    - type: "ws"
EOF
)"

case "$VITOOM_SERVICE_GROUP" in
  visual)
    write_config "/app/inference/config/image.yaml" "$(cat <<EOF
# Generated from inference/config/ex_image.yaml template.
service_id: "image"
name: "Image generation service"
type: "diffusers"
service_type: "image"
supervisor_url: "${VITOOM_SUPERVISOR_URL}"
EOF
)"
    write_config "/app/inference/config/video.yaml" "$(cat <<EOF
# Generated from inference/config/ex_video.yaml template.
service_id: "video"
name: "Video generation service"
type: "diffusers"
service_type: "video"
supervisor_url: "${VITOOM_SUPERVISOR_URL}"
EOF
)"
    ;;
  text)
    TEXT_GPU_MEMORY_UTILIZATION="$(compute_text_gpu_memory_utilization)"
    write_config "/app/inference/config/text.yaml" "$(cat <<EOF
# Generated from inference/config/ex_text.yaml template.
service_id: "text"
name: "Text Inference Service"
type: "vllm"
service_type: "text"
supervisor_url: "${VITOOM_SUPERVISOR_URL}"

config:
  runtime:
    backend: "vllm"
    max_model_len: 65536
    max_tokens: 2048
    trust_remote_code: true
    enable_thinking: false
    vllm:
      tensor_parallel_size: 1
      # First-time generation: auto-computed from default 4B ~14GiB VRAM cap and total GPU VRAM; adjust manually for larger models.
      gpu_memory_utilization: ${TEXT_GPU_MEMORY_UTILIZATION}
      engine_kwargs:
        safetensors_load_strategy: "eager"
        load_format: "fastsafetensors"
        max_num_batched_tokens: 4096
        limit_mm_per_prompt:
          image: 8
          video: 1
          audio: 0
        enforce_eager: false
    transformers:
      dtype: "auto"
      device_map: "cuda:0"
      allow_cpu_offload: false
      disable_mmap: true
      pin_memory: auto
      model_kwargs:
        attn_implementation: "sdpa"
        low_cpu_mem_usage: true
EOF
)"
    write_config "/app/inference/config/translate.yaml" "$(cat <<EOF
# Generated from inference/config/ex_translate.yaml template.
service_id: "translate"
name: "Translate Inference Service (TranslateGemma)"
type: "transformers"
service_type: "translate"
supervisor_url: "${VITOOM_SUPERVISOR_URL}"

host: "127.0.0.1"
port: 8007

config:
  runtime:
    backend: "transformers"
    trust_remote_code: true
    temperature: 0.0
    top_p: 1.0
    max_new_tokens: 768
    dtype: "auto"
    device_map: "cuda:0"
    transformers:
      model_kwargs:
        attn_implementation: "sdpa"
        low_cpu_mem_usage: true
EOF
)"
    ;;
  audio)
    write_config "/app/inference/config/qwen_asr.yaml" "$(cat <<EOF
# Generated from inference/config/ex_qwen_asr.yaml template.
service_id: "qwen_asr"
name: "Qwen ASR Inference Service"
type: "audio"
service_type: "audio"
supervisor_url: "${VITOOM_SUPERVISOR_URL}"

host: "127.0.0.1"
port: 8011

config:
  capabilities: ["asr"]
  fixed_model: "Qwen3-ASR-0.6B"
  fixed_family: "Qwen-asr"
  supported_models:
    - "Qwen3-ASR-0.6B"
    - "Qwen3-ASR-1.7B"
  runtime:
    backend: "vllm"
    forced_aligner: "Qwen3-ForcedAligner-0.6B"
    vllm:
      gpu_memory_utilization: 0.3
      max_model_len: 4096
      enforce_eager: false
      streaming:
        unfixed_chunk_num: 2
        unfixed_token_num: 8
        chunk_size_sec: 1.5
EOF
)"
    write_config "/app/inference/config/qwen_tts.yaml" "$(cat <<EOF
# Generated from inference/config/ex_qwen_tts.yaml template.
service_id: "qwen_tts"
name: "Qwen TTS Inference Service"
type: "audio"
service_type: "audio"
supervisor_url: "${VITOOM_SUPERVISOR_URL}"

host: "127.0.0.1"
port: 8010

config:
  capabilities: ["tts"]
  fixed_family: "Qwen-tts"
  supported_models:
    - "Qwen3-TTS-12Hz-0.6B-Base"
    - "Qwen3-TTS-12Hz-1.7B-CustomVoice"
    - "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    - "Qwen3-TTS-12Hz-0.6B-CustomVoice"
  runtime:
    backend: "transformers"
    nano_vllm:
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.20
      max_num_batched_tokens: 1024
      max_num_seqs: 4
      max_model_len: 1024
      kvcache_block_size: 256
      enforce_eager: false
EOF
)"
    ;;
  mini)
    write_config "/app/inference/config/mini.yaml" "$(cat <<EOF
# Generated from inference/config/ex_mini.yaml template.
service_id: "mini"
name: "Mini Inference Service (OCR & small models)"
type: "transformers"
service_type: "mini"
supervisor_url: "${VITOOM_SUPERVISOR_URL}"

host: "127.0.0.1"
port: 8006

config:
  runtime:
    backend: "transformers"
    trust_remote_code: true
    temperature: 0.0
    top_p: 1.0
    max_new_tokens: 4096
    dtype: "auto"
    transformers:
      model_kwargs:
        attn_implementation: "sdpa"
        low_cpu_mem_usage: true
    vllm:
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.35
      max_model_len: 8192
      max_images_per_prompt: 4
      engine_kwargs:
        enforce_eager: false
EOF
)"
    ;;
  download)
    write_config "/app/inference/config/download.yaml" "$(cat <<EOF
# Generated from inference/config/ex_download.yaml template.
service_id: "download"
name: "Download service"
type: "download"
service_type: "download"
supervisor_url: "${VITOOM_SUPERVISOR_URL}"

config:
  capabilities:
    - huggingface
    - modelscope
    - civitai
  civitai_token: "${VITOOM_CIVITAI_TOKEN}"
EOF
)"
    ;;
  *)
    ;;
esac

seed_config_defaults
sync_inference_env_urls

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

# Mirror supervisor program logs to container stdout so `docker logs` still works.
(
  while true; do
    if compgen -G "/app/logs/supervisor/*.log" > /dev/null; then
      tail -F /app/logs/supervisor/*.log
    fi
    sleep 1
  done
) &

exec /usr/bin/supervisord -c "$VITOOM_SUPERVISOR_CONF"
