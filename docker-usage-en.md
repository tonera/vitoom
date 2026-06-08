# Vitoom Docker User Guide

**English** | [中文](docker-usage-cn.md) | [日本語](docker-usage-jp.md)

## 1. Prepare the environment

You need:

- Docker
- Docker Compose

To run inference services, you also need:

- NVIDIA GPU
- **NVIDIA driver with CUDA 13.0 support** (matches `cu130` inference images)
- NVIDIA Container Toolkit

Verify that Docker can access the GPU and the CUDA 13.0 runtime:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

On Windows, use Docker Desktop + WSL2 and ensure file sharing is enabled for the disk where your model directories live.

## 2. Prepare configuration

Generate `.env` using **one** of the following (do not do both):

### Method A: Setup wizard (recommended)

```bash
python scripts/setup_vitoom.py
```

Interactively choose components; auto-detect architecture, write image tags, generate secrets; optionally fetch Docker images at the end.

### Method B: Manual edit

```bash
cp .env.example .env
```

Windows PowerShell: `copy .env.example .env`

Edit `.env` and fill in all required deployment values (missing values cause startup failures or wrong images):

```env
VITOOM_TARGET_ARCH=x86_64
VITOOM_INFERENCE_UPLOAD_AUTH_SECRET=use-a-long-random-string
VITOOM_SERVER_PORT=8888
VITOOM_BACKEND_URL=http://BACKEND_IP:8888
VITOOM_WS_URL=ws://BACKEND_IP:8888
```

Image tags (must match Docker Hub / offline tar; for aarch64 see `scripts/vitoom_setup/constants.py`):

```env
VITOOM_BACKEND_IMAGE=tonera/vitoom-backend:latest-x86_64
VITOOM_VISUAL_IMAGE=tonera/vitoom-inference-visual:experimental-cu130-torch2.11-x86_64
VITOOM_TEXT_IMAGE=tonera/vitoom-inference-text:experimental-cu130-torch2.11-x86_64
VITOOM_AUDIO_IMAGE=tonera/vitoom-inference-audio:experimental-cu130-torch2.9.1-x86_64
VITOOM_MINI_IMAGE=tonera/vitoom-inference-mini:experimental-cu130-torch2.11-x86_64
VITOOM_DOWNLOAD_IMAGE=tonera/vitoom-inference-download:experimental-x86_64
```

On inference nodes, also set the same secret as the Backend, and Supervisor URLs for services deployed on that host (leave empty if not deployed):

```env
VITOOM_VISUAL_SUPERVISOR_URL=http://INFERENCE_IP:9001
VITOOM_TEXT_SUPERVISOR_URL=http://INFERENCE_IP:9002
VITOOM_AUDIO_SUPERVISOR_URL=http://INFERENCE_IP:9003
VITOOM_DOWNLOAD_SUPERVISOR_URL=http://INFERENCE_IP:9004
VITOOM_MINI_SUPERVISOR_URL=http://INFERENCE_IP:9005
```

Use the LAN IP for the Backend address, not `127.0.0.1` or container names. When changing the port, update `VITOOM_SERVER_PORT`, `VITOOM_BACKEND_URL`, and `VITOOM_WS_URL` together.

---

Once `.env` is ready, continue to [§3 Obtaining images](#3-obtaining-images) (skip if Method A already fetched images in the wizard). To batch-download common models, see [§6 Initial model download (optional)](#6-initial-model-download-optional).

## 3. Obtaining images

```bash
python scripts/load_vitoom_images.py
```

Prefer `docker load` from tar files under `images/<VITOOM_TARGET_ARCH>/` in the project directory; if missing, fall back to `docker pull`.

Fetch only selected services:

```bash
python scripts/load_vitoom_images.py --components backend,visual
```

Force reload:

```bash
python scripts/load_vitoom_images.py --force
```

## 4. Start Backend

Start:

```bash
docker compose up -d backend
```

Check status and logs:

```bash
docker compose ps
docker compose logs -f backend
```

Health check:

```bash
curl http://127.0.0.1:8888/api/health
```

Open in browser:

```text
http://127.0.0.1:8888
```

If you changed `VITOOM_SERVER_PORT`, replace the port in the URLs above.

## 5. Start inference services

Start Visual (image and video generation; first run is slow):

```bash
docker compose -f docker-compose.inference.release.yml --profile visual up -d
```

Start Text (large language model service; first run is slow, ~5 minutes on DGX Spark):

```bash
docker compose -f docker-compose.inference.release.yml --profile text up -d
```

Start Audio (audio generation):

```bash
docker compose -f docker-compose.inference.release.yml --profile audio up -d
```

Start Download (model download service):

```bash
docker compose -f docker-compose.inference.release.yml --profile download up -d
```

Start Mini (small-model service):

```bash
docker compose -f docker-compose.inference.release.yml --profile mini up -d
```

Check status:

```bash
docker compose -f docker-compose.inference.release.yml ps
```

View logs:

```bash
docker compose -f docker-compose.inference.release.yml logs -f visual
docker compose -f docker-compose.inference.release.yml logs -f text
docker compose -f docker-compose.inference.release.yml logs -f audio
docker compose -f docker-compose.inference.release.yml logs -f mini
docker compose -f docker-compose.inference.release.yml logs -f download
```

Check in-container supervisor status:

```bash
docker exec -it vitoom-inference-visual supervisorctl -s unix:///tmp/supervisor.sock status
docker exec -it vitoom-inference-text supervisorctl -s unix:///tmp/supervisor.sock status
docker exec -it vitoom-inference-audio supervisorctl -s unix:///tmp/supervisor.sock status
docker exec -it vitoom-inference-mini supervisorctl -s unix:///tmp/supervisor.sock status
```

## 6. Initial model download (optional)

After running `python scripts/setup_vitoom.py` and generating `.env`, you can run the initial model download script for a quicker first experience. Total size is about **100G+**.

Run from the deployment directory (repository root containing `.env`):

```bash
python scripts/download_initial_models.py
```

## 7. Resource directories

Inference services mount by default:

```text
resources/models
resources/weights
resources/loras
resources/outputs
```

If models, weights, LoRAs, or outputs live elsewhere, set in `.env`:

```env
VITOOM_MODELS_HOST_DIR=/data/vitoom/models
VITOOM_WEIGHTS_HOST_DIR=/data/vitoom/weights
VITOOM_LORAS_HOST_DIR=/data/vitoom/loras
VITOOM_OUTPUTS_HOST_DIR=/data/vitoom/outputs
```

On Windows, use forward slashes:

```env
VITOOM_MODELS_HOST_DIR=C:/vitoom/models
VITOOM_WEIGHTS_HOST_DIR=C:/vitoom/weights
VITOOM_LORAS_HOST_DIR=C:/vitoom/loras
VITOOM_OUTPUTS_HOST_DIR=C:/vitoom/outputs
```

## 8. Data directories

Backend data is stored under `data/` in the deployment directory:

```text
data/config             User configuration
data/inference/config   Inference service configuration
data/resources          SQLite database, outputs, knowledge base, built-in ES data
data/logs               Backend and ES logs
data/logs/inference     Inference service logs
data/inference/cache    Inference compile and acceleration caches
```

Do not delete `data/` when upgrading or restarting.


### View Backend logs

Application logs: `data/logs/app.log` — `docker compose exec backend tail -f /app/logs/app.log`. Built-in Elasticsearch logs: `data/logs/elasticsearch/`.


After changing `VITOOM_BACKEND_URL` / `VITOOM_WS_URL` in `.env`, restart inference containers; the entrypoint syncs `api_base_url` / `ws_url` in `data/inference/config/inference.yaml` from `.env`.

To **fully rewrite** config files from entrypoint templates (including `storage`, per-service yaml, etc.), temporarily set:

```env
VITOOM_OVERWRITE_CONFIG=1
```

Then restart the relevant inference service:

```bash
docker compose -f docker-compose.inference.release.yml --profile visual up -d --force-recreate
```

After confirming config is updated, set back in `.env`:

```env
VITOOM_OVERWRITE_CONFIG=0
```

## 9. Distributed deployment

Backend is the control plane. Visual, Text, Audio, Mini, and Download can each run as separate inference nodes on different GPU servers.

On inference machines, `.env` must at least reach the Backend:

```env
VITOOM_BACKEND_URL=http://BACKEND_IP:8888
VITOOM_WS_URL=ws://BACKEND_IP:8888
VITOOM_INFERENCE_UPLOAD_AUTH_SECRET=same secret as Backend
```


## 10. Stop and upgrade

Stop Backend:

```bash
docker compose down
```

Stop inference services:

```bash
docker compose -f docker-compose.inference.release.yml --profile visual down
docker compose -f docker-compose.inference.release.yml --profile text down
docker compose -f docker-compose.inference.release.yml --profile audio down
docker compose -f docker-compose.inference.release.yml --profile mini down
```

Upgrade Backend:

```bash
python scripts/load_vitoom_images.py --components backend --force
docker compose up -d backend
```

Upgrade inference services:

```bash
python scripts/load_vitoom_images.py --components visual --force
docker compose -f docker-compose.inference.release.yml --profile visual up -d --force-recreate
```

Back up before upgrading:

```text
data/
resources/
```

## 11. Troubleshooting

Is Backend healthy:

```bash
curl http://127.0.0.1:8888/api/health
docker compose logs --tail=200 backend
```

Are inference services running:

```bash
docker compose -f docker-compose.inference.release.yml ps
docker compose -f docker-compose.inference.release.yml logs --tail=200 visual
```

Check whether inference config still has old addresses:

```bash
cat data/inference/config/inference.yaml
```

If addresses are stale, set `VITOOM_OVERWRITE_CONFIG=1` and restart inference containers.

Confirm image tags match `.env`:

```bash
docker images | grep vitoom-inference
```

## 12. Tips

Paths below assume Docker deployment: Backend config in `data/config/` (first start copies defaults such as `default.yaml`, `tts_speakers.json` from the image; you can add `app.yaml` to override). Inference config in `data/inference/config/`. When running Backend directly on the host, use `config/` at the project root.

Restart the **corresponding service** after editing YAML; run **`docker compose up -d`** to recreate containers after `.env` changes. If global/per-service inference config was written by the entrypoint but changes do not apply, temporarily set `VITOOM_OVERWRITE_CONFIG=1` and restart inference containers (see §8).

### 12.1 Tune text LLM VRAM (`gpu_memory_utilization`)

This setting controls the **fraction of GPU memory** reserved by the vLLM text service. Range `(0, 1]`; higher values use more VRAM. Only applies to text services with `config.runtime.backend: vllm`.

**Docker (edit persistent file)**

Edit text service config (created after first Text start):

```text
data/inference/config/text.yaml
```

Adjust under `config.runtime.vllm`, for example:

```yaml
config:
  runtime:
    vllm:
      gpu_memory_utilization: 0.75
```

- Tight VRAM: **lower** (e.g. `0.5`–`0.7`).
- Larger context or weights: may need to **raise** it and review `max_model_len`.
- On first `text.yaml` generation, entrypoint auto-computes a ratio from ~14GiB / total GPU memory; **it does not auto-update later** — change manually when swapping GPU or model.

Restart Text after edits:

```bash
docker compose -f docker-compose.inference.release.yml --profile text restart
```

**Web admin**: log in as admin → Inference services → select text service (e.g. `text`) → Service config, edit `config.runtime.vllm.gpu_memory_utilization` (restart per UI prompt after save).

**Host development**: edit the same path in `inference/config/ex_text.yaml` or local `inference/config/text.yaml`, then restart the text inference process.

### 12.2 Switch storage to S3

Where Backend tasks/uploads are stored is determined by **`storage.default`** (`server` | `s3` | `oss`).

**1) Backend**

Add to `data/config/app.yaml` (create if missing), for example:

```yaml
storage:
  default: s3
  s3:
    endpoint: "https://s3.amazonaws.com"   # or MinIO / S3-compatible endpoint
    region: "ap-southeast-1"
    bucket: "your-bucket"
    access_key_id: "YOUR_ACCESS_KEY"
    secret_access_key: "YOUR_SECRET_KEY"
    public_base_url: "https://your-bucket.s3.ap-southeast-1.amazonaws.com"
```

`public_base_url` is used for publicly accessible file URLs and must match how the bucket is exposed.

Restart Backend:

```bash
docker compose up -d backend
```

**2) Inference**

Also edit the `storage` section in `data/inference/config/inference.yaml` (`default: s3` and `storage.s3` credentials aligned with Backend or per-bucket policy), then restart the relevant inference containers.


### 12.3 Change the default text LLM


**Change default chat model**

In `data/config/app.yaml` (Text inference running and weights available under `resources/models`):

```yaml
agents:
  default_model: "your-model-name"
```

Restart Backend. New sessions/tasks pick it up; existing sessions bound to an old `load_name` need a model switch in the UI or a new session.

**Pin text inference to a single model**

In `text.yaml`, uncomment and set:

```yaml
config:
  fixed_model: "your-model-name"
```

See `inference/config/ex_gemma_text.yaml` and `inference/config/ex_text.yaml` for examples.

Save and restart the Text container; also re-evaluate `gpu_memory_utilization` per §12.1.

### 12.4 Change the default video generation model


In `data/config/app.yaml`:

```yaml
agents:
  tools:
    video_generator:
      default_model_name: "TurboWan2.1-T2V-1.3B-480P"
```

The name must match a registered video **model name** in the system, with Visual/Video inference deployed and weights in place (default: `resources/models`).

### 12.5 Supported model families

Video: Wan series, TurboWan series  
Audio: Qwen-tts, Qwen-asr, VoxCPM  
Image: SDXL, Qwen-Image, Z-Image, Flux, Flux.2, and other mainstream image models  
Language: Qwen series

### 12.6 Enable live web search

Apply for an API key at https://www.tavily.com/ (free within quota), then set `TAVILY_API_KEY` in `.env`.

### 12.7 Cache models to speed up inference

Add or edit `data/inference/config/{image,video,text,qwen_asr,qwen_tts}.yaml` and set `pipeline_cache_ttl_seconds` to a value greater than 0 so the corresponding inference service keeps models cached, greatly speeding up subsequent runs. **Note:** cached models hold VRAM until the TTL expires.

You can also change this value in `data/inference/config/inference.yaml` to apply to all inference services.

```yaml
pipeline_cache_ttl_seconds: 1800
```
