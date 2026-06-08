# Vitoom Docker Image Build Guide

This guide is for developers and release maintainers only. It explains how to build Docker images and export deliverables. End-user deployment and runtime instructions are in [`docker-usage-en.md`](docker-usage-en.md).

Run all commands from the repository root unless noted otherwise.

## 1. Build prerequisites

You need:

- Docker
- Docker Compose
- Python 3.11+ (to download build artifacts; Windows / macOS / Linux)

If you are in mainland China or on an unstable network, configure build mirrors in `.env`:

```env
APT_MIRROR=https://mirrors.aliyun.com/debian
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
VITOOM_WHEEL_BASE_URL=http://192.168.31.17
VITOOM_TARGET_ARCH=x86_64
```

### 1.1 Download build artifacts

Backend and inference images no longer download large files during `docker build`. You must run this first:

```bash
python scripts/setup_vitoom.py
```

The script will:

- Choose region (mainland China writes Aliyun apt/pip mirrors into `.env`)
- Choose mode **[1] Image build preparation** and auto-detect CPU architecture (`x86_64` / `aarch64`)
- Interactively select components to prepare (Backend and inference can be prepared on different machines as needed)
- For the E5 model (`Xenova/multilingual-e5-small`), probe Hugging Face / ModelScope by region and network and prefer the faster source
- Download files into the expected directories; skip files that already exist
- Interactively choose UI language (中文 / 日本語 / English) and network region (mainland China / other)

Resulting directory layout:

```text
docker/backend/artifacts/x86_64/
  elasticsearch-8.15.5-linux-x86_64.tar.gz
  pandoc-3.6.4-1-amd64.deb
  multilingual-e5-small-onnx/onnx/model.onnx
  multilingual-e5-small-onnx/tokenizer.json

docker/inference/wheels/x86_64/
  flash_attn-*.whl
  spas_sage_attn-*.whl
  nunchaku-*.whl
  turbodiffusion-*.whl
```

File list and versions are in `docker/build-artifacts.manifest.json`. Missing files cause `docker compose build` to fail with a prompt to run the command above.

When the E5 model must be downloaded, the script checks for `hf` / `huggingface_hub` and `modelscope`; if missing, it installs them via `pip install` (honors `PIP_INDEX_URL` in `.env`).

## 2. Build Backend image

The Backend image includes the FastAPI server, frontend static assets, SQLite init and migrations, built-in Elasticsearch, LibreOffice, Pandoc, and `multilingual-e5-small-onnx`.

x86_64 image:

```bash
VITOOM_TARGET_ARCH=x86_64 docker compose build backend
```

NVIDIA Spark / aarch64 image:

```bash
VITOOM_TARGET_ARCH=aarch64 docker compose build backend
```

`VITOOM_TARGET_ARCH` must match the Docker build platform (building aarch64 on an x86_64 host requires `docker buildx build --platform linux/arm64` with `VITOOM_TARGET_ARCH=aarch64`).

Verify the image:

```bash
docker images vitoom-backend
```

Export x86_64 image (tags match `scripts/vitoom_setup/constants.py` for direct use in `.env` from the setup script):

```bash
mkdir -p images/x86_64
docker tag vitoom-backend:latest tonera/vitoom-backend:latest-x86_64
docker save -o images/x86_64/vitoom-backend-latest-x86_64.tar tonera/vitoom-backend:latest-x86_64
```

Export NVIDIA Spark / aarch64 image:

```bash
mkdir -p images/aarch64
docker tag vitoom-backend:latest tonera/vitoom-backend:latest-aarch64
docker save -o images/aarch64/vitoom-backend-latest-aarch64.tar tonera/vitoom-backend:latest-aarch64
```

## 3. Build inference base images

Inference images are built with `docker-compose.inference.yml`. That file is for developers and release maintainers only, not for end-user deployment.

Build the shared Python runtime:

```bash
docker compose -f docker-compose.inference.yml --profile build-base build python-runtime
```

Verify the base image locally:

```bash
docker images vitoom-python-runtime
```

If you skip this step, later `torch-runtime` builds use `FROM vitoom-python-runtime:py3.11` and Docker Hub may return `pull access denied` for that name.

Build the PyTorch runtime for your target CUDA/Torch combo. `cu128` is no longer shipped; use `cu130` with two maintained base lines. Base images contain runtime dependencies only—no on-the-fly CUDA extension compile toolchain:

- Default baseline: `cu130 + torch 2.11.0` for Visual and other mainline inference images.
- Compatibility baseline: `cu130 + torch 2.9.1` for services or experimental images that still need the torch 2.9.1 stack.

Build `cu130 + torch 2.11.0`:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
TORCH_EXTRA_INDEX_URL=https://pypi.org/simple \
TORCH_VERSION=2.11.0 \
TORCHVISION_VERSION=0.26.0 \
TORCHAUDIO_VERSION=2.11.0 \
VITOOM_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.11-py3.11-cu130 \
docker compose -f docker-compose.inference.yml --profile build-base build torch-runtime
```

Build `cu130 + torch 2.9.1`:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
TORCH_EXTRA_INDEX_URL=https://pypi.org/simple \
TORCH_VERSION=2.9.1 \
TORCHVISION_VERSION=0.24.1 \
TORCHAUDIO_VERSION=2.9.1 \
VITOOM_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.9.1-py3.11-cu130 \
docker compose -f docker-compose.inference.yml --profile build-base build torch-runtime
```

## 4. Prebuilt wheels

Release images do not compile `flash_attn`, `spas_sage_attn`, `nunchaku`, or `turbodiffusion` at build time. Download wheels via `setup_vitoom.py` into `docker/inference/wheels/<arch>/` before building.

Supported scope:

- **x86_64**: general users; consumer NVIDIA GPUs RTX 30 / 40 / 50 series.
- **aarch64**: NVIDIA Spark (DGX Spark / RTX Spark), GB10 / Blackwell; build target covers `sm120`; image tag suffix `nvidia-spark`.
- Visual / Text / Mini and other mainline services: `torch 2.11 + cu130 + Python 3.11`.
- Audio: `torch 2.9.1 + cu130 + Python 3.11`; requires a separately built `flash_attn` wheel (filename contains `torch2.9`); cannot reuse torch 2.11 wheels.

Build fails if prebuilt wheels are missing or if multiple wheels match.

Prebuilt wheels must have **glibc version ≤ the base image**. `python-runtime` is based on Debian Trixie (glibc 2.38+), aligned with aarch64 wheels built on NVIDIA Spark hosts; x86_64 wheels built on older-glibc systems remain valid.

If you still have old Bookworm base images locally, rebuild `python-runtime` and `torch-runtime` before service images.

## 5. Build Visual inference image

Prepare wheels first:

```bash
python scripts/setup_vitoom.py
```

x86_64 image:

```bash
VITOOM_TARGET_ARCH=x86_64 \
VITOOM_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.11-py3.11-cu130 \
VITOOM_VISUAL_IMAGE=vitoom-inference-visual:experimental-cu130-torch2.11-x86_64 \
docker compose -f docker-compose.inference.yml build visual
```

NVIDIA Spark / aarch64 image:

```bash
python scripts/setup_vitoom.py

VITOOM_TARGET_ARCH=aarch64 \
VITOOM_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.11-py3.11-cu130 \
VITOOM_VISUAL_IMAGE=vitoom-inference-visual:experimental-cu130-torch2.11-aarch64-nvidia-spark \
docker compose -f docker-compose.inference.yml build visual
```

Verify the image:

```bash
docker images vitoom-inference-visual
```


## 6. Build Text inference image

The Text image does not require extra prebuilt wheels.

x86_64 image:

```bash
VITOOM_TARGET_ARCH=x86_64 \
VITOOM_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.11-py3.11-cu130 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
VLLM_VERSION=0.21.0 \
CUDA_KEYRING_VERSION=1.1-1 \
TEXT_CUDA_TOOLKIT_PACKAGE_SUFFIX=13-1 \
VITOOM_TEXT_IMAGE=vitoom-inference-text:experimental-cu130-torch2.11-x86_64 \
docker compose -f docker-compose.inference.yml build text
```

NVIDIA Spark / aarch64 image:

```bash
VITOOM_TARGET_ARCH=aarch64 \
VITOOM_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.11-py3.11-cu130 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
VLLM_VERSION=0.21.0 \
CUDA_KEYRING_VERSION=1.1-1 \
TEXT_CUDA_TOOLKIT_PACKAGE_SUFFIX=13-2 \
VITOOM_TEXT_IMAGE=vitoom-inference-text:experimental-cu130-torch2.11-aarch64-nvidia-spark \
docker compose -f docker-compose.inference.yml build text
```

Verify the image:

```bash
docker images vitoom-inference-text
```

## 7. Build Audio inference image

Prepare wheels first:

```bash
python scripts/setup_vitoom.py
```

x86_64 image:

```bash
VITOOM_TARGET_ARCH=x86_64 \
VITOOM_AUDIO_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.9.1-py3.11-cu130 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
AUDIO_FLASHINFER_VERSION=0.5.3 \
VITOOM_AUDIO_IMAGE=vitoom-inference-audio:experimental-cu130-torch2.9.1-x86_64 \
docker compose -f docker-compose.inference.yml build audio
```

NVIDIA Spark / aarch64 image:

```bash
python scripts/setup_vitoom.py

VITOOM_TARGET_ARCH=aarch64 \
VITOOM_AUDIO_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.9.1-py3.11-cu130 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
AUDIO_FLASHINFER_VERSION=0.5.3 \
VITOOM_AUDIO_IMAGE=vitoom-inference-audio:experimental-cu130-torch2.9.1-aarch64-nvidia-spark \
docker compose -f docker-compose.inference.yml build audio
```

Verify the image:

```bash
docker images vitoom-inference-audio
```

## 8. Build Mini inference image

The Mini image hosts OCR and other small-model services on the `torch 2.11 + cu130` base.

Prepare wheels first:

```bash
python scripts/setup_vitoom.py
```

x86_64 image:

```bash
VITOOM_TARGET_ARCH=x86_64 \
VITOOM_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.11-py3.11-cu130 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
VITOOM_MINI_IMAGE=vitoom-inference-mini:experimental-cu130-torch2.11-x86_64 \
docker compose -f docker-compose.inference.yml build mini
```

NVIDIA Spark / aarch64 image:

```bash
python scripts/setup_vitoom.py

VITOOM_TARGET_ARCH=aarch64 \
VITOOM_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.11-py3.11-cu130 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
VITOOM_MINI_IMAGE=vitoom-inference-mini:experimental-cu130-torch2.11-aarch64-nvidia-spark \
docker compose -f docker-compose.inference.yml build mini
```

Verify the image:

```bash
docker images vitoom-inference-mini
```

## 9. Build Download inference image

The Download image handles model and resource download tasks. It uses the shared Python runtime and does not depend on the Torch runtime.

Build:

```bash
VITOOM_PYTHON_BASE_IMAGE=vitoom-python-runtime:py3.11 \
docker compose -f docker-compose.inference.yml build download
```

Verify the image:

```bash
docker images vitoom-inference-download
```

## 10. Export inference images

Export x86_64 images (tags match `scripts/vitoom_setup/constants.py`):

```bash
mkdir -p images/x86_64
docker tag vitoom-inference-visual:experimental-cu130-torch2.11-x86_64 tonera/vitoom-inference-visual:experimental-cu130-torch2.11-x86_64
docker tag vitoom-inference-text:experimental-cu130-torch2.11-x86_64 tonera/vitoom-inference-text:experimental-cu130-torch2.11-x86_64
docker tag vitoom-inference-audio:experimental-cu130-torch2.9.1-x86_64 tonera/vitoom-inference-audio:experimental-cu130-torch2.9.1-x86_64
docker tag vitoom-inference-mini:experimental-cu130-torch2.11-x86_64 tonera/vitoom-inference-mini:experimental-cu130-torch2.11-x86_64
docker tag vitoom-inference-download:experimental tonera/vitoom-inference-download:experimental-x86_64
docker save -o images/x86_64/vitoom-inference-visual-cu130-torch2.11-x86_64.tar tonera/vitoom-inference-visual:experimental-cu130-torch2.11-x86_64
docker save -o images/x86_64/vitoom-inference-text-cu130-torch2.11-x86_64.tar tonera/vitoom-inference-text:experimental-cu130-torch2.11-x86_64
docker save -o images/x86_64/vitoom-inference-audio-cu130-torch2.9.1-x86_64.tar tonera/vitoom-inference-audio:experimental-cu130-torch2.9.1-x86_64
docker save -o images/x86_64/vitoom-inference-mini-cu130-torch2.11-x86_64.tar tonera/vitoom-inference-mini:experimental-cu130-torch2.11-x86_64
docker save -o images/x86_64/vitoom-inference-download-experimental-x86_64.tar tonera/vitoom-inference-download:experimental-x86_64
```

Export NVIDIA Spark / aarch64 images:

```bash
mkdir -p images/aarch64
docker tag vitoom-inference-visual:experimental-cu130-torch2.11-aarch64-nvidia-spark tonera/vitoom-inference-visual:experimental-cu130-torch2.11-aarch64-nvidia-spark
docker tag vitoom-inference-text:experimental-cu130-torch2.11-aarch64-nvidia-spark tonera/vitoom-inference-text:experimental-cu130-torch2.11-aarch64-nvidia-spark
docker tag vitoom-inference-audio:experimental-cu130-torch2.9.1-aarch64-nvidia-spark tonera/vitoom-inference-audio:experimental-cu130-torch2.9.1-aarch64-nvidia-spark
docker tag vitoom-inference-mini:experimental-cu130-torch2.11-aarch64-nvidia-spark tonera/vitoom-inference-mini:experimental-cu130-torch2.11-aarch64-nvidia-spark
docker tag vitoom-inference-download:experimental tonera/vitoom-inference-download:experimental-aarch64-nvidia-spark
docker save -o images/aarch64/vitoom-inference-visual-cu130-torch2.11-aarch64-nvidia-spark.tar tonera/vitoom-inference-visual:experimental-cu130-torch2.11-aarch64-nvidia-spark
docker save -o images/aarch64/vitoom-inference-text-cu130-torch2.11-aarch64-nvidia-spark.tar tonera/vitoom-inference-text:experimental-cu130-torch2.11-aarch64-nvidia-spark
docker save -o images/aarch64/vitoom-inference-audio-cu130-torch2.9.1-aarch64-nvidia-spark.tar tonera/vitoom-inference-audio:experimental-cu130-torch2.9.1-aarch64-nvidia-spark
docker save -o images/aarch64/vitoom-inference-mini-cu130-torch2.11-aarch64-nvidia-spark.tar tonera/vitoom-inference-mini:experimental-cu130-torch2.11-aarch64-nvidia-spark
docker save -o images/aarch64/vitoom-inference-download-experimental-aarch64.tar tonera/vitoom-inference-download:experimental-aarch64-nvidia-spark
```
