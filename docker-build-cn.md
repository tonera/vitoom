# Vitoom Docker 镜像构建指南

本文只面向开发者和交付包维护者，说明如何构建 Docker 镜像和导出交付物。最终用户部署和运行请看 `docker-usage.md`。

所有命令默认在项目根目录执行。

## 1. 构建准备

需要安装：

- Docker
- Docker Compose
- Python 3.11+（用于下载构建 artifacts，Windows / macOS / Linux 均可）

如果机器在中国大陆或网络不稳定，建议在 `.env` 中配置构建镜像源：

```env
APT_MIRROR=https://mirrors.aliyun.com/debian
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
VITOOM_WHEEL_BASE_URL=http://192.168.31.17
VITOOM_TARGET_ARCH=x86_64
```

### 1.1 下载构建 artifacts

Backend 与推理镜像不再在 `docker build` 过程中下载大文件。构建前必须先运行：

```bash
python scripts/setup_vitoom.py
```

脚本会：

- 选择地区（中国大陆会写入阿里云 apt/pip 镜像到 `.env`）
- 选择模式 **[1] 镜像构建准备**，并自动检测 CPU 架构（`x86_64` / `aarch64`）
- 交互式选择要准备的组件（Backend 与推理服务可分机器部署，按需下载）
- 对 E5 模型（`Xenova/multilingual-e5-small`）按地区与网络探测 HuggingFace / ModelScope，优先使用更快的源
- 将文件下载到约定目录，已存在则跳过
- 交互式选择界面语言（中文 / 日本語 / English）与网络地区（中国大陆 / 其他）

下载后的目录结构：

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

文件清单与版本见 `docker/build-artifacts.manifest.json`。缺文件时 `docker compose build` 会直接失败并提示运行上述命令。

脚本在需要下载 E5 模型时会自动检测 `hf` / `huggingface_hub` 和 `modelscope`；若未安装，会通过 `pip install` 自动安装（会读取 `.env` 中的 `PIP_INDEX_URL`）。

## 2. 构建 Backend 镜像

Backend 镜像包含 FastAPI 后端、前端静态页面、SQLite 初始化与迁移、内置 Elasticsearch、LibreOffice、Pandoc 和 `multilingual-e5-small-onnx`。

x86_64 镜像：

```bash
VITOOM_TARGET_ARCH=x86_64 docker compose build backend
```

NVIDIA Spark / aarch64 镜像：

```bash
VITOOM_TARGET_ARCH=aarch64 docker compose build backend
```

`VITOOM_TARGET_ARCH` 必须与 Docker 构建平台一致（在 x86_64 机器上构建 aarch64 镜像需使用 `docker buildx build --platform linux/arm64`，并同样传入 `VITOOM_TARGET_ARCH=aarch64`）。

确认镜像：

```bash
docker images vitoom-backend
```

导出 x86_64 镜像（tag 与 `scripts/vitoom_setup/constants.py` 一致，供 setup 脚本写入的 `.env` 直接使用）：

```bash
mkdir -p images/x86_64
docker tag vitoom-backend:latest tonera/vitoom-backend:latest-x86_64
docker save -o images/x86_64/vitoom-backend-latest-x86_64.tar tonera/vitoom-backend:latest-x86_64
```

导出 NVIDIA Spark / aarch64 镜像：

```bash
mkdir -p images/aarch64
docker tag vitoom-backend:latest tonera/vitoom-backend:latest-aarch64
docker save -o images/aarch64/vitoom-backend-latest-aarch64.tar tonera/vitoom-backend:latest-aarch64
```

## 3. 构建推理基础镜像

推理镜像使用 `docker-compose.inference.yml` 构建。该文件只用于开发者和交付包维护者构建，不用于最终用户部署。

构建公共 Python runtime：

```bash
docker compose -f docker-compose.inference.yml --profile build-base build python-runtime
```

确认本地已有基础镜像：

```bash
docker images vitoom-python-runtime
```

如果跳过这一步，后续 `torch-runtime` 的 `FROM vitoom-python-runtime:py3.11` 会尝试从 Docker Hub 拉取同名镜像，并出现 `pull access denied`。

按目标 CUDA/Torch 组合构建 PyTorch runtime。当前不再交付 `cu128`，统一使用 `cu130`，并维护两条基础镜像。基础镜像只包含运行时依赖，不再内置 CUDA 扩展现场编译工具链：

- 默认基线：`cu130 + torch 2.11.0`，用于 Visual 等主线推理镜像。
- 兼容基线：`cu130 + torch 2.9.1`，用于仍需要 torch 2.9.1 依赖栈的服务或实验镜像。

构建 `cu130 + torch 2.11.0`：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
TORCH_EXTRA_INDEX_URL=https://pypi.org/simple \
TORCH_VERSION=2.11.0 \
TORCHVISION_VERSION=0.26.0 \
TORCHAUDIO_VERSION=2.11.0 \
VITOOM_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.11-py3.11-cu130 \
docker compose -f docker-compose.inference.yml --profile build-base build torch-runtime
```

构建 `cu130 + torch 2.9.1`：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
TORCH_EXTRA_INDEX_URL=https://pypi.org/simple \
TORCH_VERSION=2.9.1 \
TORCHVISION_VERSION=0.24.1 \
TORCHAUDIO_VERSION=2.9.1 \
VITOOM_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.9.1-py3.11-cu130 \
docker compose -f docker-compose.inference.yml --profile build-base build torch-runtime
```

## 4. 预编译 wheel 说明

正式交付镜像不再支持现场编译 `flash_attn`、`spas_sage_attn`、`nunchaku`、`turbodiffusion`。构建前必须通过 `setup_vitoom.py` 下载 wheel 到 `docker/inference/wheels/<arch>/`。

当前支持边界：

- x86_64：普通用户，覆盖 RTX 30 / 40 / 50 系消费级 NVIDIA 显卡。
- aarch64：NVIDIA Spark 平台（DGX Spark / RTX Spark），GB10 / Blackwell，构建目标按 `sm120` 覆盖；镜像 tag 后缀为 `nvidia-spark`。
- Visual / Text / Mini 等主线服务：`torch 2.11 + cu130 + Python 3.11`。
- Audio：`torch 2.9.1 + cu130 + Python 3.11`，必须使用单独编译的 `flash_attn` wheel（文件名含 `torch2.9`），不能复用 torch 2.11 的 wheel。

缺少预编译 wheel 或匹配到多个 wheel 时，构建会失败。

预编译 wheel 的 **glibc 版本必须不高于基础镜像**。`python-runtime` 基于 Debian Trixie（glibc 2.38+），与 NVIDIA Spark 宿主机编译的 aarch64 wheel 对齐；在 glibc 更旧的系统上编译的 x86_64 wheel 仍可正常使用。

如果本地已有旧版 Bookworm 基础镜像，需先重建 `python-runtime` 与 `torch-runtime`，再构建服务镜像。

## 5. 构建 Visual 推理镜像

先准备 wheel：

```bash
python scripts/setup_vitoom.py
```

x86_64 镜像：

```bash
VITOOM_TARGET_ARCH=x86_64 \
VITOOM_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.11-py3.11-cu130 \
VITOOM_VISUAL_IMAGE=vitoom-inference-visual:experimental-cu130-torch2.11-x86_64 \
docker compose -f docker-compose.inference.yml build visual
```

NVIDIA Spark / aarch64 镜像：

```bash
python scripts/setup_vitoom.py

VITOOM_TARGET_ARCH=aarch64 \
VITOOM_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.11-py3.11-cu130 \
VITOOM_VISUAL_IMAGE=vitoom-inference-visual:experimental-cu130-torch2.11-aarch64-nvidia-spark \
docker compose -f docker-compose.inference.yml build visual
```

确认镜像：

```bash
docker images vitoom-inference-visual
```


## 6. 构建 Text 推理镜像

Text 镜像不需要额外预编译 wheel。

x86_64 镜像：

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

NVIDIA Spark / aarch64 镜像：

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

确认镜像：

```bash
docker images vitoom-inference-text
```

## 7. 构建 Audio 推理镜像

先准备 wheel：

```bash
python scripts/setup_vitoom.py
```

x86_64 镜像：

```bash
VITOOM_TARGET_ARCH=x86_64 \
VITOOM_AUDIO_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.9.1-py3.11-cu130 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
AUDIO_FLASHINFER_VERSION=0.5.3 \
VITOOM_AUDIO_IMAGE=vitoom-inference-audio:experimental-cu130-torch2.9.1-x86_64 \
docker compose -f docker-compose.inference.yml build audio
```

NVIDIA Spark / aarch64 镜像：

```bash
python scripts/setup_vitoom.py

VITOOM_TARGET_ARCH=aarch64 \
VITOOM_AUDIO_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.9.1-py3.11-cu130 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
AUDIO_FLASHINFER_VERSION=0.5.3 \
VITOOM_AUDIO_IMAGE=vitoom-inference-audio:experimental-cu130-torch2.9.1-aarch64-nvidia-spark \
docker compose -f docker-compose.inference.yml build audio
```

确认镜像：

```bash
docker images vitoom-inference-audio
```

## 8. 构建 Mini 推理镜像

Mini 镜像承载 OCR 等小模型服务，基于 `torch 2.11 + cu130` 基础镜像。

先准备 wheel：

```bash
python scripts/setup_vitoom.py
```

x86_64 镜像：

```bash
VITOOM_TARGET_ARCH=x86_64 \
VITOOM_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.11-py3.11-cu130 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
VITOOM_MINI_IMAGE=vitoom-inference-mini:experimental-cu130-torch2.11-x86_64 \
docker compose -f docker-compose.inference.yml build mini
```

NVIDIA Spark / aarch64 镜像：

```bash
python scripts/setup_vitoom.py

VITOOM_TARGET_ARCH=aarch64 \
VITOOM_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.11-py3.11-cu130 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
VITOOM_MINI_IMAGE=vitoom-inference-mini:experimental-cu130-torch2.11-aarch64-nvidia-spark \
docker compose -f docker-compose.inference.yml build mini
```

确认镜像：

```bash
docker images vitoom-inference-mini
```

## 9. 构建 Download 推理镜像

Download 镜像负责模型和资源下载任务，基于公共 Python runtime，不依赖 Torch runtime。

构建：

```bash
VITOOM_PYTHON_BASE_IMAGE=vitoom-python-runtime:py3.11 \
docker compose -f docker-compose.inference.yml build download
```

确认镜像：

```bash
docker images vitoom-inference-download
```

## 10. 导出推理镜像

导出 x86_64 镜像（tag 与 `scripts/vitoom_setup/constants.py` 一致）：

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

导出 NVIDIA Spark / aarch64 镜像：

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
