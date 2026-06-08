# syntax=docker/dockerfile:1.7
ARG VITOOM_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.9.1-py3.11-cu130
FROM ${VITOOM_TORCH_BASE_IMAGE}

ARG TARGETARCH
ARG APT_MIRROR=
ARG PIP_INDEX_URL=
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG VITOOM_TARGET_ARCH=x86_64
ARG FLASHINFER_VERSION=0.5.3
ARG CUDA_KEYRING_VERSION=1.1-1
ARG CUDA_TOOLKIT_PACKAGE_SUFFIX=

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    PYTHONPATH=/app:/app/inference

WORKDIR /app

RUN if [ -n "$PIP_INDEX_URL" ]; then python -m pip config set global.index-url "$PIP_INDEX_URL"; fi

COPY inference ./inference
COPY config/default.yaml ./config/default.yaml
COPY sitecustomize.py ./sitecustomize.py
COPY docker/inference/common ./docker/inference/common
COPY docker/inference/services/audio ./docker/inference/services/audio

RUN --mount=type=cache,id=vitoom-pip-cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=bind,source=docker/inference/wheels,target=/vitoom-wheels,readonly \
    set -eux; \
    if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|http://deb.debian.org/debian|$APT_MIRROR|g; s|http://deb.debian.org/debian-security|$APT_MIRROR-security|g" /etc/apt/sources.list.d/debian.sources; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        libnuma1; \
    case "${TARGETARCH:-}" in \
        amd64) cuda_repo_arch="x86_64"; cuda_default_suffix="13-1" ;; \
        arm64) cuda_repo_arch="sbsa"; cuda_default_suffix="13-2" ;; \
        *) echo "Unsupported Docker target architecture for CUDA Toolkit: ${TARGETARCH:-<empty>}" >&2; exit 1 ;; \
    esac; \
    cuda_pkg_suffix="${CUDA_TOOLKIT_PACKAGE_SUFFIX:-$cuda_default_suffix}"; \
    cuda_home_version="$(printf '%s' "$cuda_pkg_suffix" | sed 's/-/./')"; \
    curl -fL \
        "https://developer.download.nvidia.com/compute/cuda/repos/debian13/${cuda_repo_arch}/cuda-keyring_${CUDA_KEYRING_VERSION}_all.deb" \
        -o /tmp/cuda-keyring.deb; \
    dpkg -i /tmp/cuda-keyring.deb; \
    rm -f /tmp/cuda-keyring.deb; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        "cuda-nvcc-${cuda_pkg_suffix}"; \
    ln -sfn "/usr/local/cuda-${cuda_home_version}" /usr/local/cuda; \
    test -x /usr/local/cuda/bin/ptxas; \
    rm -rf /var/lib/apt/lists/*; \
    wheel_dir="/vitoom-wheels/${VITOOM_TARGET_ARCH}"; \
    if [ ! -d "$wheel_dir" ]; then \
        echo "Missing wheel directory: ${wheel_dir}" >&2; \
        echo "Run: python scripts/setup_vitoom.py" >&2; \
        exit 1; \
    fi; \
    flash_attn_wheels="$(find "$wheel_dir" -maxdepth 1 -type f -name 'flash_attn-*torch2.9*.whl' | sort)"; \
    flash_attn_count="$(printf '%s\n' "$flash_attn_wheels" | sed '/^$/d' | wc -l | tr -d ' ')"; \
    if [ "$flash_attn_count" != "1" ]; then \
        echo "Expected exactly one flash_attn-*torch2.9*.whl in ${wheel_dir}, found ${flash_attn_count}." >&2; \
        echo "Run: python scripts/setup_vitoom.py" >&2; \
        exit 1; \
    fi; \
    python -m pip install --no-deps "$flash_attn_wheels"; \
    if [ -n "$TORCH_INDEX_URL" ]; then \
        python -m pip install -r /app/docker/inference/services/audio/requirements.txt --extra-index-url "$TORCH_INDEX_URL"; \
    else \
        python -m pip install -r /app/docker/inference/services/audio/requirements.txt; \
    fi; \
    TORCH_INDEX_URL="$TORCH_INDEX_URL" FLASHINFER_VERSION="$FLASHINFER_VERSION" \
        bash /app/docker/inference/services/audio/post-install.sh; \
    bash /app/docker/inference/common/link-triton-ptxas.sh; \
    cp -R /app/inference/config /app/inference/config.defaults; \
    rm -f /app/inference/config.defaults/inference.yaml \
      /app/inference/config.defaults/image.yaml \
      /app/inference/config.defaults/video.yaml \
      /app/inference/config.defaults/text.yaml \
      /app/inference/config.defaults/translate.yaml \
      /app/inference/config.defaults/qwen_asr.yaml \
      /app/inference/config.defaults/qwen_tts.yaml \
      /app/inference/config.defaults/mini.yaml \
      /app/inference/config.defaults/download.yaml; \
    chmod +x /app/docker/inference/common/docker-entrypoint.sh /app/docker/inference/services/audio/docker-entrypoint.sh /app/docker/inference/common/link-triton-ptxas.sh

VOLUME ["/app/resources", "/app/inference/config", "/app/logs"]

ENTRYPOINT ["/app/docker/inference/services/audio/docker-entrypoint.sh"]
CMD []
