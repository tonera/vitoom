# syntax=docker/dockerfile:1.7
ARG VITOOM_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.11-py3.11-cu130
FROM ${VITOOM_TORCH_BASE_IMAGE}

ARG TARGETARCH
ARG APT_MIRROR=
ARG PIP_INDEX_URL=
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG VLLM_VERSION=0.21.0
ARG CUDA_KEYRING_VERSION=1.1-1
ARG CUDA_TOOLKIT_PACKAGE_SUFFIX=

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PYTHONPATH=/app:/app/inference

WORKDIR /app

RUN set -eux; \
    if [ -n "$PIP_INDEX_URL" ]; then python -m pip config set global.index-url "$PIP_INDEX_URL"; fi; \
    if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|http://deb.debian.org/debian|$APT_MIRROR|g; s|http://deb.debian.org/debian-security|$APT_MIRROR-security|g" /etc/apt/sources.list.d/debian.sources; \
    fi

COPY inference ./inference
COPY config/default.yaml ./config/default.yaml
COPY sitecustomize.py ./sitecustomize.py
COPY docker/inference/common ./docker/inference/common
COPY docker/inference/services/text ./docker/inference/services/text

RUN --mount=type=cache,id=vitoom-pip-cache,target=/root/.cache/pip,sharing=locked \
    set -eux; \
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
        build-essential \
        "cuda-cccl-${cuda_pkg_suffix}" \
        "cuda-cudart-dev-${cuda_pkg_suffix}" \
        "cuda-nvcc-${cuda_pkg_suffix}" \
        "libcurand-dev-${cuda_pkg_suffix}" \
        libnuma1 \
        python3-dev; \
    ln -sfn "/usr/local/cuda-${cuda_home_version}" /usr/local/cuda; \
    test -f /usr/local/cuda/include/curand.h; \
    /usr/local/cuda/bin/nvcc --version; \
    TORCH_INDEX_URL="$TORCH_INDEX_URL" VLLM_VERSION="$VLLM_VERSION" \
        bash /app/docker/inference/services/text/post-install.sh; \
    python -m pip install -r /app/docker/inference/services/text/requirements.txt; \
    rm -rf /var/lib/apt/lists/*; \
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
    chmod +x /app/docker/inference/common/docker-entrypoint.sh /app/docker/inference/services/text/docker-entrypoint.sh

VOLUME ["/app/resources", "/app/inference/config", "/app/logs"]

ENTRYPOINT ["/app/docker/inference/services/text/docker-entrypoint.sh"]
CMD []
