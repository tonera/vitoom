# syntax=docker/dockerfile:1.7
ARG VITOOM_TORCH_BASE_IMAGE=vitoom-torch-runtime:2.11-py3.11-cu130
FROM ${VITOOM_TORCH_BASE_IMAGE}

ARG APT_MIRROR=
ARG PIP_INDEX_URL=
ARG TORCH_INDEX_URL=
ARG VITOOM_TARGET_ARCH=x86_64

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app:/app/inference

WORKDIR /app

RUN set -eux; \
    if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|http://deb.debian.org/debian|$APT_MIRROR|g; s|http://deb.debian.org/debian-security|$APT_MIRROR-security|g" /etc/apt/sources.list.d/debian.sources; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential; \
    rm -rf /var/lib/apt/lists/*; \
    if [ -n "$PIP_INDEX_URL" ]; then python -m pip config set global.index-url "$PIP_INDEX_URL"; fi

COPY inference ./inference
COPY config/default.yaml ./config/default.yaml
COPY sitecustomize.py ./sitecustomize.py
COPY docker/inference/common ./docker/inference/common
COPY docker/inference/services/visual ./docker/inference/services/visual

RUN --mount=type=cache,id=vitoom-pip-cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=bind,source=docker/inference/wheels,target=/vitoom-wheels,readonly \
    set -eux; \
    wheel_dir="/vitoom-wheels/${VITOOM_TARGET_ARCH}"; \
    if [ ! -d "$wheel_dir" ]; then \
        echo "Missing wheel directory: ${wheel_dir}" >&2; \
        echo "Run: python scripts/setup_vitoom.py" >&2; \
        exit 1; \
    fi; \
    if [ -n "$TORCH_INDEX_URL" ]; then \
        python -m pip install -r /app/docker/inference/services/visual/requirements.txt --extra-index-url "$TORCH_INDEX_URL"; \
    else \
        python -m pip install -r /app/docker/inference/services/visual/requirements.txt; \
    fi; \
    VITOOM_WHEELS_DIR="$wheel_dir" VITOOM_TARGET_ARCH="$VITOOM_TARGET_ARCH" \
        bash /app/docker/inference/services/visual/post-install.sh; \
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
    chmod +x /app/docker/inference/common/docker-entrypoint.sh /app/docker/inference/services/visual/docker-entrypoint.sh

VOLUME ["/app/resources", "/app/inference/config", "/app/logs"]

ENTRYPOINT ["/app/docker/inference/services/visual/docker-entrypoint.sh"]
CMD []
