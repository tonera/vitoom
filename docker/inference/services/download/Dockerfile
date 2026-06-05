# syntax=docker/dockerfile:1.7
ARG VITOOM_PYTHON_BASE_IMAGE=vitoom-python-runtime:py3.11
FROM ${VITOOM_PYTHON_BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app:/app/inference

ARG PIP_INDEX_URL=

WORKDIR /app

RUN if [ -n "$PIP_INDEX_URL" ]; then python -m pip config set global.index-url "$PIP_INDEX_URL"; fi

COPY inference ./inference
COPY config/default.yaml ./config/default.yaml
COPY docker/inference ./docker/inference

RUN --mount=type=cache,id=vitoom-pip-cache,target=/root/.cache/pip,sharing=locked \
    set -eux; \
    python -m pip install -r /app/docker/inference/services/download/requirements.txt; \
    bash /app/docker/inference/services/download/post-install.sh; \
    chmod +x /app/docker/inference/common/docker-entrypoint.sh /app/docker/inference/services/download/docker-entrypoint.sh

VOLUME ["/app/resources", "/app/inference/config", "/app/logs"]

ENTRYPOINT ["/app/docker/inference/services/download/docker-entrypoint.sh"]
CMD []
