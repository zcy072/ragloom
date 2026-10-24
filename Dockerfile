# 基础镜像前缀默认走 Daocloud 的 Docker Hub 镜像；可用 DOCKER_LIBRARY_PREFIX=docker.io/library 改回官方源。
# Python 包索引通过 PIP_INDEX_URL 单独配置。
ARG DOCKER_LIBRARY_PREFIX=docker.m.daocloud.io/library
FROM ${DOCKER_LIBRARY_PREFIX}/python:3.10.22-slim AS runtime

WORKDIR /app

# 这一层不依赖 requirements.lock，后续更新 Python 依赖时无需重装系统库。
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_ROOT_USER_ACTION=ignore

COPY requirements.lock .
# BuildKit 缓存即使 requirements.lock 变化也会保留已下载的 wheel。
RUN --mount=type=cache,id=pymodel-pip,target=/root/.cache/pip,sharing=locked \
    pip install --prefer-binary --index-url "${PIP_INDEX_URL}" -r requirements.lock

# Runtime identity is independent from dependency layers so UID/GID changes stay cheap.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home app \
    && mkdir -p /home/app/.paddlex \
    && chown -R app:app /home/app/.paddlex

# Keep the runtime image limited to application code. Tests, frontend
# sources, and project notes are mounted from the workspace only when needed.
COPY api.py app_storage.py ./
COPY routers ./routers
COPY services ./services

ENV HOME=/home/app
USER app

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# Static-analysis tools live in a test-only image layer, never in the runtime image or host workspace.
FROM runtime AS check

USER root
RUN --mount=type=cache,id=pymodel-check-pip,target=/root/.cache/pip,sharing=locked \
    PIP_CACHE_DIR=/root/.cache/pip \
    pip install --prefer-binary --index-url "${PIP_INDEX_URL}" \
    ruff==0.16.0 \
    "pyright[nodejs]==1.1.408" \
    uv==0.11.18

USER app
