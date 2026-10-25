ARG BASE_IMAGE=registry.cn-qingdao.aliyuncs.com/wod/comfyui:0.1.0
FROM ${BASE_IMAGE}

USER root

ARG COMFYUI_REVISION=067f853305e74a051dd17f15964d5b0e7887abf5
ARG PYPI_INDEX_URL=https://pypi.org/simple

LABEL org.opencontainers.image.title="VerdantFlare Image MOSS API" \
      org.opencontainers.image.version="0.1.4" \
      org.opencontainers.image.source="https://github.com/verdantflarehub/verdantflare-app-image" \
      org.opencontainers.image.revision="${COMFYUI_REVISION}" \
      org.opencontainers.image.licenses="GPL-3.0-only"

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    HOME=/tmp \
    TRITON_CACHE_DIR=/tmp/triton \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MOSS_COMFYUI_ROOT=/opt/comfyui \
    MOSS_IMAGE_MODEL_ROOT=/models/moss-image

RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN rm -rf /opt/comfyui \
    && git clone --filter=blob:none --no-checkout https://github.com/Comfy-Org/ComfyUI.git /opt/comfyui \
    && cd /opt/comfyui \
    && git checkout --detach "${COMFYUI_REVISION}" \
    && test "$(git rev-parse HEAD)" = "${COMFYUI_REVISION}" \
    && test -f LICENSE \
    && rm -rf .git \
    && python -m pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" -r requirements.txt

WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
RUN python -m pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" . \
    && python -m pip freeze > /app/verdantflare-python-packages.txt

COPY upstream.lock.json /app/upstream.lock.json
COPY src/ /app/src/

RUN chown -R 1000:1000 /opt/comfyui /app

USER 1000:1000
EXPOSE 8000

HEALTHCHECK --start-period=60s --interval=10s --timeout=5s --retries=6 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
