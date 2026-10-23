FROM nvidia/cuda@sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9

ARG COMFYUI_REVISION=355cd6b14c92c19ac3188fe528e47a3f21ef5178
ARG PYPI_INDEX_URL=https://pypi.org/simple

LABEL org.opencontainers.image.title="VerdantFlare Image MOSS API" \
      org.opencontainers.image.version="0.1.0" \
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
      ca-certificates git gcc python3 python3-dev python3-venv libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && python -m pip install --no-cache-dir --upgrade pip

WORKDIR /opt/comfyui
RUN git clone --filter=blob:none --no-checkout https://github.com/Comfy-Org/ComfyUI.git . \
    && git checkout --detach "${COMFYUI_REVISION}" \
    && test "$(git rev-parse HEAD)" = "${COMFYUI_REVISION}" \
    && test -f LICENSE \
    && rm -rf .git

RUN python -m pip install --no-cache-dir \
      torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
      --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" -r requirements.txt

WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
RUN python -m pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" . \
    && python -m pip check \
    && python -m pip freeze > /app/verdantflare-python-packages.txt

COPY upstream.lock.json /app/upstream.lock.json
COPY src/ /app/src/

RUN chown -R 1000:1000 /opt/comfyui /app

USER 1000:1000
EXPOSE 8000

HEALTHCHECK --start-period=60s --interval=10s --timeout=5s --retries=6 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
