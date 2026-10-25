# Official PyTorch CUDA 13 runtime: comfy-kitchen only enables its CUDA kernels (INT8 / ConvRot) on cu130+.
ARG BASE_IMAGE=pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime
FROM ${BASE_IMAGE}

USER root

ARG COMFYUI_REVISION=19ae172bff6c9e4c738e7f2e00fe087f5b5b46ab
ARG PYPI_INDEX_URL=https://pypi.org/simple

LABEL org.opencontainers.image.title="VerdantFlare Image MOSS API" \
      org.opencontainers.image.version="0.2.0" \
      org.opencontainers.image.source="https://github.com/verdantflarehub/verdantflare-app-image" \
      org.opencontainers.image.revision="${COMFYUI_REVISION}" \
      org.opencontainers.image.licenses="GPL-3.0-only"

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/tmp \
    TRITON_CACHE_DIR=/tmp/triton \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MOSS_COMFYUI_ROOT=/opt/comfyui \
    MOSS_IMAGE_MODEL_ROOT=/models/moss-image

# git for the pinned checkout; gcc and python headers for Triton's first-call driver build on Blackwell
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git gcc libc6-dev python3-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
# Keep the base image's torch stack: constrain ComfyUI's bare torch requirements to the installed versions.
RUN git clone --filter=blob:none --no-checkout https://github.com/Comfy-Org/ComfyUI.git /opt/comfyui \
    && cd /opt/comfyui \
    && git checkout --detach "${COMFYUI_REVISION}" \
    && test "$(git rev-parse HEAD)" = "${COMFYUI_REVISION}" \
    && test -f LICENSE \
    && rm -rf .git \
    && python -m pip freeze | grep -iE '^(torch|torchvision|torchaudio)==' > /tmp/torch-constraints.txt \
    && python -m pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" -c /tmp/torch-constraints.txt -r requirements.txt \
    && python -c "import torch, sysconfig, os; assert torch.version.cuda.startswith('13'), torch.version.cuda; assert os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h'))"

WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
RUN python -m pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" -c /tmp/torch-constraints.txt . \
    && python -m pip freeze > /app/verdantflare-python-packages.txt

COPY upstream.lock.json /app/upstream.lock.json
COPY src/ /app/src/

RUN chown -R 1000:1000 /opt/comfyui /app

USER 1000:1000
EXPOSE 8000

HEALTHCHECK --start-period=60s --interval=10s --timeout=5s --retries=6 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
