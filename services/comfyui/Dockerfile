FROM nvidia/cuda@sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9

ARG COMFYUI_REVISION=c2bcbecd82ec5ae66594340b395c24ef0217b238
ARG PYPI_INDEX_URL=https://pypi.org/simple

LABEL org.opencontainers.image.title="VerdantFlare ComfyUI" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.source="https://github.com/Comfy-Org/ComfyUI" \
      org.opencontainers.image.revision="${COMFYUI_REVISION}" \
      org.opencontainers.image.licenses="GPL-3.0-only"

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git python3 python3-venv libgl1 libglib2.0-0 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && python -m pip install --no-cache-dir --upgrade pip

WORKDIR /opt/comfyui
RUN git clone --filter=blob:none --no-checkout https://github.com/Comfy-Org/ComfyUI.git . \
    && git checkout --detach "${COMFYUI_REVISION}" \
    && test "$(git rev-parse HEAD)" = "${COMFYUI_REVISION}" \
    && test -f LICENSE \
    && rm -rf .git

# The cu128 wheel set is required for the target Blackwell GPU. Keep the
# upstream requirements in the image, and record the final resolved versions.
RUN python -m pip install --no-cache-dir \
      torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
      --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" -r requirements.txt \
    && python -m pip check \
    && python -m pip freeze > /opt/comfyui/verdantflare-python-packages.txt

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin comfyui \
    && mkdir -p /data \
    && chown -R comfyui:comfyui /opt/comfyui /data

USER 1000:1000
EXPOSE 8188

CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--base-directory", "/data", "--disable-auto-launch"]
