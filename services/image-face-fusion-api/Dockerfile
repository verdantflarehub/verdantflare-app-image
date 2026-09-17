ARG BASE_IMAGE=docker.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

USER root

ARG DEBIAN_FRONTEND=noninteractive
ARG PYPI_INDEX_URL=https://pypi.org/simple

LABEL org.opencontainers.image.source="https://github.com/verdantflarehub/verdantflare-app-image"
LABEL org.opencontainers.image.title="VerdantFlare Image Face Fusion API"
LABEL org.opencontainers.image.version="0.2.0"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-venv \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" . && pip check

COPY upstream.lock.json /app/upstream.lock.json
COPY scripts/ /app/scripts/
RUN python3 /app/scripts/fetch_upstream.py /opt/facefusion

COPY src/ /app/src/

ENV FACEFUSION_SOURCE_ROOT=/opt/facefusion \
    FACEFUSION_MODEL_ROOT=/models/facefusion \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --start-period=30s --interval=10s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
