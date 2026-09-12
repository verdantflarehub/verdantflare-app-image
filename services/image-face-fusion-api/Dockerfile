ARG BASE_IMAGE=docker.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

USER root

ARG DEBIAN_FRONTEND=noninteractive
ARG PYPI_INDEX_URL=https://pypi.org/simple

LABEL org.opencontainers.image.source="https://github.com/verdantflarehub/verdantflare-app-image"
LABEL org.opencontainers.image.title="VerdantFlare Image Face Fusion API"
LABEL org.opencontainers.image.version="0.1.0"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" -r requirements.txt

COPY src/ /app/src/

ENV INSIGHTFACE_MODEL_ROOT=/models/insightface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

VOLUME ["/models/insightface", "/data/projects"]
EXPOSE 8000

HEALTHCHECK --start-period=30s --interval=10s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
