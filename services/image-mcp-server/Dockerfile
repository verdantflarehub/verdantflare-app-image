FROM docker.io/library/python:3.11.10-slim

ARG PYPI_INDEX_URL=https://pypi.org/simple
ARG SERVICE_VERSION=v0.2.67
LABEL org.opencontainers.image.source="https://github.com/verdantflarehub/verdantflare-app-image"
LABEL org.opencontainers.image.title="VerdantFlare Image MCP Server"
LABEL org.opencontainers.image.version="0.2.67"
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN SERVICE_VERSION="${SERVICE_VERSION}" python3 -c "import os, tomllib; decl = tomllib.loads(open('pyproject.toml', 'rb').read())['project']['version'].strip(); raw = os.environ.get('SERVICE_VERSION', '').strip(); norm = raw[1:] if raw.startswith('v') else raw; assert not norm or norm == decl, f'SERVICE_VERSION ({raw}) != pyproject.toml ({decl})'"
ENV IMAGE_MCP_SERVICE_VERSION=${SERVICE_VERSION}
RUN pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" . && pip check

COPY src ./src
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["python3", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
