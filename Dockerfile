# Origin (or your brand of it). The models run in Ollama: see docker-compose.yml.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY origin ./origin
RUN pip install .
COPY main.py ./

# Listen on every interface inside the container (publish the port on 127.0.0.1, or
# enable authentication before exposing it: see docs/deployment.md). Data and the
# brand live outside the image.
ENV ORIGIN_ENV=prod \
    ORIGIN_HOST=0.0.0.0 \
    ORIGIN_PORT=8000 \
    OLLAMA_BASE_URL=http://ollama:11434 \
    SQLITE_PATH=/data/origin.db \
    CHROMA_PERSIST_DIR=/data/chroma \
    CALIBRATION_PATH=/data/calibration.json \
    MEMORY_BACKUP_DIR=/data/backups \
    ORIGIN_BRAND_PATH=/brand/brand.json

RUN useradd --create-home --uid 1000 origin && mkdir -p /data /brand && chown origin /data
USER origin
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/brand', timeout=4)"

CMD ["python", "main.py"]
