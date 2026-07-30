FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    NLTK_DATA=/tmp/nltk_data \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    XDG_CACHE_HOME=/tmp/.cache \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt /app/requirements.docker.txt

RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r /app/requirements.docker.txt

RUN python -c "import nltk; nltk.download('punkt', download_dir='/tmp/nltk_data'); nltk.download('punkt_tab', download_dir='/tmp/nltk_data')"

RUN useradd -m -u 1000 appuser \
    && mkdir -p /tmp/huggingface /tmp/nltk_data /tmp/torchinductor /tmp/.cache \
    && chown -R appuser:appuser /tmp/huggingface /tmp/nltk_data /tmp/torchinductor /tmp/.cache /app

USER appuser

COPY backend /app/backend

EXPOSE 8000

CMD ["python", "-m", "backend.docker_entrypoint"]
