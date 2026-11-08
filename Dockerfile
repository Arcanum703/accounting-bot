FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DB_PATH=/data/accounting.db \
    CHROMA_PATH=/data/chroma \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so the container starts offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY src ./src
COPY data/bank_statement.csv ./data/bank_statement.csv

RUN useradd --create-home bot && mkdir -p /data && chown bot /data
USER bot
VOLUME /data

CMD ["python", "-m", "src.bot.main"]
