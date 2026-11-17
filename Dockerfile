# SupportDesk — FastAPI app (uvicorn) + HuggingFace embeddings + Qdrant client
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /code

# System deps for sentence-transformers / torch wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install deps first for better layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Application source
COPY main.py ./
COPY src ./src
COPY frontend ./frontend

# App expects `from config` / `from settings` and dataset at ./dataset/data
RUN ln -s /code/src/dataset /code/dataset

# Install the project (includes fastapi[standard] + uvicorn)
RUN uv sync --frozen --no-dev

ENV PATH="/code/.venv/bin:$PATH" \
    PYTHONPATH="/code/src:/code" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/code/.cache/huggingface

EXPOSE 8000

# FastAPI app object: main:app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
