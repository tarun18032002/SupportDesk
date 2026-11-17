# Slim production image — OpenAI embeddings (no PyTorch)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /code

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY main.py ./
COPY src ./src
COPY frontend ./frontend

RUN ln -s /code/src/dataset /code/dataset

RUN uv sync --frozen --no-dev

ENV PATH="/code/.venv/bin:$PATH" \
    PYTHONPATH="/code/src:/code" \
    PYTHONUNBUFFERED=1 \
    EMBEDDING_PROVIDER=openai \
    LLM_PROVIDER=openai

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
