from llama_index.core import Settings

from config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    OPENAI_API_KEY,
    logger,
)


def initialize_settings():
    """Configure the shared LlamaIndex embedding model.

    OpenAI embeddings keep the deploy bundle under Vercel's size limit.
    HuggingFace / sentence-transformers pull PyTorch (~multi-GB) and are
    optional local-only installs.
    """

    if EMBEDDING_PROVIDER == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        if not OPENAI_API_KEY:
            raise RuntimeError(
                "Set OPENAI_API_KEY in .env when EMBEDDING_PROVIDER=openai"
            )

        Settings.embed_model = OpenAIEmbedding(
            model=EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
        )
        logger.info(
            "Embeddings ready: provider=openai model=%s dim=%s",
            EMBEDDING_MODEL,
            EMBEDDING_DIMENSION,
        )
        return

    if EMBEDDING_PROVIDER == "huggingface":
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFace embeddings require optional deps. "
                "Install with: uv sync --extra local-embed"
            ) from exc

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            device="cpu",
            normalize=True,
        )
        logger.info(
            "Embeddings ready: provider=huggingface model=%s dim=%s",
            EMBEDDING_MODEL,
            EMBEDDING_DIMENSION,
        )
        return

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r}. "
        "Use 'openai' or 'huggingface'."
    )
