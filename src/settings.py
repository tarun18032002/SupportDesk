from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .config import EMBEDDING_MODEL


def initialize_settings():

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        device="cpu",
        normalize=True,
    )