import os
import logging

from dotenv import load_dotenv


logger = logging.getLogger(__name__)

load_dotenv()

dataset_path = "src/dataset/data"

# Qdrant
db_url = os.getenv("DB_URL")
db_api_key = os.getenv("DB_API_KEY")

# Chunking
CHUNK_SIZE = 256
CHUNK_OVERLAP = 25

# Embeddings
# Use "openai" for Vercel / slim deploys (no PyTorch).
# Use "huggingface" only on machines where local-embed extras are installed.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()

if EMBEDDING_PROVIDER == "huggingface":
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
else:
    # Default deploy-safe path
    EMBEDDING_PROVIDER = "openai"
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

DEFAULT_TOP_K = 3

RETRIVER_MODE = os.getenv("RETRIVER_MODE", "Dense")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "supportdesk_documents")

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434",
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
