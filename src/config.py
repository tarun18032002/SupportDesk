import os
import logging


from dotenv import load_dotenv


# logging 
logger = logging.getLogger(__name__)

load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


dataset_path = "dataset/data" 

# local_index_path = "./qdrant_db" 

# qdrant 
db_url = os.getenv("DB_URL")
db_api_key = os.getenv("DB_API_KEY")

# sentence chunking
CHUNK_SIZE = 256
CHUNK_OVERLAP = 25


# Retrieval 
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384

DEFAULT_TOP_K = 3

RETRIVER_MODE = os.getenv("RETRIVER_MODE", "Dense")  

COLLECTION_NAME = "supportdesk_documents"

# llm 
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b")

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")






