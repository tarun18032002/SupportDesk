import os
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from dotenv import load_dotenv


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


dataset_path = "dataset/data" 

local_index_path = "policy_index" 

# sentence chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20


# Embedding model settings
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")  # or "text-embedding-3-large" for better accuracy
Settings.llm = OpenAI(model="gpt-6-luna")

# Retrieval 
TOP_K = 3
 
