from llama_index.core import StorageContext,load_index_from_storage
from llama_index.core.query_engine import CitationQueryEngine
from config import local_index_path,TOP_K

def load_local_index(index_path):
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)
    return index

def query_index(index,query):
    query_engine =CitationQueryEngine.from_args(
                    index,
                    similarity_top_k=TOP_K,
                    streaming=True,
                )
    return query_engine.query(query)

import time
print("Loading local index...",time.time())
index = load_local_index(local_index_path)
print(query_index(index,"Can I return unused, unopened headphones delivered 20 days ago? I have the original box and receipt."))
print("Querying local index...",time.time())