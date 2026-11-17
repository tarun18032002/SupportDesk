from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,StorageContext
from llama_index.core.node_parser import MarkdownNodeParser,SentenceSplitter
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import StorageContext,VectorStoreIndex


from src.vector_store import store_nodes_in_index

from pathlib import Path

# local import
from .config import dataset_path, CHUNK_SIZE,CHUNK_OVERLAP,logger

def read_dataset(dataset_path):
    """Read the dataset using simple directory reader from llama_index.core."""
    if not Path(dataset_path).exists():
        raise "folder path is not exist, check the folder path : " + dataset_path
    
    documents = SimpleDirectoryReader(input_dir=dataset_path).load_data()
    logger.info(f"Total number of documents: {len(documents)}")
    return documents

def custom_sentence_splitter(text: str) -> list[str]:
    """
    Split text at sentence boundaries without using NLTK.
    """
    import re
    return re.split(r'(?<=[.!?])\s+', text.strip())

def split_documents(documents):
    """Split the documents into smaller chunks for better useage of polcies."""
    markdown_parser  = MarkdownNodeParser()
    heading_nodes  = markdown_parser .get_nodes_from_documents(documents,show_progress=True)
    sentence_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,chunking_tokenizer_fn=custom_sentence_splitter)
    nodes = sentence_splitter.get_nodes_from_documents(heading_nodes, show_progress=True)
    tokenizer = sentence_splitter._tokenizer
    logger.info(f"\nTotal Markdown sections: {len(heading_nodes)}")
    logger.info(f"Total final chunks: {len(nodes)}")
    for index, node in enumerate(nodes):

        token_count = len(tokenizer(node.text))

        logger.info("\n" + "=" * 70)
        logger.info(f"Chunk {index +1}")
        logger.info("="*70)

        logger.info(f"Node ID       : {node.node_id}")
        logger.info(f"Characters    : {len(node.text)}")
        logger.info(f"Token Count   : {token_count}")
        logger.info(f"Metadata      : {node.metadata}")
        logger.info(f"Content       :\n{node.text}")

    return nodes

def store_chunks_locally(chunks, persist_dir):
    """Store the chunks locally for later use."""
    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(chunks, storage_context=storage_context)
    index.storage_context.persist(persist_dir=persist_dir)
    logger.info(f"Chunks stored locally at {persist_dir}")
    return index