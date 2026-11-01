from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,StorageContext
from llama_index.core.node_parser import MarkdownNodeParser,SentenceSplitter
from llama_index.core.vector_stores import SimpleVectorStore

# local import
from config import dataset_path, CHUNK_SIZE,CHUNK_OVERLAP,local_index_path

def read_dataset(dataset_path):
    """Read the dataset using simple directory reader from llama_index.core."""
    documents = SimpleDirectoryReader(input_dir=dataset_path).load_data()
    print(f"Total number of documents: {len(documents)}")
    return documents

def split_documents(documents):
    """Split the documents into smaller chunks for better useage of polcies."""
    markdown_parser  = MarkdownNodeParser()
    heading_nodes  = markdown_parser .get_nodes_from_documents(documents,show_progress=True)
    sentence_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = sentence_splitter.get_nodes_from_documents(heading_nodes)
    print(f"Total number of chunks: {len(nodes)}")
    return nodes

def store_chunks_locally(chunks, persist_dir):
    """Store the chunks locally for later use."""
    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(chunks, storage_context=storage_context)
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"Chunks stored locally at {persist_dir}")
    return index


documents = read_dataset(dataset_path)
nodes = split_documents(documents)
index = store_chunks_locally(nodes,local_index_path)