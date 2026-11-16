
from typing import Sequence, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.vector_stores.qdrant import QdrantVectorStore

from config import DEFAULT_TOP_K, EMBEDDING_DIMENSION


# ==================================================
# 1. QDRANT CONNECTION
# ==================================================

def get_client(
    url: str,
    api_key: Optional[str] = None,
    timeout: int = 30,
    prefer_grpc: bool = False,
) -> QdrantClient:

    if not url:
        raise ValueError("Qdrant URL is required")

    client = QdrantClient(
        url=url,
        api_key=api_key,
        timeout=timeout,
        prefer_grpc=prefer_grpc,
    )

    try:
        # Verify connection and authentication
        client.get_collections()

        return client

    except Exception:
        client.close()
        raise


# ==================================================
# 2. CREATE COLLECTION
# Provisioning operation - not called during startup
# ==================================================

def create_collection(
    client: QdrantClient,
    collection: str,
    vector_size: int = EMBEDDING_DIMENSION,
    distance: Distance = Distance.COSINE,
) -> bool:

    if client.collection_exists(collection):
        raise ValueError(
            f"Collection '{collection}' already exists."
        )

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=vector_size,
            distance=distance,
        ),
    )

    return True

def ensure_collection_exists(
    client: QdrantClient,
    collection: str,
) -> None:

    created = create_collection(
        client=client,
        collection=collection,
    )

    if created:
        print(f"Collection '{collection}' created successfully.")
    else:
        print(f"Collection '{collection}' already exists.")


# ==================================================
# 3. VALIDATE COLLECTION CONFIGURATION
# ==================================================

def validate_collection_config(
    client: QdrantClient,
    collection: str,
    expected_dimension: int = EMBEDDING_DIMENSION,
    expected_distance: Distance = Distance.COSINE,
) -> None:

    info = client.get_collection(collection)

    vectors = info.config.params.vectors

    # This implementation expects one unnamed dense vector.
    if isinstance(vectors, dict):
        raise ValueError(
            "Named vector configuration detected. "
            "This implementation expects a single unnamed vector."
        )

    if vectors.size != expected_dimension:
        raise ValueError(
            f"Vector dimension mismatch. "
            f"Expected {expected_dimension}, "
            f"got {vectors.size}"
        )

    if vectors.distance != expected_distance:
        raise ValueError(
            f"Distance metric mismatch. "
            f"Expected {expected_distance}, "
            f"got {vectors.distance}"
        )


# ==================================================
# 4. ATTACH QDRANT TO LLAMAINDEX
# ==================================================

def attach_client_llama_index(
    collection: str,
    client: QdrantClient,
) -> QdrantVectorStore:

    if not client.collection_exists(collection):
        raise RuntimeError(
            f"Collection '{collection}' does not exist. "
            "Provision the collection before application startup."
        )

    # Validate before connecting
    validate_collection_config(
        client=client,
        collection=collection,
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection,
    )

    return vector_store


# ==================================================
# 5. LOAD EXISTING INDEX
# Requires settings.initialize_settings() first
# ==================================================

def get_stored_index(
    vector_store: QdrantVectorStore,
) -> VectorStoreIndex:

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model,
    )


# ==================================================
# 6. INSERT NODES
# ==================================================

def store_nodes_in_index(
    index: VectorStoreIndex,
    nodes: Sequence[BaseNode],
) -> None:

    if not nodes:
        raise ValueError("No nodes provided for insertion")

    node_ids = [node.node_id for node in nodes]

    if len(node_ids) != len(set(node_ids)):
        raise ValueError(
            "Duplicate node IDs found in the ingestion batch"
        )

    index.insert_nodes(
        list(nodes),
        show_progress=True,
    )


# ==================================================
# 7. RETRIEVE NODES
# ==================================================

def retrieve_nodes(
    index: VectorStoreIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[NodeWithScore]:

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k < 1:
        raise ValueError("top_k must be greater than zero")

    retriever = index.as_retriever(
        similarity_top_k=top_k
    )

    return retriever.retrieve(query)


# ==================================================
# 8. COLLECTION INFORMATION
# ==================================================

def get_collection_info(
    client: QdrantClient,
    collection: str,
) -> dict:

    if not client.collection_exists(collection):
        raise ValueError(
            f"Collection '{collection}' does not exist"
        )

    info = client.get_collection(collection)

    return {
        "collection_name": collection,
        "points_count": info.points_count,
        "indexed_vectors_count": info.indexed_vectors_count,
        "status": str(info.status),
    }


# ==================================================
# 9. COMPLETE INITIALIZATION
# ==================================================

def initialize_qdrant_index(
    url: str,
    collection: str,
    api_key: Optional[str] = None,
    timeout: int = 30,
    prefer_grpc: bool = False,
):

    client = get_client(
        url=url,
        api_key=api_key,
        timeout=timeout,
        prefer_grpc=prefer_grpc,
    )

    try:

        # Attach existing Qdrant collection
        vector_store = attach_client_llama_index(
            collection=collection,
            client=client,
        )

        # Load existing index
        index = get_stored_index(
            vector_store=vector_store,
        )

        # Get collection information
        collection_info = get_collection_info(
            client=client,
            collection=collection,
        )

        return {
            "client": client,
            "vector_store": vector_store,
            "index": index,
            "collection_info": collection_info,
        }

    except Exception:
        client.close()
        raise
