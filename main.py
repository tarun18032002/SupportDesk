# from llm1 import LocalLLMService
from src.vector_store import initialize_qdrant_index,store_nodes_in_index
from settings import initialize_settings
from src.ingest import read_dataset, split_documents
from config import(
    LLM_PROVIDER,
    LLM_MODEL,
    OLLAMA_HOST,
    OPENAI_API_KEY,
    COLLECTION_NAME,
    logger,
    db_url,
    db_api_key

)

from src.llm import LLMService


from config import dataset_path
from src.api.webscoket import router

# creating endpoints for ingestion and getting query using 
from fastapi import FastAPI,Request,HTTPException
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting SupportDesk...")

    # 1. Initialize embedding model
    initialize_settings()

    # 2. Initialize Qdrant and index
    resources = initialize_qdrant_index(
        url=db_url,
        api_key=db_api_key,
        collection=COLLECTION_NAME,
        timeout=30,
        prefer_grpc=False,
    )

    if LLM_PROVIDER == "ollama":
        model_options = {
            "base_url": OLLAMA_HOST,
            "temperature": 0
        }

    elif LLM_PROVIDER == "openai":
        model_options = {
            "api_key": OPENAI_API_KEY,
            # "temperature": 0
        }

    else:
        raise ValueError(f"Unsupported provider: {LLM_PROVIDER}")


    # 3. Store shared resources
    app.state.index = resources["index"]
    app.state.vector_store = resources["vector_store"]
    app.state.qdrant_client = resources["client"]
    app.state.llm = LLMService(
        provider=LLM_PROVIDER,
        model_name=LLM_MODEL,
        model_options=model_options
    )

    logger.info(
        f"Connected to collection: "
        f"{resources['collection_info']}"
    )

    try:
        yield

    finally:
        app.state.qdrant_client.close()
        logger.info("SupportDesk shutdown completed")



app = FastAPI(lifespan=lifespan)
app.include_router(
    router,
    prefix="/api/v1"
)



@app.post("/ingest")
def ingestion(request: Request):

    try:
        # 1. Read documents
        documents = read_dataset(dataset_path)

        if not documents:
            raise ValueError("No documents found")

        # 2. Split documents
        nodes = split_documents(documents)

        if not nodes:
            raise ValueError("No chunks generated")

        logger.info(f"Documents: {len(documents)}")
        logger.info(f"Chunks: {len(nodes)}")

        # 3. Store nodes in Qdrant
        index = store_nodes_in_index(
            index=request.app.state.index,
            nodes=nodes
        )

        return {
            "status": "success",
            "documents_count": len(documents),
            "chunks_count": len(nodes)
        }

    except Exception:
        logger.exception("Ingestion failed")
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed. Check server logs."
        )