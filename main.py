from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.vector_store import initialize_qdrant_index, store_nodes_in_index
from src.settings import initialize_settings
from src.ingest import read_dataset, split_documents
from src.config import (
    LLM_PROVIDER,
    LLM_MODEL,
    OLLAMA_HOST,
    OPENAI_API_KEY,
    COLLECTION_NAME,
    logger,
    db_url,
    db_api_key,
    dataset_path,
)
from src.llm import LLMService
from src.api.webscoket import router


FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting SupportDesk...")

    initialize_settings()

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
            "temperature": 0,
        }
    elif LLM_PROVIDER == "openai":
        model_options = {
            "api_key": OPENAI_API_KEY,
        }
    else:
        raise ValueError(f"Unsupported provider: {LLM_PROVIDER}")

    app.state.index = resources["index"]
    app.state.vector_store = resources["vector_store"]
    app.state.qdrant_client = resources["client"]
    app.state.llm = LLMService(
        provider=LLM_PROVIDER,
        model_name=LLM_MODEL,
        model_options=model_options,
    )

    logger.info(
        "Connected to collection: %s",
        resources["collection_info"],
    )

    try:
        yield
    finally:
        app.state.qdrant_client.close()
        logger.info("SupportDesk shutdown completed")


app = FastAPI(
    title="SupportDesk",
    description="Evidence-grounded ShopLite policy assistant",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    router,
    prefix="/api/v1",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "SupportDesk",
    }


@app.post("/ingest")
def ingestion(request: Request):

    try:
        documents = read_dataset(dataset_path)

        if not documents:
            raise ValueError("No documents found")

        nodes = split_documents(documents)

        if not nodes:
            raise ValueError("No chunks generated")

        logger.info("Documents: %s", len(documents))
        logger.info("Chunks: %s", len(nodes))

        store_nodes_in_index(
            index=request.app.state.index,
            nodes=nodes,
        )

        return {
            "status": "success",
            "documents_count": len(documents),
            "chunks_count": len(nodes),
        }

    except Exception:
        logger.exception("Ingestion failed")
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed. Check server logs.",
        )


# Serve the LinkedIn demo UI last so /health, /ingest, and /api/* keep priority.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning("Frontend directory missing: %s", FRONTEND_DIR)
