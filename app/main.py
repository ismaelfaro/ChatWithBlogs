"""
ChatMyIdeas — FastAPI application entry point.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.rag import BlogRAG
from config import settings

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_vector_store():
    if settings.vector_store == "redis":
        from app.core.vectorstore.redis_store import RedisVectorStore

        return RedisVectorStore(
            redis_url=settings.redis_url,
            index_name=settings.redis_index_name,
            embedding_dim=settings.embedding_dimension,
        )
    # default: ChromaDB (SQLite)
    from app.core.vectorstore.chroma import ChromaVectorStore

    return ChromaVectorStore(db_path=settings.chroma_db_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ChatMyIdeas (vector_store=%s, model=%s)", settings.vector_store, settings.ollama_model)
    vector_store = _build_vector_store()
    app.state.rag = BlogRAG(vector_store=vector_store)
    yield
    logger.info("Shutting down ChatMyIdeas")


app = FastAPI(
    title="ChatMyIdeas — Blog Digital Twin API",
    description=(
        "Create a conversational digital twin of any blog author "
        "using local LLMs and RAG."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve the web chat UI
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(_static_dir / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )
