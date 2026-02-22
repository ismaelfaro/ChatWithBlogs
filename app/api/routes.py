"""
FastAPI router — all /api/v1/* endpoints.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.rag import BlogRAG, check_ollama
from app.models.schemas import (
    BlogInfo,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


def _rag(request: Request) -> BlogRAG:
    return request.app.state.rag


# ── Health ────────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(rag: BlogRAG = Depends(_rag)) -> HealthResponse:
    ollama_ok = check_ollama()
    return HealthResponse(
        status="ok",
        ollama_available=ollama_ok,
        model=settings.ollama_model,
        vector_store=settings.vector_store,
        embedding_model=settings.embedding_model,
    )


# ── Ingestion ─────────────────────────────────────────────────────────────────


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["ingestion"],
)
async def ingest(body: IngestRequest, rag: BlogRAG = Depends(_rag)) -> IngestResponse:
    """Fetch all articles from *url* and build the digital twin index."""
    try:
        info = rag.ingest(body.url, author_filter=body.author_filter)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("Ingestion failed for %s", body.url)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )
    return IngestResponse(**info)


# ── Blog listing ──────────────────────────────────────────────────────────────


@router.get("/blogs", response_model=List[BlogInfo], tags=["blogs"])
async def list_blogs(rag: BlogRAG = Depends(_rag)) -> List[BlogInfo]:
    return [BlogInfo(**b) for b in rag.list_blogs()]


@router.get("/blogs/{blog_id}", response_model=BlogInfo, tags=["blogs"])
async def get_blog(blog_id: str, rag: BlogRAG = Depends(_rag)) -> BlogInfo:
    info = rag.get_blog(blog_id)
    if not info:
        raise HTTPException(status_code=404, detail="Blog not found")
    return BlogInfo(**info)


@router.delete("/blogs/{blog_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["blogs"])
async def delete_blog(blog_id: str, rag: BlogRAG = Depends(_rag)) -> None:
    info = rag.get_blog(blog_id)
    if not info:
        raise HTTPException(status_code=404, detail="Blog not found")
    rag.delete_blog(blog_id)


# ── Chat ──────────────────────────────────────────────────────────────────────


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(body: ChatRequest, rag: BlogRAG = Depends(_rag)) -> ChatResponse:
    """Send a message to the digital twin of a blog's author."""
    if not rag.get_blog(body.blog_id):
        raise HTTPException(status_code=404, detail="Blog not found. Ingest it first.")

    try:
        result = rag.chat(
            blog_id=body.blog_id,
            message=body.message,
            author=body.author,
            history=body.history,
        )
    except Exception as exc:
        logger.exception("Chat failed for blog %s", body.blog_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM error: {exc}. Is Ollama running with model '{settings.ollama_model}'?",
        )
    return result
