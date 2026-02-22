"""
Pydantic schemas for request / response validation.
"""

from typing import List, Optional

from pydantic import BaseModel, field_validator


# ── Ingestion ────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    url: str
    author_filter: Optional[str] = None

    @field_validator("url")
    @classmethod
    def url_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("url must not be empty")
        if not v.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v


class BlogInfo(BaseModel):
    id: str
    url: str
    title: str
    authors: List[str]
    article_count: int
    created_at: Optional[str] = None


class IngestResponse(BlogInfo):
    message: str = "Blog ingested successfully"


# ── Chat ─────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    blog_id: str
    message: str
    author: Optional[str] = None
    history: List[ChatMessage] = []


class Source(BaseModel):
    title: str
    url: str


class ChatResponse(BaseModel):
    response: str
    sources: List[Source] = []
    author: Optional[str] = None


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    model: str
    vector_store: str
    embedding_model: str
