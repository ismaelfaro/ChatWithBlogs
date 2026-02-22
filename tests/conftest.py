"""
Shared pytest fixtures.
"""

from __future__ import annotations

import os
import sys
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Make sure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("VECTOR_STORE", "chroma")
os.environ.setdefault("CHROMA_DB_PATH", "/tmp/chatmyideas_test_chroma")
os.environ.setdefault("OLLAMA_MODEL", "llama3.2")


# ── In-memory vector store for tests ─────────────────────────────────────────

from app.core.vectorstore.base import Document, SearchResult, VectorStore


class InMemoryVectorStore(VectorStore):
    def __init__(self):
        self._docs: dict[str, Document] = {}

    def add(self, documents):
        for d in documents:
            self._docs[d.id] = d

    def search(self, query_embedding, top_k=5, filter=None):
        blog_id = (filter or {}).get("blog_id")
        author  = (filter or {}).get("author")
        results = []
        for doc in self._docs.values():
            if blog_id and doc.metadata.get("blog_id") != blog_id:
                continue
            if author and doc.metadata.get("author") != author:
                continue
            results.append(
                SearchResult(
                    id=doc.id,
                    text=doc.text,
                    metadata=doc.metadata,
                    score=0.9,
                )
            )
        return results[:top_k]

    def delete_collection(self, blog_id):
        to_del = [k for k, v in self._docs.items() if v.metadata.get("blog_id") == blog_id]
        for k in to_del:
            del self._docs[k]

    def list_collections(self):
        return list({v.metadata.get("blog_id") for v in self._docs.values()})


# ── Mock embedder ─────────────────────────────────────────────────────────────

import numpy as np


class MockEmbedder:
    def encode(self, texts, **kwargs):
        return np.random.rand(len(texts), 384).astype("float32")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def rag(vector_store, tmp_path):
    """Return a BlogRAG instance wired to in-memory store and mock embedder."""
    from app.core.rag import BlogMetadataDB, BlogRAG

    db_path = str(tmp_path / "blogs.db")
    # Bypass __init__ to avoid loading the real SentenceTransformer model
    instance = BlogRAG.__new__(BlogRAG)
    instance._store = vector_store
    instance._db = BlogMetadataDB(db_path=db_path)
    instance._embedder = MockEmbedder()
    return instance


@pytest.fixture
def client(rag) -> Generator:
    """FastAPI test client with mocked RAG.

    Uses dependency_overrides so the lifespan's BlogRAG instance never
    reaches the route handlers during tests.
    """
    from app.main import app
    from app.api.routes import _rag

    # Inject test rag via FastAPI dependency override (survives lifespan)
    app.dependency_overrides[_rag] = lambda: rag

    # Prevent lifespan from loading real SentenceTransformer / ChromaDB
    with patch("app.main.BlogRAG") as MockRAG, \
         patch("app.main._build_vector_store", return_value=MagicMock()):
        MockRAG.return_value = MagicMock()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    del app.dependency_overrides[_rag]


@pytest.fixture
def sample_articles():
    from app.core.ingestion import Article

    return [
        Article(
            url="https://blog.example.com/post/1",
            title="My First Post",
            content="This is the content of my first post. " * 30,
            author="Jane Doe",
            published="2024-01-01",
        ),
        Article(
            url="https://blog.example.com/post/2",
            title="Thoughts on AI",
            content="Artificial intelligence is changing the world. " * 30,
            author="Jane Doe",
            published="2024-02-01",
        ),
    ]
