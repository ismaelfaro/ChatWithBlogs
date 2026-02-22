"""
Tests for the RAG pipeline (app/core/rag.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.rag import BlogMetadataDB, BlogRAG, _chunk_text, _url_id
from app.core.vectorstore.base import Document, SearchResult


# ── Chunker ───────────────────────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        chunks = _chunk_text("hello world", size=100, overlap=10)
        assert chunks == ["hello world"]

    def test_long_text_is_split(self):
        words = ["word"] * 200
        text = " ".join(words)
        chunks = _chunk_text(text, size=50, overlap=10)
        assert len(chunks) > 1

    def test_overlap_is_honoured(self):
        words = [f"w{i}" for i in range(100)]
        text = " ".join(words)
        chunks = _chunk_text(text, size=20, overlap=5)
        # The start of chunk[1] should overlap with the end of chunk[0]
        end_chunk0   = chunks[0].split()[-5:]
        start_chunk1 = chunks[1].split()[:5]
        assert end_chunk0 == start_chunk1

    def test_empty_text_returns_empty_list(self):
        chunks = _chunk_text("", size=50, overlap=10)
        assert chunks == []


# ── URL ID ────────────────────────────────────────────────────────────────────

class TestUrlId:
    def test_same_url_same_id(self):
        assert _url_id("https://example.com/blog") == _url_id("https://example.com/blog")

    def test_trailing_slash_stripped(self):
        assert _url_id("https://example.com/blog/") == _url_id("https://example.com/blog")

    def test_different_urls_different_ids(self):
        assert _url_id("https://a.com") != _url_id("https://b.com")


# ── BlogMetadataDB ────────────────────────────────────────────────────────────

class TestBlogMetadataDB:
    def test_upsert_and_get(self, tmp_path):
        db = BlogMetadataDB(db_path=str(tmp_path / "test.db"))
        info = {
            "id": "abc123",
            "url": "https://example.com",
            "title": "My Blog",
            "authors": ["Alice"],
            "article_count": 5,
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        db.upsert(info)
        result = db.get("abc123")
        assert result is not None
        assert result["title"] == "My Blog"
        assert result["authors"] == ["Alice"]

    def test_list_all_returns_rows(self, tmp_path):
        db = BlogMetadataDB(db_path=str(tmp_path / "test.db"))
        for i in range(3):
            db.upsert({"id": f"id{i}", "url": f"https://e{i}.com", "title": f"Blog {i}",
                        "authors": [], "article_count": i})
        rows = db.list_all()
        assert len(rows) == 3

    def test_delete_removes_row(self, tmp_path):
        db = BlogMetadataDB(db_path=str(tmp_path / "test.db"))
        db.upsert({"id": "del1", "url": "https://x.com", "title": "X",
                    "authors": [], "article_count": 1})
        db.delete("del1")
        assert db.get("del1") is None

    def test_missing_id_returns_none(self, tmp_path):
        db = BlogMetadataDB(db_path=str(tmp_path / "test.db"))
        assert db.get("nonexistent") is None


# ── BlogRAG.ingest ────────────────────────────────────────────────────────────

class TestBlogRAGIngest:
    def test_ingest_stores_chunks(self, rag, sample_articles):
        with patch("app.core.rag.ingest_blog", return_value=sample_articles):
            with patch("app.core.rag.detect_blog_title", return_value="Test Blog"):
                info = rag.ingest("https://blog.example.com")

        assert info["article_count"] == len(sample_articles)
        assert "Jane Doe" in info["authors"]
        # Vector store should have documents
        assert len(rag._store._docs) > 0

    def test_ingest_raises_on_empty(self, rag):
        with patch("app.core.rag.ingest_blog", return_value=[]):
            with pytest.raises(ValueError, match="No articles found"):
                rag.ingest("https://empty.example.com")

    def test_ingest_author_filter(self, rag, sample_articles):
        with patch("app.core.rag.ingest_blog", return_value=sample_articles):
            with patch("app.core.rag.detect_blog_title", return_value="Blog"):
                info = rag.ingest("https://blog.example.com", author_filter="Jane Doe")
        assert info["article_count"] == 2

    def test_ingest_author_filter_no_match(self, rag, sample_articles):
        with patch("app.core.rag.ingest_blog", return_value=sample_articles):
            with pytest.raises(ValueError, match="No articles found for author"):
                rag.ingest("https://blog.example.com", author_filter="Unknown Person")

    def test_ingest_persists_metadata(self, rag, sample_articles, tmp_path):
        with patch("app.core.rag.ingest_blog", return_value=sample_articles):
            with patch("app.core.rag.detect_blog_title", return_value="Persisted"):
                info = rag.ingest("https://persist.example.com")

        fetched = rag._db.get(info["id"])
        assert fetched is not None
        assert fetched["title"] == "Persisted"


# ── BlogRAG.chat ──────────────────────────────────────────────────────────────

class TestBlogRAGChat:
    def _setup_blog(self, rag, sample_articles):
        with patch("app.core.rag.ingest_blog", return_value=sample_articles):
            with patch("app.core.rag.detect_blog_title", return_value="Test Blog"):
                return rag.ingest("https://blog.example.com")

    def test_chat_returns_response(self, rag, sample_articles):
        info = self._setup_blog(rag, sample_articles)

        mock_ollama_response = MagicMock()
        mock_ollama_response.message.content = "I believe in open source."

        with patch("app.core.rag.ollama.Client") as MockClient:
            MockClient.return_value.chat.return_value = mock_ollama_response
            result = rag.chat(blog_id=info["id"], message="What do you believe in?")

        assert result.response == "I believe in open source."
        assert isinstance(result.sources, list)

    def test_chat_no_hits_returns_fallback(self, rag, sample_articles, vector_store):
        info = self._setup_blog(rag, sample_articles)
        # Clear the store so no hits are returned
        vector_store._docs.clear()

        result = rag.chat(blog_id=info["id"], message="anything")
        assert "couldn't find" in result.response.lower()

    def test_chat_sources_are_unique(self, rag, sample_articles):
        info = self._setup_blog(rag, sample_articles)

        mock_ollama_response = MagicMock()
        mock_ollama_response.message.content = "Sure thing."

        with patch("app.core.rag.ollama.Client") as MockClient:
            MockClient.return_value.chat.return_value = mock_ollama_response
            result = rag.chat(blog_id=info["id"], message="Tell me more.")

        urls = [s.url for s in result.sources]
        assert len(urls) == len(set(urls))


# ── BlogRAG.delete_blog ───────────────────────────────────────────────────────

class TestBlogRAGDelete:
    def test_delete_removes_metadata_and_vectors(self, rag, sample_articles):
        with patch("app.core.rag.ingest_blog", return_value=sample_articles):
            with patch("app.core.rag.detect_blog_title", return_value="Blog"):
                info = rag.ingest("https://blog.example.com")

        rag.delete_blog(info["id"])
        assert rag._db.get(info["id"]) is None
        assert all(
            d.metadata.get("blog_id") != info["id"]
            for d in rag._store._docs.values()
        )
