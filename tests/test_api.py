"""
Tests for the REST API (app/api/routes.py).
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        with patch("app.api.routes.check_ollama", return_value=True):
            r = client.get("/api/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "model" in data
        assert "vector_store" in data

    def test_health_ollama_offline(self, client):
        with patch("app.api.routes.check_ollama", return_value=False):
            r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["ollama_available"] is False


# ── Ingest ────────────────────────────────────────────────────────────────────

class TestIngest:
    def _mock_ingest(self, rag, blog_info):
        rag.ingest = MagicMock(return_value=blog_info)
        return rag

    def test_ingest_success(self, client, rag):
        blog_info = {
            "id": "abc123",
            "url": "https://blog.example.com",
            "title": "Example Blog",
            "authors": ["Jane Doe"],
            "article_count": 10,
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        rag.ingest = MagicMock(return_value=blog_info)

        r = client.post("/api/v1/ingest", json={"url": "https://blog.example.com"})
        assert r.status_code == 201
        data = r.json()
        assert data["id"] == "abc123"
        assert data["article_count"] == 10
        assert "Jane Doe" in data["authors"]

    def test_ingest_invalid_url(self, client):
        r = client.post("/api/v1/ingest", json={"url": "not-a-url"})
        assert r.status_code == 422

    def test_ingest_empty_url(self, client):
        r = client.post("/api/v1/ingest", json={"url": ""})
        assert r.status_code == 422

    def test_ingest_no_articles_found(self, client, rag):
        rag.ingest = MagicMock(side_effect=ValueError("No articles found at that URL"))
        r = client.post("/api/v1/ingest", json={"url": "https://blog.example.com"})
        assert r.status_code == 422
        assert "No articles" in r.json()["detail"]

    def test_ingest_server_error(self, client, rag):
        rag.ingest = MagicMock(side_effect=RuntimeError("connection timeout"))
        r = client.post("/api/v1/ingest", json={"url": "https://blog.example.com"})
        assert r.status_code == 500


# ── Blog listing ──────────────────────────────────────────────────────────────

class TestBlogs:
    def _seed(self, rag):
        info = {
            "id": "blog1",
            "url": "https://blog.example.com",
            "title": "Test Blog",
            "authors": ["Alice"],
            "article_count": 3,
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        rag._db.upsert(info)
        return info

    def test_list_blogs_empty(self, client):
        r = client.get("/api/v1/blogs")
        assert r.status_code == 200
        assert r.json() == []

    def test_list_blogs_returns_seeded(self, client, rag):
        self._seed(rag)
        r = client.get("/api/v1/blogs")
        assert r.status_code == 200
        assert len(r.json()) == 1
        assert r.json()[0]["id"] == "blog1"

    def test_get_blog_found(self, client, rag):
        self._seed(rag)
        r = client.get("/api/v1/blogs/blog1")
        assert r.status_code == 200
        assert r.json()["title"] == "Test Blog"

    def test_get_blog_not_found(self, client):
        r = client.get("/api/v1/blogs/nonexistent")
        assert r.status_code == 404

    def test_delete_blog_success(self, client, rag):
        self._seed(rag)
        rag.delete_blog = MagicMock()
        r = client.delete("/api/v1/blogs/blog1")
        assert r.status_code == 204
        rag.delete_blog.assert_called_once_with("blog1")

    def test_delete_blog_not_found(self, client):
        r = client.delete("/api/v1/blogs/ghost")
        assert r.status_code == 404


# ── Chat ──────────────────────────────────────────────────────────────────────

class TestChat:
    def _seed(self, rag):
        info = {
            "id": "blog1",
            "url": "https://blog.example.com",
            "title": "Test Blog",
            "authors": ["Alice"],
            "article_count": 3,
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        rag._db.upsert(info)

    def test_chat_success(self, client, rag):
        self._seed(rag)
        from app.models.schemas import ChatResponse, Source

        mock_response = ChatResponse(
            response="I think AI is fascinating.",
            sources=[Source(title="Post 1", url="https://blog.example.com/1")],
            author="Alice",
        )
        rag.chat = MagicMock(return_value=mock_response)

        r = client.post("/api/v1/chat", json={
            "blog_id": "blog1",
            "message": "What do you think about AI?",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["response"] == "I think AI is fascinating."
        assert len(data["sources"]) == 1

    def test_chat_blog_not_found(self, client):
        r = client.post("/api/v1/chat", json={
            "blog_id": "ghost",
            "message": "Hello?",
        })
        assert r.status_code == 404

    def test_chat_with_history(self, client, rag):
        self._seed(rag)
        from app.models.schemas import ChatResponse

        rag.chat = MagicMock(return_value=ChatResponse(response="Yes, indeed."))

        r = client.post("/api/v1/chat", json={
            "blog_id": "blog1",
            "message": "Can you elaborate?",
            "history": [
                {"role": "user", "content": "Tell me about AI."},
                {"role": "assistant", "content": "AI is transformative."},
            ],
        })
        assert r.status_code == 200
        # Verify history was passed through
        call_kwargs = rag.chat.call_args
        assert len(call_kwargs.kwargs["history"]) == 2

    def test_chat_ollama_error_returns_503(self, client, rag):
        self._seed(rag)
        rag.chat = MagicMock(side_effect=RuntimeError("model not found"))
        r = client.post("/api/v1/chat", json={
            "blog_id": "blog1",
            "message": "hello",
        })
        assert r.status_code == 503
        assert "LLM error" in r.json()["detail"]

    def test_chat_with_author_filter(self, client, rag):
        self._seed(rag)
        from app.models.schemas import ChatResponse

        rag.chat = MagicMock(return_value=ChatResponse(response="Yes.", author="Alice"))

        r = client.post("/api/v1/chat", json={
            "blog_id": "blog1",
            "message": "Hello?",
            "author": "Alice",
        })
        assert r.status_code == 200
        call_kwargs = rag.chat.call_args
        assert call_kwargs.kwargs["author"] == "Alice"
