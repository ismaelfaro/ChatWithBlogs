"""
RAG pipeline: ingestion, chunking, embedding, retrieval, and generation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama
from sentence_transformers import SentenceTransformer

from config import settings
from app.core.ingestion import Article, detect_blog_title, ingest_blog
from app.core.vectorstore.base import Document, VectorStore
from app.models.schemas import BlogInfo, ChatMessage, ChatResponse, Source

logger = logging.getLogger(__name__)

# ── Text chunker ──────────────────────────────────────────────────────────────


def _chunk_text(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    """Split *text* into overlapping chunks of approximately *size* words."""
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return chunks


# ── Blog metadata store (SQLite) ──────────────────────────────────────────────


class BlogMetadataDB:
    """Thin SQLite wrapper to persist blog-level metadata across restarts."""

    def __init__(self, db_path: str = "./data/blogs.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init()

    def _init(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blogs (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                authors TEXT NOT NULL,
                article_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def upsert(self, info: Dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO blogs (id, url, title, authors, article_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                info["id"],
                info["url"],
                info["title"],
                json.dumps(info["authors"]),
                info["article_count"],
                info.get("created_at", _now()),
            ),
        )
        self._conn.commit()

    def get(self, blog_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT id, url, title, authors, article_count, created_at FROM blogs WHERE id=?",
            (blog_id,),
        ).fetchone()
        return _row_to_dict(row) if row else None

    def list_all(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, url, title, authors, article_count, created_at FROM blogs ORDER BY created_at DESC"
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def delete(self, blog_id: str) -> None:
        self._conn.execute("DELETE FROM blogs WHERE id=?", (blog_id,))
        self._conn.commit()


def _row_to_dict(row) -> Dict[str, Any]:
    return {
        "id": row[0],
        "url": row[1],
        "title": row[2],
        "authors": json.loads(row[3]),
        "article_count": row[4],
        "created_at": row[5],
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Digital-twin system prompt ─────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a digital twin of {author_name}.
    You embody the knowledge, writing style, perspectives, and ideas expressed in their published articles.

    When answering:
    - Draw exclusively from the ideas in the provided article excerpts below.
    - Maintain the author's characteristic voice and intellectual tone.
    - If the question goes beyond the provided articles, acknowledge the limitation
      while staying in character and offering what the articles do cover.
    - Never fabricate facts or opinions not supported by the excerpts.
    - Cite article titles naturally in your answers when relevant.

    --- Article excerpts ---
    {context}
    --- End of excerpts ---
    """
)

# ── Main RAG class ────────────────────────────────────────────────────────────


class BlogRAG:
    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store
        self._db = BlogMetadataDB()
        logger.info("Loading embedding model: %s", settings.embedding_model)
        self._embedder = SentenceTransformer(settings.embedding_model)
        logger.info("BlogRAG ready (model=%s, store=%s)", settings.ollama_model, settings.vector_store)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, url: str, author_filter: Optional[str] = None) -> Dict[str, Any]:
        blog_id = _url_id(url)

        logger.info("Ingesting blog %s (id=%s)", url, blog_id)
        articles = ingest_blog(url)
        if not articles:
            raise ValueError(f"No articles found at {url}")

        if author_filter:
            articles = [a for a in articles if a.author == author_filter]
            if not articles:
                raise ValueError(f"No articles found for author '{author_filter}'")

        # Chunk + embed
        docs: List[Document] = []
        for article in articles:
            chunks = _chunk_text(article.content, settings.chunk_size, settings.chunk_overlap)
            texts = chunks
            embeddings = self._embedder.encode(texts, show_progress_bar=False).tolist()
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                doc_id = f"{article.id}_{i}"
                docs.append(
                    Document(
                        id=doc_id,
                        text=chunk,
                        embedding=emb,
                        metadata={
                            "blog_id": blog_id,
                            "article_url": article.url,
                            "article_title": article.title,
                            "author": article.author or "Unknown",
                        },
                    )
                )

        self._store.add(docs)

        authors = sorted({a.author for a in articles if a.author})
        title = detect_blog_title(url)

        info = {
            "id": blog_id,
            "url": url,
            "title": title,
            "authors": authors,
            "article_count": len(articles),
            "created_at": _now(),
        }
        self._db.upsert(info)
        logger.info(
            "Ingested %d articles, %d chunks for blog %s", len(articles), len(docs), blog_id
        )
        return info

    # ── Chat ──────────────────────────────────────────────────────────────────

    def chat(
        self,
        blog_id: str,
        message: str,
        author: Optional[str] = None,
        history: Optional[List[ChatMessage]] = None,
    ) -> ChatResponse:
        history = history or []

        # 1. Embed the user query
        query_emb = self._embedder.encode([message], show_progress_bar=False)[0].tolist()

        # 2. Retrieve relevant chunks
        f: Dict[str, str] = {"blog_id": blog_id}
        if author:
            f["author"] = author
        hits = self._store.search(query_emb, top_k=settings.top_k_chunks, filter=f)

        if not hits:
            return ChatResponse(
                response="I couldn't find relevant articles to answer your question.",
                sources=[],
                author=author,
            )

        # 3. Build context
        context = "\n\n---\n\n".join(h.text for h in hits)
        seen: set[str] = set()
        sources: List[Source] = []
        for h in hits:
            url = h.metadata.get("article_url", "")
            if url and url not in seen:
                seen.add(url)
                sources.append(
                    Source(
                        title=h.metadata.get("article_title", url),
                        url=url,
                    )
                )

        # 4. Build messages
        author_name = author or _get_blog_author(self._db, blog_id)
        system_content = _SYSTEM_PROMPT.format(author_name=author_name, context=context)

        messages = [{"role": "system", "content": system_content}]
        for h in history:
            messages.append({"role": h.role, "content": h.content})
        messages.append({"role": "user", "content": message})

        # 5. Generate via Ollama
        client = ollama.Client(host=settings.ollama_host)
        response = client.chat(
            model=settings.ollama_model,
            messages=messages,
            options={"num_predict": 1024},
        )
        answer = response.message.content.strip()

        return ChatResponse(response=answer, sources=sources, author=author)

    # ── Metadata helpers ──────────────────────────────────────────────────────

    def list_blogs(self) -> List[Dict[str, Any]]:
        return self._db.list_all()

    def get_blog(self, blog_id: str) -> Optional[Dict[str, Any]]:
        return self._db.get(blog_id)

    def delete_blog(self, blog_id: str) -> None:
        self._store.delete_collection(blog_id)
        self._db.delete(blog_id)


# ── Utilities ─────────────────────────────────────────────────────────────────


def _url_id(url: str) -> str:
    return hashlib.md5(url.strip().rstrip("/").encode()).hexdigest()


def _get_blog_author(db: BlogMetadataDB, blog_id: str) -> str:
    info = db.get(blog_id)
    if info and info["authors"]:
        return info["authors"][0]
    return "the blog author"


def check_ollama() -> bool:
    """Return True if Ollama is reachable and the configured model exists."""
    try:
        client = ollama.Client(host=settings.ollama_host)
        models = client.list()
        names = [m.model for m in models.models]
        return any(settings.ollama_model in n for n in names)
    except Exception:
        return False
