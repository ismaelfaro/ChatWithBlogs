"""
Redis vector store backend (requires Redis Stack or Redis with RediSearch module).

Install Redis Stack: https://redis.io/docs/stack/get-started/install/
Or run: docker run -p 6379:6379 redis/redis-stack-server:latest
"""

from __future__ import annotations

import json
import logging
import struct
from typing import Dict, List, Optional

from .base import Document, SearchResult, VectorStore

logger = logging.getLogger(__name__)


def _floats_to_bytes(floats: List[float]) -> bytes:
    return struct.pack(f"{len(floats)}f", *floats)


class RedisVectorStore(VectorStore):
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        index_name: str = "chatmyideas_idx",
        embedding_dim: int = 384,
    ) -> None:
        try:
            import redis
            from redis.commands.search.field import TextField, VectorField, TagField
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        except ImportError as exc:
            raise RuntimeError(
                "redis package not installed. Run: pip install redis"
            ) from exc

        self._redis = redis.from_url(redis_url, decode_responses=False)
        self._index = index_name
        self._dim = embedding_dim
        self._TextField = TextField
        self._VectorField = VectorField
        self._TagField = TagField
        self._IndexDefinition = IndexDefinition
        self._IndexType = IndexType

        self._ensure_index()
        logger.info("Redis vector store connected at %s (index=%s)", redis_url, index_name)

    def _ensure_index(self) -> None:
        try:
            self._redis.ft(self._index).info()
            logger.debug("Redis index %s already exists", self._index)
        except Exception:
            schema = (
                self._TagField("$.blog_id", as_name="blog_id"),
                self._TagField("$.author", as_name="author"),
                self._TextField("$.text", as_name="text"),
                self._TextField("$.article_title", as_name="article_title"),
                self._TextField("$.article_url", as_name="article_url"),
                self._VectorField(
                    "$.embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self._dim,
                        "DISTANCE_METRIC": "COSINE",
                    },
                    as_name="embedding",
                ),
            )
            self._redis.ft(self._index).create_index(
                schema,
                definition=self._IndexDefinition(
                    prefix=["chatmyideas:doc:"], index_type=self._IndexType.JSON
                ),
            )
            logger.info("Created Redis index %s", self._index)

    # ── VectorStore interface ─────────────────────────────────────────────────

    def add(self, documents: List[Document]) -> None:
        pipe = self._redis.pipeline(transaction=False)
        for doc in documents:
            key = f"chatmyideas:doc:{doc.id}"
            payload = {
                "blog_id": doc.metadata.get("blog_id", ""),
                "author": doc.metadata.get("author", ""),
                "text": doc.text,
                "article_title": doc.metadata.get("article_title", ""),
                "article_url": doc.metadata.get("article_url", ""),
                "embedding": doc.embedding,
            }
            pipe.json().set(key, "$", payload)
        pipe.execute()
        logger.debug("Added %d documents to Redis", len(documents))

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[SearchResult]:
        from redis.commands.search.query import Query

        blog_id = (filter or {}).get("blog_id", "")
        author = (filter or {}).get("author", "")

        filter_parts = []
        if blog_id:
            filter_parts.append(f"@blog_id:{{{blog_id}}}")
        if author:
            safe = author.replace(" ", "\\ ")
            filter_parts.append(f"@author:{{{safe}}}")

        pre_filter = " ".join(filter_parts) if filter_parts else "*"
        vec_bytes = _floats_to_bytes(query_embedding)

        q = (
            Query(f"({pre_filter})=>[KNN {top_k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("text", "article_title", "article_url", "author", "score")
            .dialect(2)
            .paging(0, top_k)
        )
        raw = self._redis.ft(self._index).search(q, query_params={"vec": vec_bytes})

        hits: List[SearchResult] = []
        for doc in raw.docs:
            hits.append(
                SearchResult(
                    id=doc.id,
                    text=getattr(doc, "text", ""),
                    metadata={
                        "blog_id": blog_id,
                        "author": getattr(doc, "author", ""),
                        "article_title": getattr(doc, "article_title", ""),
                        "article_url": getattr(doc, "article_url", ""),
                    },
                    score=1.0 - float(getattr(doc, "score", 1.0)),
                )
            )
        return hits

    def delete_collection(self, blog_id: str) -> None:
        from redis.commands.search.query import Query

        q = Query(f"@blog_id:{{{blog_id}}}").return_fields("id").paging(0, 10000).dialect(2)
        try:
            results = self._redis.ft(self._index).search(q)
            pipe = self._redis.pipeline(transaction=False)
            for doc in results.docs:
                pipe.delete(doc.id)
            pipe.execute()
            logger.info("Deleted %d docs for blog %s", len(results.docs), blog_id)
        except Exception as exc:
            logger.warning("Error deleting blog %s: %s", blog_id, exc)

    def list_collections(self) -> List[str]:
        # Enumerate distinct blog_ids via tag aggregation
        try:
            from redis.commands.search.aggregation import AggregateRequest, Asc
            req = AggregateRequest("*").group_by(["@blog_id"])
            res = self._redis.ft(self._index).aggregate(req)
            return [row[1].decode() if isinstance(row[1], bytes) else row[1] for row in res.rows]
        except Exception:
            return []
