"""
ChromaDB vector store backend (default — uses SQLite, zero extra services).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from .base import Document, SearchResult, VectorStore

logger = logging.getLogger(__name__)

_COLLECTION_PREFIX = "chatmyideas_"


class ChromaVectorStore(VectorStore):
    def __init__(self, db_path: str = "./data/chroma") -> None:
        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB initialised at %s", db_path)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _col(self, blog_id: str):
        name = f"{_COLLECTION_PREFIX}{blog_id}"
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── VectorStore interface ─────────────────────────────────────────────────

    def add(self, documents: List[Document]) -> None:
        if not documents:
            return
        # Group by blog_id
        by_blog: Dict[str, List[Document]] = {}
        for doc in documents:
            bid = doc.metadata.get("blog_id", "default")
            by_blog.setdefault(bid, []).append(doc)

        for blog_id, docs in by_blog.items():
            col = self._col(blog_id)
            col.upsert(
                ids=[d.id for d in docs],
                embeddings=[d.embedding for d in docs],
                documents=[d.text for d in docs],
                metadatas=[d.metadata for d in docs],
            )
        logger.debug("Upserted %d documents", len(documents))

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[SearchResult]:
        blog_id = (filter or {}).get("blog_id", "default")
        where: Optional[Dict] = None

        extra_filters = {k: v for k, v in (filter or {}).items() if k != "blog_id"}
        if extra_filters:
            where = extra_filters

        col = self._col(blog_id)
        n_results = min(top_k, col.count() or 1)

        results = col.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )

        hits: List[SearchResult] = []
        for idx, doc_id in enumerate(results["ids"][0]):
            hits.append(
                SearchResult(
                    id=doc_id,
                    text=results["documents"][0][idx],
                    metadata=results["metadatas"][0][idx],
                    score=1.0 - results["distances"][0][idx],  # cosine similarity
                )
            )
        return hits

    def delete_collection(self, blog_id: str) -> None:
        name = f"{_COLLECTION_PREFIX}{blog_id}"
        try:
            self._client.delete_collection(name)
            logger.info("Deleted collection %s", name)
        except Exception as exc:
            logger.warning("Could not delete collection %s: %s", name, exc)

    def list_collections(self) -> List[str]:
        cols = self._client.list_collections()
        return [
            c.name.removeprefix(_COLLECTION_PREFIX)
            for c in cols
            if c.name.startswith(_COLLECTION_PREFIX)
        ]
