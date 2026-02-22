"""
Abstract base class for vector stores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class VectorStore(ABC):
    """Minimal interface all vector store backends must implement."""

    @abstractmethod
    def add(self, documents: List[Document]) -> None:
        """Persist *documents* in the store."""

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[SearchResult]:
        """Return the *top_k* most similar documents, optionally filtered."""

    @abstractmethod
    def delete_collection(self, blog_id: str) -> None:
        """Remove all documents that belong to *blog_id*."""

    @abstractmethod
    def list_collections(self) -> List[str]:
        """Return all blog_ids that have been indexed."""
