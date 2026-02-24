"""
Retrieval runtime scaffold.

Phase A scope:
- shared `retrieve(query) -> passages` interface
- stub behavior only (returns empty list)
"""
from __future__ import annotations

from typing import List

from .base import RetrievalMode, Passage


class RetrievalRuntime:
    """Unified retrieval runtime for offline-pack and online modes."""

    def __init__(self, mode: RetrievalMode = RetrievalMode.OFFLINE_PACK):
        self.mode = mode
        self.last_query = ""
        self.last_top_k = 0

    def retrieve(self, query: str, top_k: int = 5) -> List[Passage]:
        """
        Retrieve passages for a query.

        Returns an empty list in Phase A scaffolding.
        """
        self.last_query = query
        self.last_top_k = top_k
        return []

    def get_diagnostics(self) -> dict:
        return {
            "mode": self.mode.value,
            "implemented": False,
            "last_query": self.last_query,
            "last_top_k": self.last_top_k,
        }


_DEFAULT_RUNTIME = RetrievalRuntime()


def retrieve(query: str, top_k: int = 5) -> List[Passage]:
    """Module-level retrieval interface used by training/eval code."""
    return _DEFAULT_RUNTIME.retrieve(query=query, top_k=top_k)


def get_runtime_diagnostics() -> dict:
    return _DEFAULT_RUNTIME.get_diagnostics()
