"""
Retrieval interfaces and shared data structures.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RetrievalMode(str, Enum):
    OFFLINE_PACK = "offline_pack"
    ONLINE = "online"


@dataclass
class Passage:
    """Single retrieved passage returned by retrieval backends."""

    passage_id: str
    text: str
    source: str
    score: Optional[float] = None
