"""
Tests for Phase A retrieval interface scaffolding.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval import RetrievalRuntime, RetrievalMode, retrieve


def test_module_retrieve_stub_returns_empty_list():
    passages = retrieve("What is ASNN-Goose?")
    assert passages == []


def test_runtime_modes_share_same_interface():
    offline = RetrievalRuntime(mode=RetrievalMode.OFFLINE_PACK)
    online = RetrievalRuntime(mode=RetrievalMode.ONLINE)

    assert offline.retrieve("query", top_k=3) == []
    assert online.retrieve("query", top_k=3) == []
