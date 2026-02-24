"""
Verifier interface for RLVR scaffolding.

Phase A scope: interface + deterministic stubs only.
"""
from __future__ import annotations

from typing import Any, Dict, Protocol


VerifierResult = Dict[str, Any]


class Verifier(Protocol):
    """Minimal verifier contract used by RLVR pipelines."""

    def verify(self, prompt: str, completion: str) -> VerifierResult:
        """
        Return a verifier result in the format:
        {"reward": float, "meta": dict}
        """
        ...


def make_stub_result(
    verifier_name: str,
    prompt: str,
    completion: str,
    reason: str,
) -> VerifierResult:
    """Return a consistent placeholder response for stub verifiers."""
    return {
        "reward": 0.0,
        "meta": {
            "verifier": verifier_name,
            "implemented": False,
            "reason": reason,
            "prompt_chars": len(prompt),
            "completion_chars": len(completion),
        },
    }
