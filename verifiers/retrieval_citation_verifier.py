"""
Retrieval citation verifier stub.
"""
from __future__ import annotations

from .base import VerifierResult, make_stub_result


class RetrievalCitationVerifier:
    """Stub verifier for citation correctness against retrieved passages."""

    def verify(self, prompt: str, completion: str) -> VerifierResult:
        return make_stub_result(
            verifier_name="retrieval_citation",
            prompt=prompt,
            completion=completion,
            reason="Phase A stub. Implement citation span matching later.",
        )
