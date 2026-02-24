"""
Python math verifier stub.
"""
from __future__ import annotations

from .base import VerifierResult, make_stub_result


class PythonMathVerifier:
    """Stub verifier for deterministic math/science checks."""

    def verify(self, prompt: str, completion: str) -> VerifierResult:
        return make_stub_result(
            verifier_name="python_math",
            prompt=prompt,
            completion=completion,
            reason="Phase A stub. Implement symbolic/numeric equivalence later.",
        )
