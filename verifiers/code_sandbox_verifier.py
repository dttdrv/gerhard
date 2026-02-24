"""
Code sandbox verifier stub.
"""
from __future__ import annotations

from .base import VerifierResult, make_stub_result


class CodeSandboxVerifier:
    """Stub verifier for execution-based coding rewards."""

    def verify(self, prompt: str, completion: str) -> VerifierResult:
        return make_stub_result(
            verifier_name="code_sandbox",
            prompt=prompt,
            completion=completion,
            reason="Phase A stub. Implement sandboxed test execution later.",
        )
