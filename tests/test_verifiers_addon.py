"""
Tests for Phase A verifier interface stubs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from verifiers import (
    PythonMathVerifier,
    CodeSandboxVerifier,
    RetrievalCitationVerifier,
)


def _assert_verify_contract(result):
    assert "reward" in result
    assert "meta" in result
    assert isinstance(result["reward"], float)
    assert isinstance(result["meta"], dict)


def test_python_math_verifier_stub_contract():
    verifier = PythonMathVerifier()
    result = verifier.verify("Compute 2+2", "4")

    _assert_verify_contract(result)
    assert result["meta"]["implemented"] is False
    assert result["meta"]["verifier"] == "python_math"


def test_code_sandbox_verifier_stub_contract():
    verifier = CodeSandboxVerifier()
    result = verifier.verify("Write a function", "def add(a,b): return a+b")

    _assert_verify_contract(result)
    assert result["meta"]["implemented"] is False
    assert result["meta"]["verifier"] == "code_sandbox"


def test_retrieval_citation_verifier_stub_contract():
    verifier = RetrievalCitationVerifier()
    result = verifier.verify("Answer with citations", "[1] source text")

    _assert_verify_contract(result)
    assert result["meta"]["implemented"] is False
    assert result["meta"]["verifier"] == "retrieval_citation"
