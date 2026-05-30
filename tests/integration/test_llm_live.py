"""
Integration tests for the live LLM connection.
Requires Ollama running with gemma3:4b pulled.

Run with: make test-integration
"""

import pytest
from tests.conftest import requires_ollama
from llm.ollama_client import chat, build_messages, is_ollama_running

pytestmark = [pytest.mark.integration, requires_ollama]


def test_ollama_is_running():
    assert is_ollama_running(), "Ollama server is not reachable"


def test_non_streaming_returns_string():
    messages = build_messages("Say the word OK and nothing else.", context_chunks=[])
    response = chat(messages, stream=False)
    assert isinstance(response, str)
    assert len(response) > 0


def test_streaming_yields_chunks():
    messages = build_messages("Say the word OK and nothing else.", context_chunks=[])
    chunks = list(chat(messages, stream=True))
    assert len(chunks) > 0
    full = "".join(chunks)
    assert len(full) > 0


def test_full_response_is_assembled_from_stream():
    messages = build_messages("Say the word OK and nothing else.", context_chunks=[])
    chunks = list(chat(messages, stream=True))
    full = "".join(chunks)
    assert isinstance(full, str)
    assert len(full.strip()) > 0
