"""
Integration tests for the RAG pipeline.
Requires Ollama running (for embeddings) and at least one PDF ingested.

Run with: make test-integration
"""

import pytest
from tests.conftest import requires_ollama
from rag.query import rag_engine

pytestmark = [pytest.mark.integration, requires_ollama]


class TestRAGEngine:
    def test_retrieve_returns_list(self):
        results = rag_engine.retrieve("how do I treat a sprained ankle")
        assert isinstance(results, list)

    def test_retrieve_returns_strings(self):
        results = rag_engine.retrieve("patient assessment")
        for chunk in results:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_retrieve_respects_top_k(self):
        results = rag_engine.retrieve("hypothermia treatment", top_k=3)
        assert len(results) <= 3

    def test_retrieve_wilderness_medicine_query_returns_relevant_chunks(self):
        """Smoke test: a medical query should return chunks with medical language."""
        results = rag_engine.retrieve("signs of shock in the wilderness")
        assert len(results) > 0
        combined = " ".join(results).lower()
        # At least one of these should appear in retrieved medical text
        medical_terms = ["patient", "pulse", "blood", "shock", "skin", "breathing", "heart"]
        assert any(term in combined for term in medical_terms), (
            f"Retrieved chunks don't appear to contain medical content: {combined[:300]}"
        )

    def test_is_empty_returns_bool(self):
        result = rag_engine.is_empty()
        assert isinstance(result, bool)
