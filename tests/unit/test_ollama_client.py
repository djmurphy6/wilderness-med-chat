"""
Unit tests for llm/ollama_client.py.
No Ollama connection required — tests only deterministic logic.
"""

import pytest
from llm.ollama_client import build_messages, SYSTEM_PROMPT


pytestmark = pytest.mark.unit


class TestBuildMessages:
    def test_always_starts_with_system_message(self, sample_context_chunks):
        messages = build_messages("test query", sample_context_chunks)
        assert messages[0]["role"] == "system"

    def test_last_message_is_user_query(self, sample_context_chunks):
        query = "What do I do for a suspected spinal injury?"
        messages = build_messages(query, sample_context_chunks)
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == query

    def test_context_chunks_injected_into_system_message(self, sample_context_chunks):
        messages = build_messages("any query", sample_context_chunks)
        system_content = messages[0]["content"]
        for chunk in sample_context_chunks:
            assert chunk in system_content

    def test_empty_context_still_produces_valid_messages(self, empty_context_chunks):
        messages = build_messages("any query", empty_context_chunks)
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert "No relevant context found" in messages[0]["content"]

    def test_history_is_inserted_between_system_and_current_query(self, sample_context_chunks, sample_history):
        messages = build_messages("new question", sample_context_chunks, history=sample_history)
        roles = [m["role"] for m in messages]
        assert roles[0] == "system"
        assert roles[-1] == "user"
        assert messages[-1]["content"] == "new question"
        # History turns should appear in the middle
        history_roles = roles[1:-1]
        assert history_roles == [m["role"] for m in sample_history]

    def test_no_history_produces_only_system_and_user(self, sample_context_chunks):
        messages = build_messages("solo query", sample_context_chunks, history=None)
        assert len(messages) == 2

    def test_context_chunks_separated_by_delimiter(self, sample_context_chunks):
        messages = build_messages("any query", sample_context_chunks)
        system_content = messages[0]["content"]
        assert "---" in system_content


class TestSystemPrompt:
    def test_pas_steps_present(self):
        """System prompt must reference each PAS step by name (case-insensitive)."""
        required_terms = [
            "scene size-up",
            "primary survey",
            "sample",
            "vital signs",
            "opqrst",
            "avpu",
        ]
        prompt_lower = SYSTEM_PROMPT.lower()
        for term in required_terms:
            assert term in prompt_lower, f"'{term}' missing from system prompt"

    def test_evacuation_mentioned(self):
        assert "evacuate" in SYSTEM_PROMPT.lower() or "evacuation" in SYSTEM_PROMPT.lower()

    def test_no_definitive_diagnosis_instruction(self):
        assert "diagnose definitively" in SYSTEM_PROMPT.lower() or "never diagnose" in SYSTEM_PROMPT.lower()

    def test_conciseness_instruction(self):
        """Model should be instructed to be brief."""
        assert "concise" in SYSTEM_PROMPT.lower() or "short" in SYSTEM_PROMPT.lower()

    def test_aid_not_decision_maker(self):
        assert "aid" in SYSTEM_PROMPT.lower()
        assert "decision" in SYSTEM_PROMPT.lower()
