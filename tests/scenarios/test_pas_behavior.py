"""
Scenario eval tests — behavioral assertions against the live model.

These tests check that the model follows PAS structure and stays within
appropriate boundaries (no definitive diagnoses, flags life threats promptly, etc.).

Requires Ollama running with gemma3:4b pulled.
Run with: make test-scenarios
"""

import yaml
import pytest
from pathlib import Path

from llm.ollama_client import build_messages, chat
from tests.conftest import requires_ollama

pytestmark = [pytest.mark.scenario, requires_ollama]

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "scenarios.yaml"


def load_scenarios():
    with open(FIXTURES_PATH) as f:
        data = yaml.safe_load(f)
    return [(s["id"], s) for s in data["scenarios"]]


def get_first_response(opening_line: str) -> str:
    """Send an opening line and return the model's first response (no RAG)."""
    messages = build_messages(
        user_query=opening_line,
        context_chunks=[],  # no RAG — testing model behavior from prompt alone
        history=None,
    )
    return chat(messages, stream=False)


# ---------------------------------------------------------------------------
# Parametrized scenario tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario_id,scenario", load_scenarios())
def test_first_response_asks_about_scene_or_flags_life_threat(scenario_id, scenario):
    """
    For non-life-threatening scenarios: first response must contain scene size-up language.
    For life-threatening scenarios: first response must flag the threat AND/OR contain
    expected terms.
    """
    response = get_first_response(scenario["opening_line"])
    response_lower = response.lower()

    expected_terms = [t.lower() for t in scenario["expected_in_first_response"]]
    matched = [t for t in expected_terms if t in response_lower]

    assert matched, (
        f"[{scenario_id}] Expected at least one of {expected_terms} in first response.\n"
        f"Got: {response[:500]}"
    )


@pytest.mark.parametrize("scenario_id,scenario", load_scenarios())
def test_response_does_not_contain_forbidden_language(scenario_id, scenario):
    """Model must not use overconfident or definitive diagnostic language."""
    response = get_first_response(scenario["opening_line"])
    response_lower = response.lower()

    for forbidden in scenario.get("must_not_contain", []):
        assert forbidden.lower() not in response_lower, (
            f"[{scenario_id}] Found forbidden term '{forbidden}' in response.\n"
            f"Got: {response[:500]}"
        )


@pytest.mark.parametrize("scenario_id,scenario", load_scenarios())
def test_first_response_asks_a_question(scenario_id, scenario):
    """
    For non-life-threatening scenarios, the first response should ask at
    least one question (drives the PAS forward).
    Anaphylaxis is exempt — immediate action is appropriate.
    """
    if scenario_id == "anaphylaxis":
        pytest.skip("Anaphylaxis is an immediate-action scenario — no question required first.")

    response = get_first_response(scenario["opening_line"])
    assert "?" in response, (
        f"[{scenario_id}] First response should ask a question to drive PAS forward.\n"
        f"Got: {response[:500]}"
    )


@pytest.mark.parametrize("scenario_id,scenario", [
    (sid, s) for sid, s in load_scenarios() if s.get("life_threat")
])
def test_life_threat_response_is_prompt(scenario_id, scenario):
    """Life-threat scenarios must get an immediate, actionable first response (not just questions)."""
    response = get_first_response(scenario["opening_line"])
    response_lower = response.lower()

    expected_terms = [t.lower() for t in scenario["expected_in_first_response"]]
    matched = [t for t in expected_terms if t in response_lower]

    assert matched, (
        f"[{scenario_id}] Life-threat scenario: expected immediate mention of one of "
        f"{expected_terms}.\nGot: {response[:500]}"
    )


@pytest.mark.parametrize("scenario_id,scenario", load_scenarios())
def test_response_length_is_reasonable(scenario_id, scenario):
    """Responses should be concise — no walls of text (under ~600 words)."""
    response = get_first_response(scenario["opening_line"])
    word_count = len(response.split())
    assert word_count < 600, (
        f"[{scenario_id}] Response is too long ({word_count} words). "
        f"Model should be concise for field use.\nGot: {response[:500]}"
    )
