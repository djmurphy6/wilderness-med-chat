"""
Scenario eval tests — behavioral assertions against the live model.

These tests check that the model follows PAS structure and stays within
appropriate boundaries (no definitive diagnoses, flags life threats promptly, etc.).

Requires Ollama running with gemma3:4b pulled.
Run with: make test-scenarios
"""

from pathlib import Path
from functools import lru_cache

import yaml
import pytest

from llm.ollama_client import build_messages, chat
from tests.conftest import requires_ollama

pytestmark = pytest.mark.scenario

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "scenarios.yaml"

BEHAVIOR_TOPIC_GROUPS = {
    "asks_scene_safety": "scene_safety",
    "asks_primary_survey_before_secondary": "primary_survey",
    "flags_possible_spine_concern": "spine_precautions",
    "asks_avpu_or_mental_status": "mental_status",
    "checks_airway_or_breathing": "airway",
    "prioritizes_epinephrine_if_available": "epinephrine",
    "asks_scene_safety_without_delaying_action": "scene_safety",
    "stops_cold_exposure_first": "exposure_control",
    "asks_mental_status": "mental_status",
    "checks_swallow_before_oral_intake": "safe_oral_intake",
    "flags_hypothermia_concern_without_definitive_diagnosis": "exposure_control",
    "reduces_exertion": "exertion_reduction",
    "positions_for_comfort": "exertion_reduction",
    "considers_aspirin_if_appropriate": "aspirin",
    "asks_about_contraindications_or_history": "cardiac_history",
}

NEGATIVE_BEHAVIOR_TERMS = {
    "does_not_diagnose_definitively": [
        "definitely",
        "clearly is",
        "you have a",
        "diagnosis is",
    ],
    "does_not_realign_closed_extremity_injury": ["realign", "straighten it", "pull it straight"],
    "avoids_oral_intake_with_altered_mental_status": ["give him water", "give water", "drink water"],
    "does_not_recommend_antihistamine_first": ["benadryl first", "antihistamine first"],
    "does_not_wait_and_see": ["wait and see"],
    "avoids_rubbing_or_aggressive_rewarming": ["rub", "fire", "hot bath"],
    "does_not_have_patient_walk_out": ["walk out", "walk him out", "hike out"],
    "does_not_introduce_unrelated_cold_injury": ["frostbite", "hypothermia", "cold injury"],
}

SPECIAL_BEHAVIORS = {
    "starts_with_pas_not_treatment",
    "limits_questions_to_2_or_3",
    "flags_life_threat_immediately",
}

KNOWN_BEHAVIORS = set(BEHAVIOR_TOPIC_GROUPS) | set(NEGATIVE_BEHAVIOR_TERMS) | SPECIAL_BEHAVIORS


def load_scenarios():
    with open(FIXTURES_PATH) as f:
        data = yaml.safe_load(f)
    return [(s["id"], s) for s in data["scenarios"]]


def load_behavior_cases():
    return [
        (scenario_id, scenario, behavior)
        for scenario_id, scenario in load_scenarios()
        for behavior in scenario.get("expected_behaviors", [])
    ]


@lru_cache
def get_first_response(opening_line: str) -> str:
    """Send an opening line and return the model's first response (no RAG)."""
    messages = build_messages(
        user_query=opening_line,
        context_chunks=[],  # no RAG — testing model behavior from prompt alone
        history=None,
    )
    return chat(messages, stream=False)


def any_term_in(terms: list[str], text: str) -> tuple[bool, list[str]]:
    """Return (matched, matched_terms) for a list of terms against lowercased text."""
    text_lower = text.lower()
    matched = [t for t in terms if t.lower() in text_lower]
    return bool(matched), matched


def question_count(text: str) -> int:
    return text.count("?")


def assert_has_topic_group(scenario_id: str, scenario: dict, response: str, group_name: str):
    groups = scenario.get("expected_topic_groups", {})
    assert group_name in groups, (
        f"[{scenario_id}] Behavior requires expected_topic_groups.{group_name}, "
        "but the scenario fixture does not define it."
    )

    matched, found = any_term_in(groups[group_name], response)
    assert matched, (
        f"[{scenario_id}] Expected topic group '{group_name}' with one of "
        f"{groups[group_name]}.\nGot: {response[:500]}"
    )


def assert_avoids_terms(scenario_id: str, behavior: str, response: str, terms: list[str]):
    response_lower = response.lower()
    for term in terms:
        assert term.lower() not in response_lower, (
            f"[{scenario_id}] Behavior '{behavior}' failed: found forbidden term "
            f"'{term}'.\nGot: {response[:500]}"
        )


def assert_behavior(scenario_id: str, scenario: dict, behavior: str, response: str):
    if behavior in BEHAVIOR_TOPIC_GROUPS:
        assert_has_topic_group(scenario_id, scenario, response, BEHAVIOR_TOPIC_GROUPS[behavior])
        return

    if behavior in NEGATIVE_BEHAVIOR_TERMS:
        terms = NEGATIVE_BEHAVIOR_TERMS[behavior] + scenario.get("must_not_contain", [])
        assert_avoids_terms(scenario_id, behavior, response, terms)
        return

    if behavior == "limits_questions_to_2_or_3":
        count = question_count(response)
        assert 1 <= count <= 3, (
            f"[{scenario_id}] Expected 1–3 questions, got {count}.\n"
            f"Got: {response[:500]}"
        )
        return

    if behavior == "flags_life_threat_immediately":
        expected_terms = scenario.get("expected_in_first_response", [])
        matched, found = any_term_in(expected_terms, response)
        assert matched, (
            f"[{scenario_id}] Expected immediate life-threat/action language with one of "
            f"{expected_terms}.\nGot: {response[:500]}"
        )
        return

    if behavior == "starts_with_pas_not_treatment":
        assert_has_topic_group(scenario_id, scenario, response, "scene_safety")
        assert_avoids_terms(
            scenario_id,
            behavior,
            response,
            ["splint", "wrap", "ibuprofen", "walk it off"],
        )
        return

    raise AssertionError(f"[{scenario_id}] Unknown behavior label: {behavior}")


# ---------------------------------------------------------------------------
# Static fixture validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_id,scenario", load_scenarios())
def test_expected_behavior_labels_are_known(scenario_id, scenario):
    """YAML behavior labels should be explicit so typoed expectations fail fast."""
    unknown = sorted(set(scenario.get("expected_behaviors", [])) - KNOWN_BEHAVIORS)
    assert not unknown, f"[{scenario_id}] Unknown expected_behaviors labels: {unknown}"


@pytest.mark.parametrize("scenario_id,scenario,behavior", load_behavior_cases())
def test_behavior_topic_groups_are_defined(scenario_id, scenario, behavior):
    """Fixture labels that rely on synonym groups must define those groups locally."""
    group_name = BEHAVIOR_TOPIC_GROUPS.get(behavior)
    if not group_name:
        return
    assert group_name in scenario.get("expected_topic_groups", {}), (
        f"[{scenario_id}] Behavior '{behavior}' requires expected_topic_groups.{group_name}"
    )


# ---------------------------------------------------------------------------
# Parametrized live scenario tests
# ---------------------------------------------------------------------------


@requires_ollama
@pytest.mark.parametrize("scenario_id,scenario,behavior", load_behavior_cases())
def test_first_response_matches_expected_behavior(scenario_id, scenario, behavior):
    """Each YAML behavior label maps to a concrete first-response assertion."""
    response = get_first_response(scenario["opening_line"])
    assert_behavior(scenario_id, scenario, behavior, response)


@requires_ollama
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


@requires_ollama
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


@requires_ollama
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


@requires_ollama
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


@requires_ollama
@pytest.mark.parametrize("scenario_id,scenario", load_scenarios())
def test_response_length_is_reasonable(scenario_id, scenario):
    """Responses should be concise — no walls of text (under ~600 words)."""
    response = get_first_response(scenario["opening_line"])
    word_count = len(response.split())
    assert word_count < 600, (
        f"[{scenario_id}] Response is too long ({word_count} words). "
        f"Model should be concise for field use.\nGot: {response[:500]}"
    )


# ---------------------------------------------------------------------------
# Multi-turn: check that specific assessments happen before specific actions
# These use the scenario's must_assess_before_* / must_ask_or_do keys.
# ---------------------------------------------------------------------------

@lru_cache
def run_two_turn(opening_line: str, followup: str) -> str:
    """Run two turns and return the second response."""
    messages = build_messages(opening_line, context_chunks=[], history=None)
    first = chat(messages, stream=False)
    history = [
        {"role": "user", "content": opening_line},
        {"role": "assistant", "content": first},
    ]
    messages2 = build_messages(followup, context_chunks=[], history=history)
    return chat(messages2, stream=False)


@requires_ollama
@pytest.mark.parametrize("scenario_id,scenario", [
    (sid, s) for sid, s in load_scenarios() if s.get("must_assess_before_splinting")
])
def test_csm_assessed_before_splinting(scenario_id, scenario):
    """For extremity injuries, model must ask about CSM before recommending splinting."""
    # Give the model the opening + confirm scene is safe, then see what it asks next
    second_response = run_two_turn(
        scenario["opening_line"],
        followup="Scene is safe, she is alert and oriented.",
    )
    terms = scenario["must_assess_before_splinting"]
    matched, found = any_term_in(terms, second_response)
    assert matched, (
        f"[{scenario_id}] Expected CSM check ({terms}) before splinting guidance.\n"
        f"Got: {second_response[:500]}"
    )


@requires_ollama
@pytest.mark.parametrize("scenario_id,scenario", [
    (sid, s) for sid, s in load_scenarios() if s.get("must_assess_before_calories")
])
def test_swallow_assessed_before_oral_intake(scenario_id, scenario):
    """For hypothermia, model must verify ability to swallow before recommending hot liquids."""
    second_response = run_two_turn(
        scenario["opening_line"],
        followup="We've gotten them out of the rain and removed wet clothes.",
    )
    terms = scenario["must_assess_before_calories"]
    matched, found = any_term_in(terms, second_response)
    assert matched, (
        f"[{scenario_id}] Expected swallow/consciousness check ({terms}) before "
        f"recommending oral calories.\nGot: {second_response[:500]}"
    )
