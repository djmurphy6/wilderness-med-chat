"""
Shared fixtures and helpers for all test tiers.
"""

import pytest
import ollama


def ollama_available() -> bool:
    try:
        models = ollama.list()
        names = [m.model for m in models.models]
        return any("gemma3" in n for n in names)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Markers / skip guards
# ---------------------------------------------------------------------------

requires_ollama = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama is not running or gemma3:4b is not pulled yet.",
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_context_chunks() -> list[str]:
    """Realistic-looking fake retrieval chunks for unit tests."""
    return [
        "The primary survey follows the ABCDE approach: Airway, Breathing, "
        "Circulation, Disability, and Exposure. Life threats must be identified "
        "and corrected before proceeding.",
        "AVPU scale: Alert — patient is fully awake; Voice — responds to verbal "
        "stimuli; Pain — responds only to painful stimuli; Unresponsive — no "
        "response to any stimuli.",
        "Scene size-up: ensure the scene is safe before approaching. Identify "
        "mechanism of injury (MOI) or nature of illness (NOI). Note number of "
        "patients and resource needs.",
    ]


@pytest.fixture
def empty_context_chunks() -> list[str]:
    return []


@pytest.fixture
def sample_history() -> list[dict]:
    return [
        {"role": "user", "content": "I have a patient who fell from a boulder."},
        {"role": "assistant", "content": "Is the scene safe to approach? What is the mechanism of injury — how far did they fall, and onto what surface?"},
        {"role": "user", "content": "Yes safe. About 10 feet onto rocky ground."},
        {"role": "assistant", "content": "Understood. What is their level of consciousness? Are they alert, responding to voice, pain, or unresponsive?"},
    ]
