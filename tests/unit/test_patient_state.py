"""
Unit tests for deterministic patient state extraction.
"""

import pytest

from patient.state import PatientState


pytestmark = pytest.mark.unit


class TestPatientStateExtraction:
    def test_tracks_unresponsive_head_injury_that_is_still_breathing(self):
        state = PatientState()

        state.update_from_text(
            "My friend got hit in the head with a rock. "
            "They are unresponsive but still breathing."
        )

        assert state.mechanism == "head impact"
        assert state.chief_complaint == "head injury"
        assert state.mental_status == "unresponsive"
        assert state.breathing == "present"
        assert state.spine_concern == "yes"
        assert "head injury" in state.active_problem_list
        assert "unresponsive patient" in state.active_problem_list
        assert "possible spine injury" in state.active_problem_list

    def test_short_followup_retrieval_query_keeps_prior_case_context(self):
        state = PatientState()
        state.update_from_text("My friend got hit in the head with a rock.")
        state.update_from_text("They are unresponsive but still breathing.")

        query = state.to_retrieval_query("What next?")

        assert "What next?" in query
        assert "head impact" in query
        assert "head injury" in query
        assert "unresponsive" in query
        assert "present" in query
        assert "possible spine injury" in query

    def test_does_not_add_cold_injury_without_cold_findings(self):
        state = PatientState()

        state.update_from_text("My friend got hit in the head with a rock and is unresponsive.")

        assert "cold exposure" not in state.active_problem_list
        assert "possible frostbite" not in state.active_problem_list
        assert state.nature_of_illness is None

    def test_tracks_absent_breathing_before_present_breathing_phrase(self):
        state = PatientState()

        state.update_from_text("He is not breathing.")

        assert state.breathing == "absent"
        assert "not breathing" in state.active_problem_list

    def test_reset_is_just_a_fresh_state(self):
        state = PatientState()
        state.update_from_text("She has chest pain and shortness of breath.")

        state = PatientState()

        assert state.active_problem_list == []
        assert state.chief_complaint is None


class TestPatientStatePromptSection:
    def test_prompt_section_names_known_facts_and_anchor_instruction(self):
        state = PatientState()
        state.update_from_text("My friend got hit in the head and is unresponsive but still breathing.")

        section = state.to_prompt_section()

        assert "## Current Patient State" in section
        assert "Mechanism: head impact" in section
        assert "Mental status: unresponsive" in section
        assert "Breathing: present" in section
        assert "Stay anchored to these facts" in section
