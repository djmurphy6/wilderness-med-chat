"""
Lightweight patient state tracking for PAS conversations.

This module intentionally uses deterministic extraction instead of an LLM.
The goal is not perfect medical NLP; it is to keep high-value facts anchored
across turns so retrieval and prompting do not drift away from the case.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


@dataclass
class PatientState:
    mechanism: str | None = None
    nature_of_illness: str | None = None
    mental_status: str | None = None
    airway: str | None = None
    breathing: str | None = None
    major_bleeding: str | None = None
    spine_concern: str | None = None
    chief_complaint: str | None = None
    current_pas_step: str = "scene_size_up"
    active_problem_list: list[str] = field(default_factory=list)

    def update_from_text(self, text: str) -> None:
        """Extract durable patient facts from a user turn."""
        normalized = text.lower()

        self._update_mechanism(normalized)
        self._update_mental_status(normalized)
        self._update_airway_breathing(normalized)
        self._update_circulation(normalized)
        self._update_problem_list(normalized)
        self._update_pas_step()

    def to_prompt_section(self) -> str:
        """Render a compact case summary for the system prompt."""
        lines = ["## Current Patient State"]

        fields = [
            ("Mechanism", self.mechanism),
            ("Nature of illness", self.nature_of_illness),
            ("Chief complaint", self.chief_complaint),
            ("Mental status", self.mental_status),
            ("Airway", self.airway),
            ("Breathing", self.breathing),
            ("Major bleeding", self.major_bleeding),
            ("Spine concern", self.spine_concern),
            ("Current PAS step", self.current_pas_step.replace("_", " ")),
        ]

        known_any = False
        for label, value in fields:
            if value:
                known_any = True
                lines.append(f"- {label}: {value}")

        if self.active_problem_list:
            known_any = True
            lines.append(f"- Active concerns: {', '.join(self.active_problem_list)}")

        if not known_any:
            lines.append("- No patient facts captured yet.")

        lines.append(
            "- Stay anchored to these facts. Do not introduce unrelated conditions "
            "unless the user reports supporting signs."
        )
        return "\n".join(lines)

    def to_retrieval_query(self, latest_user_query: str) -> str:
        """
        Build a retrieval query anchored to durable case facts.

        This prevents short follow-up turns like "yes, still breathing" from
        pulling unrelated wilderness medicine chunks.
        """
        parts = [latest_user_query]
        parts.extend(
            value
            for value in [
                self.mechanism,
                self.nature_of_illness,
                self.chief_complaint,
                self.mental_status,
                self.airway,
                self.breathing,
                self.major_bleeding,
                self.spine_concern,
            ]
            if value
        )
        parts.extend(self.active_problem_list)
        return " ".join(parts)

    def _update_mechanism(self, text: str) -> None:
        head_impact = _contains_any(
            text,
            [
                "hit in the head",
                "hit his head",
                "hit her head",
                "hit their head",
                "head hit",
                "struck his head",
                "struck her head",
                "struck their head",
                "hit by a rock",
                "hit with a rock",
                "rock hit",
            ],
        )
        fall = _contains_any(text, ["fell", "fall", "fallen"])
        crash = _contains_any(text, ["crash", "collision", "car wreck", "bike wreck"])
        twisted = _contains_any(text, ["twisted", "rolled ankle", "rolled her ankle", "rolled his ankle"])

        if head_impact:
            self.mechanism = "head impact"
            self.chief_complaint = self.chief_complaint or "head injury"
            self.spine_concern = "yes"
            _append_unique(self.active_problem_list, "head injury")
            _append_unique(self.active_problem_list, "possible spine injury")
        elif fall:
            self.mechanism = "fall"
            self.spine_concern = self.spine_concern or "possible"
            _append_unique(self.active_problem_list, "possible spine injury")
        elif crash:
            self.mechanism = "crash or collision"
            self.spine_concern = "yes"
            _append_unique(self.active_problem_list, "possible spine injury")
        elif twisted:
            self.mechanism = "twisting extremity injury"
            self.chief_complaint = self.chief_complaint or "extremity injury"

    def _update_mental_status(self, text: str) -> None:
        if _contains_any(text, ["unresponsive", "not responding", "won't wake", "wont wake"]):
            self.mental_status = "unresponsive"
            _append_unique(self.active_problem_list, "unresponsive patient")
        elif _contains_any(text, ["responds to pain", "painful stimuli"]):
            self.mental_status = "responds to pain"
        elif _contains_any(text, ["responds to voice", "responding to voice"]):
            self.mental_status = "responds to voice"
        elif _contains_any(text, ["confused", "acting weird", "altered", "disoriented"]):
            self.mental_status = "altered or confused"
            _append_unique(self.active_problem_list, "altered mental status")
        elif _contains_any(text, ["alert and oriented", "alert", "awake"]):
            self.mental_status = "alert"

    def _update_airway_breathing(self, text: str) -> None:
        if _contains_any(text, ["not breathing", "isn't breathing", "isnt breathing", "stopped breathing"]):
            self.breathing = "absent"
            _append_unique(self.active_problem_list, "not breathing")
            return

        if _contains_any(
            text,
            [
                "still breathing",
                "is breathing",
                "they are breathing",
                "he is breathing",
                "she is breathing",
                "breathing normally",
            ],
        ):
            self.breathing = "present"

        if _contains_any(text, ["throat is swelling", "throat swelling", "airway swelling"]):
            self.airway = "threatened"
            _append_unique(self.active_problem_list, "airway swelling")
            _append_unique(self.active_problem_list, "possible anaphylaxis")
        elif _contains_any(text, ["airway is clear", "clear airway"]):
            self.airway = "clear"

    def _update_circulation(self, text: str) -> None:
        if _contains_any(text, ["no major bleeding", "no severe bleeding", "not bleeding"]):
            self.major_bleeding = "absent"
        elif _contains_any(text, ["major bleeding", "bleeding a lot", "spurting blood", "severe bleeding"]):
            self.major_bleeding = "present"
            _append_unique(self.active_problem_list, "major bleeding")

    def _update_problem_list(self, text: str) -> None:
        if _contains_any(text, ["chest pain", "short of breath", "difficulty breathing"]):
            self.chief_complaint = self.chief_complaint or "chest pain or breathing concern"
            _append_unique(self.active_problem_list, "chest pain or breathing concern")

        if _contains_any(text, ["bee sting", "stung by bees", "throat is swelling", "allergic reaction"]):
            self.nature_of_illness = self.nature_of_illness or "allergic reaction"
            _append_unique(self.active_problem_list, "possible anaphylaxis")

        if _contains_any(text, ["cold", "rain", "shivering", "hypothermia"]):
            self.nature_of_illness = self.nature_of_illness or "cold exposure"
            _append_unique(self.active_problem_list, "cold exposure")

        if _contains_any(text, ["frostbite", "frozen skin", "waxy skin"]):
            self.nature_of_illness = self.nature_of_illness or "cold injury"
            _append_unique(self.active_problem_list, "possible frostbite")

        if _contains_any(text, ["ankle", "wrist", "arm", "leg"]) and _contains_any(
            text, ["pain", "twisted", "rolled", "swollen", "deformed"]
        ):
            self.chief_complaint = self.chief_complaint or "extremity injury"
            _append_unique(self.active_problem_list, "extremity injury")

    def _update_pas_step(self) -> None:
        if self.mental_status or self.airway or self.breathing or self.major_bleeding or self.spine_concern:
            self.current_pas_step = "primary_survey"
        elif self.mechanism or self.nature_of_illness:
            self.current_pas_step = "scene_size_up"
