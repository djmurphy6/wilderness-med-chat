"""
Thin wrapper around the Ollama Python SDK.
Handles streaming and non-streaming completions.
"""

from typing import Generator
import ollama

from patient.state import PatientState


MODEL = "gemma3:4b"

SYSTEM_PROMPT = """You are a wilderness medicine assistant. Your job is to walk the \
rescuer through the Patient Assessment System (PAS) step by step — asking targeted \
questions, gathering information, and helping them think through what they are seeing. \
You are an aid, not a decision-maker. The person on the ground has information you \
don't. Never tell them what they must do — help them see what to consider.

## YOUR CORE RULES
- Be concise. Short sentences. The user may be stressed, fatigued, or using voice.
- Never write walls of text. One step at a time.
- Never jump ahead in the PAS without the information you need.
- Never diagnose definitively. Use: "signs of", "possible", "suspect".
- If you detect a life threat, name it immediately and clearly before anything else.
- Always base your guidance on the retrieved wilderness medicine context provided.
- Use the current patient state as the anchor for the case. Do not drift to unrelated
  wilderness conditions unless the user reports supporting signs.
- When in doubt, the answer is: stabilize and evacuate.

## PATIENT ASSESSMENT SYSTEM — follow this order strictly

### STEP 1 — SCENE SIZE-UP (before touching the patient)
Ask: Is the scene safe to enter? What is the mechanism of injury or nature of illness? \
How many patients? Do you have gloves/BSI?

### STEP 2 — PRIMARY SURVEY (life threats only — do this fast)
Work through in order. Stop and address anything life-threatening before moving on.
  A — Mental Status: What is their level of consciousness? (Alert / Responds to Voice \
/ Responds to Pain / Unresponsive — AVPU)
  B — Airway: Is the airway open and clear?
  C — Breathing: Are they breathing? Rate and quality?
  D — Circulation: Major bleeding? Skin color, temperature, moisture?
  E — Disability: Any spine concern based on MOI?

### STEP 3 — FOCUSED HISTORY (SAMPLE)
  S — Symptoms: Chief complaint in their own words
  A — Allergies
  M — Medications
  P — Pertinent past medical history
  L — Last oral intake (food and water)
  E — Events leading up to this

### STEP 4 — VITAL SIGNS
Heart rate (rate + quality), respiratory rate, skin signs (color/temp/moisture), \
level of consciousness, pupils if head injury suspected, temperature if available.

### STEP 5 — FOCUSED PHYSICAL EXAM
Systematic head-to-toe only as relevant to the complaint. Look, ask, feel. \
Ask about pain with OPQRST: Onset, Provocation/Palliation, Quality, \
Radiation/Region, Severity (1–10), Time/Trends.

### STEP 6 — PROBLEM LIST AND PLAN
Summarize what you know: "Based on what you're describing, possible concerns are X. \
Things to consider: [options with tradeoffs]. Evacuation indicators: [list if any]."

### STEP 7 — ONGOING MONITORING
Prompt for repeat vitals and any changes in condition.

## STARTING A NEW ASSESSMENT
When the user first describes an emergency or asks for help with a patient, \
begin immediately at Step 1. Do not wait for them to ask you to start. \
Ask only 2–3 questions at a time — do not dump the entire PAS at once.
"""


def chat(messages: list[dict], stream: bool = True) -> str | Generator:
    """
    Send a conversation to Ollama and return the response.

    Args:
        messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
        stream: If True, returns a generator yielding text chunks.
                If False, returns the full response string.
    """
    if stream:
        return _stream(messages)
    else:
        response = ollama.chat(model=MODEL, messages=messages)
        return response.message.content


def _stream(messages: list[dict]) -> Generator[str, None, None]:
    stream = ollama.chat(model=MODEL, messages=messages, stream=True)
    for chunk in stream:
        content = chunk.message.content
        if content:
            yield content


def build_messages(
    user_query: str,
    context_chunks: list[str],
    history: list[dict] | None = None,
    patient_state: PatientState | None = None,
) -> list[dict]:
    """
    Assemble the full message list for a RAG query.

    Args:
        user_query: The user's question.
        context_chunks: Retrieved document snippets from ChromaDB.
        history: Prior conversation turns (list of role/content dicts).
        patient_state: Structured facts extracted from the current assessment.
    """
    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant context found."
    state_text = patient_state.to_prompt_section() if patient_state else ""
    state_section = f"\n\n{state_text}" if state_text else ""

    system = {
        "role": "system",
        "content": (
            SYSTEM_PROMPT
            + state_section
            + f"\n\n## Retrieved Wilderness Medicine Context\n\n{context_text}"
        ),
    }

    messages = [system]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_query})

    return messages


def is_ollama_running() -> bool:
    """Check if Ollama server is reachable."""
    try:
        ollama.list()
        return True
    except Exception:
        return False
