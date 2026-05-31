"""
Phase 3: Voice-first CLI loop.

Usage:
    python main.py

Flow (voice mode):
    App starts → ● Listening → speak → silence auto-detected → transcribe
    → RAG + LLM → spoken response → ● Listening again (no Enter needed)

Flow (text fallback, when no mic):
    Prompted input → RAG + LLM → text response → repeat
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel

from llm.ollama_client import build_messages, chat, is_ollama_running
from patient.state import PatientState
from rag.query import rag_engine
from stt.transcribe import STTUnavailableError, WhisperTranscriber
from tts.speak import Speaker, select_tts_backend


MAX_HISTORY_MESSAGES = 20

console = Console()


def print_banner(voice_mode: bool) -> None:
    if voice_mode:
        subtitle = (
            "[dim]Listening automatically. Speak and pause to send.\n"
            "Say [bold]new patient[/bold] to reset. Press Ctrl+C to quit.[/dim]"
        )
    else:
        subtitle = (
            "[dim]Type your question and press Enter.\n"
            "Commands: [bold]new[/bold] / [bold]reset[/bold] / [bold]exit[/bold][/dim]"
        )
    console.print(
        Panel(
            f"[bold green]Wilderness Med Chat — Voice[/bold green]\n\n{subtitle}",
            border_style="green",
        )
    )


def build_voice_components() -> tuple[WhisperTranscriber | None, Speaker | None]:
    transcriber: WhisperTranscriber | None = None
    speaker: Speaker | None = None

    try:
        transcriber = WhisperTranscriber()
    except STTUnavailableError as err:
        console.print(f"[yellow]STT unavailable:[/yellow] {err}")
        console.print("[dim]Continuing in typed-input mode.[/dim]")

    candidate_speaker = Speaker()
    if select_tts_backend(prefer_kokoro=candidate_speaker.prefer_kokoro) == "none":
        console.print("[yellow]TTS unavailable:[/yellow] no local speech backend detected.")
        console.print("[dim]Continuing in text-output mode.[/dim]")
    else:
        speaker = candidate_speaker

    return transcriber, speaker


def listen_for_input(transcriber: WhisperTranscriber) -> str:
    """Auto-listen until speech then silence. Returns transcribed text."""
    console.print("\n[dim]● Listening...[/dim]")
    try:
        transcript = transcriber.listen_and_transcribe()
    except STTUnavailableError as err:
        console.print(f"[bold red]Mic error:[/bold red] {err}")
        return ""
    except Exception as err:
        console.print(f"[bold red]STT error:[/bold red] {err}")
        return ""

    if not transcript:
        console.print("[dim]Didn't catch anything — listening again.[/dim]")
        return ""

    console.print(f"[bold cyan]You:[/bold cyan] {transcript}")
    return transcript


def prompt_for_input() -> str:
    """Text fallback when no microphone is available."""
    return console.input("\n[bold cyan]You:[/bold cyan] ").strip()


def run_assistant_turn(
    user_input: str,
    history: list[dict],
    patient_state: PatientState,
    speaker: Speaker | None = None,
) -> str:
    patient_state.update_from_text(user_input)
    retrieval_query = patient_state.to_retrieval_query(user_input)
    context_chunks = rag_engine.retrieve(retrieval_query)

    if context_chunks:
        console.print(f"[dim]Retrieved {len(context_chunks)} chunk(s).[/dim]")

    messages = build_messages(user_input, context_chunks, history, patient_state)

    console.print("\n[bold green]Assistant:[/bold green] ", end="")

    token_stream = chat(messages, stream=True)

    if speaker is not None:
        # Speak each sentence as it completes while the LLM keeps generating.
        full_response = speaker.speak_stream(
            token_stream,
            on_token=lambda t: console.print(t, end="", highlight=False),
        )
    else:
        full_response = ""
        for chunk in token_stream:
            console.print(chunk, end="", highlight=False)
            full_response += chunk

    console.print()
    return full_response


def main() -> None:
    transcriber, speaker = build_voice_components()
    voice_mode = transcriber is not None

    print_banner(voice_mode)

    if not is_ollama_running():
        console.print("[bold red]Error:[/bold red] Ollama is not running. Start it with: [cyan]ollama serve[/cyan]")
        sys.exit(1)

    if rag_engine.is_empty():
        console.print(
            "[yellow]Warning:[/yellow] No documents in the knowledge base yet.\n"
            "Add PDFs to [cyan]data/pdfs/[/cyan] and run [cyan]python -m ingest.ingest[/cyan]\n"
            "Continuing without RAG context.\n"
        )

    history: list[dict] = []
    patient_state = PatientState()

    while True:
        try:
            if voice_mode:
                user_input = listen_for_input(transcriber)  # type: ignore[arg-type]
            else:
                user_input = prompt_for_input()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        lowered = user_input.lower().strip(".,!?")
        if lowered in ("exit", "quit", "q", "goodbye", "bye"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if lowered in ("new", "reset", "new patient", "start over"):
            history = []
            patient_state = PatientState()
            console.print("[dim]Assessment cleared. Ready for new patient.[/dim]")
            continue

        try:
            # Speaker is passed in so speak_stream can overlap generation + speech.
            full_response = run_assistant_turn(user_input, history, patient_state, speaker=speaker)
        except Exception as err:
            console.print(f"\n[bold red]Error:[/bold red] {err}")
            continue

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": full_response})
        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[-MAX_HISTORY_MESSAGES:]


if __name__ == "__main__":
    main()
