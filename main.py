"""
Phase 3: Voice-first CLI loop.

Usage:
    python main.py

Flow:
    Press Enter -> mic opens -> speak -> silence auto-detected -> transcribe
    -> RAG + LLM -> spoken response -> repeat
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


def print_banner() -> None:
    console.print(
        Panel(
            "[bold green]Wilderness Med Chat — Voice[/bold green]\n"
            "[dim]Press Enter to start speaking. Stops automatically when you pause.\n"
            "You can also type text directly.[/dim]\n\n"
            "[dim]Commands: [bold]new[/bold] / [bold]reset[/bold] / [bold]exit[/bold][/dim]",
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


def collect_user_input(transcriber: WhisperTranscriber | None) -> str:
    prompt = "\n[bold cyan]You[/bold cyan] ([dim]Enter=speak, or type[/dim]): "
    raw = console.input(prompt).strip()

    if raw:
        return raw

    if transcriber is None:
        console.print("[yellow]Microphone mode unavailable; type your message instead.[/yellow]")
        return ""

    console.print("[dim]● Listening — speak now, pausing will stop the recording...[/dim]")
    try:
        transcript = transcriber.listen_and_transcribe()
    except STTUnavailableError as err:
        console.print(f"[bold red]Mic error:[/bold red] {err}")
        return ""
    except Exception as err:
        console.print(f"[bold red]STT error:[/bold red] {err}")
        return ""

    if not transcript:
        console.print("[dim]I didn't catch anything. Try again.[/dim]")
        return ""

    console.print(f"[bold cyan]You (transcribed):[/bold cyan] {transcript}")
    return transcript


def run_assistant_turn(user_input: str, history: list[dict], patient_state: PatientState) -> str:
    patient_state.update_from_text(user_input)
    retrieval_query = patient_state.to_retrieval_query(user_input)
    context_chunks = rag_engine.retrieve(retrieval_query)

    if context_chunks:
        console.print(f"[dim]Retrieved {len(context_chunks)} context chunk(s).[/dim]")

    messages = build_messages(user_input, context_chunks, history, patient_state)

    console.print("\n[bold green]Assistant:[/bold green] ", end="")
    full_response = ""
    for chunk in chat(messages, stream=True):
        console.print(chunk, end="", highlight=False)
        full_response += chunk
    console.print()

    return full_response


def main() -> None:
    print_banner()

    if not is_ollama_running():
        console.print("[bold red]Error:[/bold red] Ollama is not running. Start it with: [cyan]ollama serve[/cyan]")
        sys.exit(1)

    if rag_engine.is_empty():
        console.print(
            "[yellow]Warning:[/yellow] No documents in the knowledge base yet.\n"
            "Add PDFs to [cyan]data/pdfs/[/cyan] and run [cyan]python -m ingest.ingest[/cyan]\n"
            "Continuing without RAG context — answers will be from model knowledge only.\n"
        )

    transcriber, speaker = build_voice_components()
    history: list[dict] = []
    patient_state = PatientState()

    while True:
        try:
            user_input = collect_user_input(transcriber)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if lowered in ("new", "reset", "new patient"):
            history = []
            patient_state = PatientState()
            console.print("[dim]Assessment cleared. Ready for new patient.[/dim]")
            continue

        try:
            full_response = run_assistant_turn(user_input, history, patient_state)
        except Exception as err:
            console.print(f"\n[bold red]Error during assistant turn:[/bold red] {err}")
            continue

        if speaker is not None:
            try:
                speaker.speak(full_response)
            except Exception as err:
                console.print(f"[yellow]TTS error:[/yellow] {err}")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": full_response})
        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[-MAX_HISTORY_MESSAGES:]


if __name__ == "__main__":
    main()
