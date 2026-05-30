"""
Phase 1: Text-only CLI chat loop.

Usage:
    python main_text.py

Type your wilderness medicine questions and get RAG-grounded answers.
No STT or TTS — pure text in / text out. Good for testing the core loop.
"""

import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.markdown import Markdown

from llm.ollama_client import build_messages, chat, is_ollama_running
from rag.query import rag_engine

console = Console()


def print_banner():
    console.print(Panel(
        "[bold green]Wilderness Med Chat[/bold green]\n"
        "[dim]Offline wilderness medicine assistant — powered by Gemma 3 + RAG[/dim]\n\n"
        "[dim]Type your question and press Enter. Type [bold]exit[/bold] to quit.[/dim]",
        border_style="green",
    ))


def main():
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

    history: list[dict] = []

    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if user_input.lower() in ("new", "reset", "new patient"):
            history = []
            console.print("[dim]Assessment cleared. Ready for new patient.[/dim]")
            continue

        # Retrieve relevant context from ChromaDB
        context_chunks = rag_engine.retrieve(user_input)

        if context_chunks:
            console.print(f"[dim]Retrieved {len(context_chunks)} context chunk(s) from knowledge base.[/dim]")

        # Build prompt with context + conversation history
        messages = build_messages(user_input, context_chunks, history)

        # Stream the response
        console.print("\n[bold green]Assistant:[/bold green] ", end="")
        full_response = ""

        try:
            for chunk in chat(messages, stream=True):
                console.print(chunk, end="", highlight=False)
                full_response += chunk
            console.print()  # newline after streamed response
        except Exception as e:
            console.print(f"\n[bold red]Error communicating with Ollama:[/bold red] {e}")
            continue

        # Keep a sliding window of the last 20 turns (10 exchanges).
        # PAS conversations run longer than simple Q&A — we need enough
        # history for the model to track what it has and hasn't assessed yet.
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": full_response})
        if len(history) > 20:
            history = history[-20:]


if __name__ == "__main__":
    main()
