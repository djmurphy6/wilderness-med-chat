"""
Thin wrapper around the Ollama Python SDK.
Handles streaming and non-streaming completions.
"""

from typing import Generator
import ollama


MODEL = "gemma3:4b"

SYSTEM_PROMPT = """You are a wilderness medicine assistant. You provide clear, calm, 
evidence-based first aid and medical guidance for backcountry emergencies.

Guidelines:
- Be concise and actionable. The user may be injured or under stress.
- Always lead with the most critical action first.
- When in doubt, recommend evacuation and professional medical care.
- Never diagnose definitively — use language like "signs of", "possible", "suspect".
- If a situation is life-threatening, say so clearly and immediately.
- Base your answers on the retrieved wilderness medicine context provided.
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


def build_messages(user_query: str, context_chunks: list[str], history: list[dict] | None = None) -> list[dict]:
    """
    Assemble the full message list for a RAG query.

    Args:
        user_query: The user's question.
        context_chunks: Retrieved document snippets from ChromaDB.
        history: Prior conversation turns (list of role/content dicts).
    """
    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant context found."

    system = {
        "role": "system",
        "content": SYSTEM_PROMPT + f"\n\n## Retrieved Wilderness Medicine Context\n\n{context_text}",
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
