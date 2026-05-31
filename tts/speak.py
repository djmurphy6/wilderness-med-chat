"""
Text-to-speech helpers for the voice loop.

Current backend priority:
1. macOS `say` command (zero setup for development)
2. no backend available -> explicit error

Kokoro integration is planned for a follow-up once model assets are added.

Streaming TTS:
  Speaker.speak_stream() consumes a token generator, detects sentence
  boundaries, and dispatches each completed sentence to a background thread
  so speech begins while the LLM is still generating the next sentence.
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable, Generator, Iterable


class TTSUnavailableError(RuntimeError):
    """Raised when no local TTS backend is available."""


# Sentence boundary: punctuation followed by whitespace then an uppercase letter
# or a digit (handles numbered list items like "1. First...").
# This avoids false splits on abbreviations like "Dr. Smith" or "e.g. the".
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(text: str) -> tuple[list[str], str]:
    """
    Split text at sentence boundaries. Returns (complete_sentences, remainder).
    The remainder has not yet reached a boundary and should be carried forward.
    """
    parts = _SENTENCE_SPLIT_RE.split(text)
    if len(parts) == 1:
        return [], text
    # All parts except the last are complete sentences.
    sentences = [s.strip() for s in parts[:-1] if s.strip()]
    return sentences, parts[-1]


def has_macos_say() -> bool:
    return shutil.which("say") is not None


def can_use_kokoro() -> bool:
    try:
        import importlib.util

        package_present = importlib.util.find_spec("kokoro_onnx") is not None
        has_model_path = bool(os.getenv("KOKORO_MODEL_PATH"))
        has_voice_path = bool(os.getenv("KOKORO_VOICES_PATH"))
        return package_present and has_model_path and has_voice_path
    except Exception:
        return False


def select_tts_backend(prefer_kokoro: bool = False) -> str:
    """Return selected backend name: 'kokoro', 'say', or 'none'."""
    if prefer_kokoro and can_use_kokoro():
        return "kokoro"
    if has_macos_say():
        return "say"
    if can_use_kokoro():
        return "kokoro"
    return "none"


@dataclass
class Speaker:
    prefer_kokoro: bool = False
    macos_voice: str | None = None
    macos_rate: int = 175

    def speak(self, text: str) -> None:
        """Speak a complete string synchronously (blocks until done)."""
        if not text.strip():
            return

        backend = select_tts_backend(prefer_kokoro=self.prefer_kokoro)

        if backend == "say":
            self._speak_with_say(text)
            return

        if backend == "kokoro":
            self._speak_with_kokoro(text)
            return

        raise TTSUnavailableError(
            "No TTS backend available. On macOS, install/enable `say`; "
            "otherwise configure Kokoro with KOKORO_MODEL_PATH and KOKORO_VOICES_PATH."
        )

    def speak_stream(
        self,
        token_gen: Iterable[str],
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """
        Consume a streaming token generator, speaking each sentence as it
        completes while the LLM continues generating the next one.

        How it works (macOS `say` backend):
          `say -o file.aiff` synthesizes to disk ~5-10× faster than real-time.
          A background playback thread plays each AIFF with `afplay` the moment
          synthesis finishes. While afplay plays sentence N (~3 s), the main
          thread is already synthesizing sentence N+1 to a file (~0.3 s).
          No per-sentence subprocess startup gaps; natural silence is from
          afplay finishing playback, which matches speech cadence exactly.

        For non-`say` backends (or when this optimisation is unavailable)
        the method falls back to a queued-speak approach.
        """
        backend = select_tts_backend(prefer_kokoro=self.prefer_kokoro)

        if backend == "say":
            return self._speak_stream_say(token_gen, on_token)

        # Generic fallback: queue sentences to a background speak() thread.
        return self._speak_stream_generic(token_gen, on_token)

    # ------------------------------------------------------------------
    # Internal stream helpers
    # ------------------------------------------------------------------

    def _speak_stream_say(
        self,
        token_gen: Iterable[str],
        on_token: Callable[[str], None] | None,
    ) -> str:
        """
        say-specific streaming: synthesize to AIFF files, play with afplay.
        Synthesis is ~5-10× real-time on Apple Silicon so there is always
        a pre-synthesized file ready before the previous one finishes playing.
        """
        import os
        import tempfile

        playback_q: queue.Queue[str | None] = queue.Queue()

        def _playback_worker() -> None:
            while True:
                path = playback_q.get()
                if path is None:
                    break
                try:
                    subprocess.run(["afplay", path], check=False)
                finally:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        playback_thread = threading.Thread(target=_playback_worker, daemon=True)
        playback_thread.start()

        def _synthesize_to_file(text: str) -> None:
            if not text.strip():
                return
            tmp = tempfile.NamedTemporaryFile(suffix=".aiff", delete=False)
            tmp.close()
            cmd = ["say", "-o", tmp.name, "-r", str(self.macos_rate)]
            if self.macos_voice:
                cmd.extend(["-v", self.macos_voice])
            cmd.append(text)
            subprocess.run(cmd, check=True)
            playback_q.put(tmp.name)

        buffer = ""
        full_text = ""

        for token in token_gen:
            if on_token:
                on_token(token)
            buffer += token
            full_text += token

            sentences, buffer = _split_sentences(buffer)
            for sentence in sentences:
                _synthesize_to_file(sentence)

        if buffer.strip():
            _synthesize_to_file(buffer.strip())

        playback_q.put(None)
        playback_thread.join()

        return full_text

    def _speak_stream_generic(
        self,
        token_gen: Iterable[str],
        on_token: Callable[[str], None] | None,
    ) -> str:
        """Fallback: queue completed sentences to a background speak() thread."""
        sentence_q: queue.Queue[str | None] = queue.Queue()

        def _worker() -> None:
            while True:
                item = sentence_q.get()
                if item is None:
                    break
                try:
                    self.speak(item)
                except Exception:
                    pass

        worker_thread = threading.Thread(target=_worker, daemon=True)
        worker_thread.start()

        buffer = ""
        full_text = ""

        for token in token_gen:
            if on_token:
                on_token(token)
            buffer += token
            full_text += token
            sentences, buffer = _split_sentences(buffer)
            for sentence in sentences:
                sentence_q.put(sentence)

        if buffer.strip():
            sentence_q.put(buffer.strip())

        sentence_q.put(None)
        worker_thread.join()
        return full_text

    def _speak_with_say(self, text: str) -> None:
        cmd = ["say", "-r", str(self.macos_rate)]
        if self.macos_voice:
            cmd.extend(["-v", self.macos_voice])
        cmd.append(text)
        subprocess.run(cmd, check=True)

    def _speak_with_kokoro(self, text: str) -> None:
        raise TTSUnavailableError(
            "Kokoro backend selected but runtime playback is not wired yet. "
            "Use macOS `say` for now, or implement Kokoro playback in this method."
        )
