"""
Speech-to-text helpers for the voice loop.

This module intentionally uses lazy imports so the rest of the app can run in
text mode even when STT dependencies are missing.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_RECORD_SECONDS = 6.0


class STTUnavailableError(RuntimeError):
    """Raised when STT dependencies are not installed or usable."""


def stt_dependencies_available() -> tuple[bool, list[str]]:
    """Return whether required STT packages are importable."""
    missing: list[str] = []
    for package in ("faster_whisper", "sounddevice"):
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    return (len(missing) == 0, missing)


@dataclass
class WhisperTranscriber:
    model_size: str = "base.en"
    sample_rate: int = DEFAULT_SAMPLE_RATE
    device: str = "auto"
    compute_type: str = "int8"

    def __post_init__(self) -> None:
        ok, missing = stt_dependencies_available()
        if not ok:
            joined = ", ".join(missing)
            raise STTUnavailableError(
                f"STT dependencies missing: {joined}. "
                "Install with: pip install faster-whisper sounddevice"
            )

        # Lazy imports keep import-time failures from breaking text-only usage.
        from faster_whisper import WhisperModel  # type: ignore

        self._sounddevice = __import__("sounddevice")
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def record(self, seconds: float = DEFAULT_RECORD_SECONDS):
        """
        Capture mono microphone audio and return float32 samples.

        Raises STTUnavailableError if the system returns silence, which
        typically means macOS has not granted microphone permission.
        """
        import numpy as np

        frame_count = int(seconds * self.sample_rate)
        if frame_count <= 0:
            raise ValueError("Recording duration must be greater than zero.")

        recording = self._sounddevice.rec(
            frame_count,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        self._sounddevice.wait()

        audio = recording.squeeze()

        if float(np.abs(audio).max()) < 1e-6:
            raise STTUnavailableError(
                "Microphone returned silence — this usually means macOS has not "
                "granted microphone permission to this terminal.\n"
                "Fix: System Settings → Privacy & Security → Microphone → "
                "enable access for Cursor (or Terminal)."
            )

        return audio

    def transcribe_audio(self, audio) -> str:
        """Transcribe in-memory audio samples into a single text string."""
        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=True,
        )
        text_parts = [segment.text.strip() for segment in segments if segment.text.strip()]
        return " ".join(text_parts).strip()

    def record_and_transcribe(self, seconds: float = DEFAULT_RECORD_SECONDS) -> str:
        """Record microphone audio and return the transcription."""
        audio = self.record(seconds=seconds)
        return self.transcribe_audio(audio)
