"""
Text-to-speech helpers for the voice loop.

Current backend priority:
1. macOS `say` command (zero setup for development)
2. no backend available -> explicit error

Kokoro integration is planned for a follow-up once model assets are added.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


class TTSUnavailableError(RuntimeError):
    """Raised when no local TTS backend is available."""


def has_macos_say() -> bool:
    return shutil.which("say") is not None


def can_use_kokoro() -> bool:
    # Package + model assets are both required. Keeping this strict check avoids
    # false positives in environments where the wheel exists but no models do.
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

    def _speak_with_say(self, text: str) -> None:
        cmd = ["say", "-r", str(self.macos_rate)]
        if self.macos_voice:
            cmd.extend(["-v", self.macos_voice])
        cmd.append(text)
        subprocess.run(cmd, check=True)

    def _speak_with_kokoro(self, text: str) -> None:
        """
        Placeholder for Kokoro ONNX playback.

        We intentionally defer full Kokoro wiring until model asset paths and
        preferred runtime settings are finalized for both Mac and Jetson.
        """
        raise TTSUnavailableError(
            "Kokoro backend selected but runtime playback is not wired yet. "
            "Use macOS `say` for now, or implement Kokoro playback in this method."
        )
