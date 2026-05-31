"""
Unit tests for voice support modules.
"""

import pytest

from stt.transcribe import STTUnavailableError, WhisperTranscriber
from tts.speak import Speaker, TTSUnavailableError, select_tts_backend


pytestmark = pytest.mark.unit


def test_transcriber_raises_when_dependencies_missing(monkeypatch):
    monkeypatch.setattr(
        "stt.transcribe.stt_dependencies_available",
        lambda: (False, ["faster_whisper", "sounddevice"]),
    )
    with pytest.raises(STTUnavailableError):
        WhisperTranscriber()


def test_select_tts_backend_prefers_kokoro_when_requested(monkeypatch):
    monkeypatch.setattr("tts.speak.can_use_kokoro", lambda: True)
    monkeypatch.setattr("tts.speak.has_macos_say", lambda: True)
    assert select_tts_backend(prefer_kokoro=True) == "kokoro"


def test_select_tts_backend_falls_back_to_say(monkeypatch):
    monkeypatch.setattr("tts.speak.can_use_kokoro", lambda: False)
    monkeypatch.setattr("tts.speak.has_macos_say", lambda: True)
    assert select_tts_backend(prefer_kokoro=False) == "say"


def test_speaker_raises_when_no_backend(monkeypatch):
    monkeypatch.setattr("tts.speak.select_tts_backend", lambda prefer_kokoro=False: "none")
    with pytest.raises(TTSUnavailableError):
        Speaker().speak("hello")
