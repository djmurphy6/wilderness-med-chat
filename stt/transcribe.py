"""
Speech-to-text helpers for the voice loop.

This module intentionally uses lazy imports so the rest of the app can run in
text mode even when STT dependencies are missing.

Design note on sample rates: sounddevice does not resample on the fly. If you
ask for 16000 Hz from a device whose native rate differs (e.g. 24000 Hz on
AirPods, 44100 Hz on MacBook Pro mic) you silently get zeros. We therefore
record at the device's native rate and resample to 16000 Hz (Whisper's expected
rate) using scipy before transcription.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

WHISPER_SAMPLE_RATE = 16000
DEFAULT_RECORD_SECONDS = 6.0

# Primes Whisper's language model toward wilderness medicine vocabulary so
# domain-specific terms (AVPU, MOI, SpO2, tourniquet…) are decoded correctly.
WILDERNESS_MED_PROMPT = (
    "Patient assessment system, mechanism of injury, nature of illness, "
    "AVPU, alert and oriented, unresponsive, airway, breathing, circulation, "
    "chief complaint, SAMPLE history, allergies, medications, pulse, "
    "respirations, skin signs, tourniquet, splint, hypothermia, frostbite, "
    "anaphylaxis, epinephrine, SpO2, evacuation, wilderness medicine."
)


class STTUnavailableError(RuntimeError):
    """Raised when STT dependencies are not installed or usable."""


def stt_dependencies_available() -> tuple[bool, list[str]]:
    """Return whether required STT packages are importable."""
    missing: list[str] = []
    for package in ("faster_whisper", "sounddevice"):
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    return (len(missing) == 0, missing)


def _native_input_rate() -> int:
    """Return the default input device's native sample rate."""
    sd = __import__("sounddevice")
    info = sd.query_devices(kind="input")
    return int(info["default_samplerate"])


def _resample(audio, from_rate: int, to_rate: int):
    """Resample a mono float32 numpy array from from_rate to to_rate."""
    if from_rate == to_rate:
        return audio
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(from_rate, to_rate)
    return resample_poly(audio, to_rate // g, from_rate // g).astype("float32")


@dataclass
class WhisperTranscriber:
    # small.en is noticeably more accurate than base.en for medical vocabulary
    # and still fast enough on M1 CPU (~1-2s for a 6s clip).
    model_size: str = "small.en"
    device: str = "auto"
    compute_type: str = "int8"
    # Set to a specific device index (int) to override the system default.
    input_device: int | None = None

    def __post_init__(self) -> None:
        ok, missing = stt_dependencies_available()
        if not ok:
            joined = ", ".join(missing)
            raise STTUnavailableError(
                f"STT dependencies missing: {joined}. "
                "Install with: pip install faster-whisper sounddevice"
            )

        from faster_whisper import WhisperModel  # type: ignore

        self._sounddevice = __import__("sounddevice")
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def _capture_rate(self) -> int:
        """Native sample rate of the selected input device."""
        if self.input_device is not None:
            info = self._sounddevice.query_devices(self.input_device)
        else:
            info = self._sounddevice.query_devices(kind="input")
        return int(info["default_samplerate"])

    def record(self, seconds: float = DEFAULT_RECORD_SECONDS):
        """
        Capture mono microphone audio and return float32 samples at
        WHISPER_SAMPLE_RATE (16000 Hz), resampling from the device's native
        rate if necessary.
        """
        import numpy as np

        if seconds <= 0:
            raise ValueError("Recording duration must be greater than zero.")

        native_rate = self._capture_rate()
        frame_count = int(seconds * native_rate)

        kwargs: dict = dict(
            samplerate=native_rate,
            channels=1,
            dtype="float32",
        )
        if self.input_device is not None:
            kwargs["device"] = self.input_device

        recording = self._sounddevice.rec(frame_count, **kwargs)
        self._sounddevice.wait()
        audio = recording.squeeze()

        peak = float(np.abs(audio).max())

        if peak < 1e-6:
            raise STTUnavailableError(
                "Microphone returned silence. Check that your mic is not muted "
                "and that this app has microphone permission in "
                "System Settings → Privacy & Security → Microphone."
            )

        # Normalize to [-1, 1] so Whisper gets a full-scale signal regardless
        # of how quiet the mic hardware is (AirPods peak at ~0.004 raw).
        audio = audio / peak

        return _resample(audio, native_rate, WHISPER_SAMPLE_RATE)

    def transcribe_audio(self, audio) -> str:
        """Transcribe in-memory 16 kHz float32 audio into text."""
        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=5,
            initial_prompt=WILDERNESS_MED_PROMPT,
            # VAD filter disabled: quiet speech can be dropped. The silence
            # check in record() already guards against truly blank input.
            vad_filter=False,
        )
        text_parts = [segment.text.strip() for segment in segments if segment.text.strip()]
        return " ".join(text_parts).strip()

    def record_and_transcribe(self, seconds: float = DEFAULT_RECORD_SECONDS) -> str:
        """Record microphone audio and return the transcription."""
        audio = self.record(seconds=seconds)
        return self.transcribe_audio(audio)

    def listen_until_silence(
        self,
        silence_duration: float = 1.2,
        max_duration: float = 30.0,
    ):
        """
        Stream microphone audio, auto-stop when silence follows speech.

        Algorithm:
          1. Sample ~0.4s of ambient noise to set an adaptive speech threshold.
          2. Collect 100ms chunks via InputStream callback.
          3. Transition WAITING → SPEECH when RMS exceeds threshold.
          4. Transition SPEECH → DONE after silence_duration seconds of quiet.
          5. Normalize and resample to 16 kHz for Whisper.

        Returns a float32 numpy array ready for transcribe_audio().
        """
        import threading
        import numpy as np

        native_rate = self._capture_rate()
        chunk_frames = int(native_rate * 0.1)   # 100 ms per chunk
        max_chunks = int(max_duration / 0.1)
        silence_chunks_needed = int(silence_duration / 0.1)

        # --- Phase 1: ambient noise calibration (~0.4 s) ---
        noise_samples: list = []

        def _noise_cb(indata, frames, time, status):
            noise_samples.append(indata[:, 0].copy())

        noise_kwargs: dict = dict(
            samplerate=native_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_frames,
            callback=_noise_cb,
        )
        if self.input_device is not None:
            noise_kwargs["device"] = self.input_device

        with self._sounddevice.InputStream(**noise_kwargs):
            import time as _time
            _time.sleep(0.4)

        if noise_samples:
            noise_floor = float(np.sqrt(np.mean(np.concatenate(noise_samples) ** 2)))
        else:
            noise_floor = 0.0

        # Speech threshold: 4× the noise floor, with a sensible minimum.
        threshold = max(noise_floor * 4.0, 5e-4)

        # --- Phase 2: listen for speech then silence ---
        audio_chunks: list = []
        speech_started = False
        silent_chunk_count = 0
        done_event = threading.Event()

        def _listen_cb(indata, frames, time, status):
            nonlocal speech_started, silent_chunk_count

            chunk = indata[:, 0].copy()
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms >= threshold:
                speech_started = True
                silent_chunk_count = 0
                audio_chunks.append(chunk)
            elif speech_started:
                # Keep the trailing silence (natural sentence endings need it).
                audio_chunks.append(chunk)
                silent_chunk_count += 1
                if silent_chunk_count >= silence_chunks_needed:
                    done_event.set()

            if len(audio_chunks) >= max_chunks:
                done_event.set()

        listen_kwargs = dict(
            samplerate=native_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_frames,
            callback=_listen_cb,
        )
        if self.input_device is not None:
            listen_kwargs["device"] = self.input_device

        with self._sounddevice.InputStream(**listen_kwargs):
            done_event.wait(timeout=max_duration + 2)

        if not audio_chunks:
            raise STTUnavailableError("No speech detected within the time limit.")

        audio = np.concatenate(audio_chunks)
        peak = float(np.abs(audio).max())
        if peak > 1e-6:
            audio = audio / peak

        return _resample(audio, native_rate, WHISPER_SAMPLE_RATE)

    def listen_and_transcribe(
        self,
        silence_duration: float = 1.2,
        max_duration: float = 30.0,
    ) -> str:
        """Auto-detect speech start and end, then transcribe."""
        audio = self.listen_until_silence(
            silence_duration=silence_duration,
            max_duration=max_duration,
        )
        return self.transcribe_audio(audio)
