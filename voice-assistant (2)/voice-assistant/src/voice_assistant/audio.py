"""Microphone capture, Whisper transcription and text-to-speech.

Heavy dependencies are imported lazily so the NLU, tests and text mode work
without a microphone, PyTorch or a speech engine installed
(`pip install -e ".[voice]"` for voice mode).

Whisper receives the recording as an in-memory float32 array, which means no
temporary WAV file and no FFmpeg installation is needed.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class Recorder:
    def __init__(self, sample_rate: int = 16_000, seconds: float = 5.0, device: int | None = None):
        import sounddevice as sd

        self._sd = sd
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.device = device

    def record(self) -> np.ndarray:
        frames = int(self.seconds * self.sample_rate)
        audio = self._sd.rec(
            frames, samplerate=self.sample_rate, channels=1, dtype="float32", device=self.device
        )
        self._sd.wait()
        return audio.flatten()

    @staticmethod
    def list_devices() -> str:
        import sounddevice as sd

        return str(sd.query_devices())


def is_silent(audio: np.ndarray, threshold: float = 0.01) -> bool:
    """Skip transcription when the recording is just background noise (RMS energy)."""
    return audio.size == 0 or float(np.sqrt(np.mean(np.square(audio)))) < threshold


class Transcriber:
    def __init__(self, model_name: str = "base"):
        import whisper

        log.info("Loading Whisper model %r", model_name)
        self._model = whisper.load_model(model_name)

    def transcribe(self, audio: np.ndarray) -> str:
        # Whisper expects 16 kHz mono float32 in [-1, 1], which is what Recorder produces.
        result = self._model.transcribe(audio.astype(np.float32), fp16=False, language="en")
        return result["text"].strip()


class Speaker:
    def __init__(self, rate: int = 165, voice_index: int | None = None):
        import pyttsx3

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)
        self._engine.setProperty("volume", 1.0)
        voices = self._engine.getProperty("voices")
        if voice_index is not None and voice_index < len(voices):
            self._engine.setProperty("voice", voices[voice_index].id)

    def say(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()
