"""Speech-to-text adapters used by the inference loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..config import AsrConfig, AudioConfig


@dataclass(slots=True, frozen=True)
class Transcript:
    """Normalized speech-to-text output."""

    text: str
    confidence: float
    source: str


class SpeechToTextAdapter(Protocol):
    """Adapter interface for keyword spotting or full STT engines."""

    def transcribe(self, segment: bytes) -> Transcript | None:
        """Return a transcript for ``segment`` or ``None`` if undecodable."""


class StubAdapter:
    """Fallback adapter that never emits transcripts."""

    def transcribe(self, segment: bytes) -> Transcript | None:  # noqa: D401
        return None


class VoskAdapter:
    """Wrapper around the Vosk/Kaldi recognizer for offline STT."""

    def __init__(self, model_path: str, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Vosk model directory '{path}' not found. "
                "Download a small model and update asr.model_path."
            )
        try:
            from vosk import Model
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "vosk package is not installed. Add it to your environment."
            ) from exc

        self._model = Model(str(path))

    def transcribe(self, segment: bytes) -> Transcript | None:
        """Run recognition on ``segment`` and return the top transcript."""
        try:
            from vosk import KaldiRecognizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("vosk package missing at runtime.") from exc

        recognizer = KaldiRecognizer(self._model, self.sample_rate)
        # AcceptWaveform expects full PCM bytes; single call is fine for segments.
        recognizer.AcceptWaveform(segment)
        data = json.loads(recognizer.Result())
        text = (data.get("text") or "").strip()
        if not text:
            return None

        words = data.get("result", [])
        confidences = [float(w.get("conf", 0.0)) for w in words if "conf" in w]
        confidence = sum(confidences) / len(confidences) if confidences else 1.0
        return Transcript(text=text, confidence=confidence, source="stt")


def build_asr_adapter(asr_cfg: AsrConfig, audio_cfg: AudioConfig) -> SpeechToTextAdapter:
    """Factory that maps config settings to a concrete adapter."""
    mode = asr_cfg.mode.lower()
    if mode == "vosk":
        return VoskAdapter(model_path=asr_cfg.model_path, sample_rate=audio_cfg.rate_hz)
    if mode == "stub":
        return StubAdapter()
    raise ValueError(f"Unsupported ASR mode: {asr_cfg.mode}")
