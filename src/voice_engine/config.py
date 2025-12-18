from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AudioConfig:
    """Parameters for the audio capture device."""

    device_name: str = "Jabra"
    rate_hz: int = 16_000
    frame_ms: int = 20
    chunk_frames: int = 10  # number of frames per capture read
    sample_width: int = 2  # bytes per sample (16-bit PCM)
    channels: int = 1  # mono capture


@dataclass(slots=True)
class VadConfig:
    """Voice Activity Detection thresholds and timing."""

    energy_thresh_db: float = -30.0
    min_speech_ms: int = 300
    max_speech_ms: int = 3000
    hangover_ms: int = 200


@dataclass(slots=True)
class GatewayConfig:
    """HTTP endpoint and retry policy for posting events."""

    url: str = "http://127.0.0.1:8765/"
    timeout_ms: int = 200
    backoff_ms: tuple[int, ...] = (200, 500, 1000, 2000)


@dataclass(slots=True)
class QueueConfig:
    """Bounded queue capacities for each pipeline stage."""

    frames: int = 256
    segments: int = 32
    posts: int = 64


@dataclass(slots=True)
class LoggingConfig:
    """Global logging preferences."""

    level: str = "INFO"

@dataclass(slots=True)
class AsrConfig:
    """Automatic Speech Recognition (ASR) settings."""

    mode: str = "vosk"  # or "stub" for tests
    model_path: str = "models/vosk"
    min_confidence: float = 0.6


@dataclass(slots=True)
class VoiceEngineConfig:
    """Top-level configuration for the voice engine service."""

    audio: AudioConfig = dataclasses.field(default_factory=AudioConfig)
    vad: VadConfig = dataclasses.field(default_factory=VadConfig)
    asr: AsrConfig = dataclasses.field(default_factory=AsrConfig)
    gateway: GatewayConfig = dataclasses.field(default_factory=GatewayConfig)
    queues: QueueConfig = dataclasses.field(default_factory=QueueConfig)
    logging: LoggingConfig = dataclasses.field(default_factory=LoggingConfig)


def load_config(path: str | Path) -> VoiceEngineConfig:
    """
    Load configuration from ``path`` if it exists; otherwise, return defaults.

    Parameters
    ----------
    path : str | Path
        Location of the YAML config file.
    """

    data: dict[str, Any] = {}
    p = Path(path)
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return VoiceEngineConfig(
        audio=AudioConfig(**data.get("audio", {})),
        vad=VadConfig(**data.get("vad", {})),
        asr=AsrConfig(**data.get("asr", {})),
        gateway=GatewayConfig(**data.get("gateway", {})),
        queues=QueueConfig(**data.get("queues", {})),
        logging=LoggingConfig(**data.get("logging", {})),
    )
