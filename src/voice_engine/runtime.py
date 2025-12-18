from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

from .config import VoiceEngineConfig
from .engine.vad import VadSegmenter
from .engine.stt import build_asr_adapter, Transcript


class BoundedQueue:
    """
    Thread-safe queue that enforces a maximum size and drops newest items
    when the queue is full. Each put/get supports a timeout so callers can
    periodically check for shutdown signals.
    """

    def __init__(self, maxsize: int, name: str) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._name = name

    def put(self, item: Any, timeout_ms: float) -> bool:
        """
        Attempt to enqueue ``item`` within the given timeout.

        Returns False when the queue is still full after ``timeout_ms``.
        Callers can log the drop and continue without blocking indefinitely.
        """
        timeout = timeout_ms / 1000.0
        try:
            self._queue.put(item, timeout=timeout)
            return True
        except queue.Full:
            return False

    def get(self, timeout_ms: float) -> tuple[bool, Any | None]:
        """
        Attempt to dequeue an item within ``timeout_ms`` milliseconds.

        Returns ``(False, None)`` when the queue is empty so consumers can
        poll again while respecting shutdown events.
        """
        timeout = timeout_ms / 1000.0
        try:
            item = self._queue.get(timeout=timeout)
            return True, item
        except queue.Empty:
            return False, None

    def qsize(self) -> int:
        """Return the current number of enqueued items (approximate)."""
        return self._queue.qsize()

    def name(self) -> str:
        """Human-readable queue name for logging."""
        return self._name


class VoiceEngineRuntime:
    """
    Orchestrates the always-listening pipeline (capture → VAD → inference → HTTP).

    This class owns the bounded queues, worker threads, and shutdown signals
    required to run the voice engine service.
    """

    def __init__(self, config: VoiceEngineConfig) -> None:
        """Loaded configuration describing audio, VAD, queue, and logging settings."""

        self.config = config
        self.frame_queue = BoundedQueue(
            maxsize=config.queues.frames, name="FrameQueue"
        )
        self.segment_queue = BoundedQueue(
            maxsize=config.queues.segments, name="SegmentQueue"
        )
        self.post_queue = BoundedQueue(
            maxsize=config.queues.posts, name="PostQueue"
        )
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.logger = logging.getLogger(__name__)
        level_name = config.logging.level.upper()
        level = getattr(logging, level_name, logging.INFO)
        self.logger.setLevel(level)

        self.vad_segmenter = VadSegmenter(
            cfg=self.config.vad,
            rate_hz=self.config.audio.rate_hz,
            frame_ms=self.config.audio.frame_ms,
            sample_width=self.config.audio.sample_width,
            channels=self.config.audio.channels,
        )
        self.asr_adapter = build_asr_adapter(self.config.asr, self.config.audio)


    def start(self) -> None:
        """Launch capture, VAD, inference, and HTTP worker threads."""
        self.logger.info("Voice engine runtime starting.")
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self.capture_loop, name="CaptureThread", daemon=True),
            threading.Thread(target=self.vad_loop, name="VADThread", daemon=True),
            threading.Thread(target=self.inference_loop, name="InferenceThread", daemon=True),
            threading.Thread(target=self.http_loop, name="HTTPThread", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        """Signal workers to exit and wait for them with a bounded timeout."""
        self.logger.info("Voice engine runtime stopping...")
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self.logger.info("Voice engine runtime stopped.")

    def should_stop(self) -> bool:
        """Return ``True`` when :meth:`stop` has been requested."""
        return self._stop.is_set()
    
    def _resolve_input_device(self, sd_module: Any) -> int | str | None:
        """
        Return the device identifier that ``sounddevice`` expects.

        The method accepts numeric IDs (``"2"``), friendly names (``"Jabra"``),
        or an empty value to fall back to the default input device. A best-effort
        lookup is performed so human-readable names select the correct hardware.
        """
        name = self.config.audio.device_name or ""  # Pull the raw config value (may be None/str).
        name = str(name).strip()  # Normalize to string so comparisons remain consistent.
        if not name:
            return None  # Empty means "use system default" per PortAudio semantics.
        try:
            return int(name)  # Numeric strings select an explicit device index.
        except ValueError:
            pass  # Non-numeric names will be resolved via enumeration.

        try:
            devices = sd_module.query_devices()  # Ask sounddevice for all known endpoints.
        except Exception as exc:  # pragma: no cover - hardware enumeration guard
            self.logger.warning("Unable to query audio devices: %s", exc)
            return name  # Fall back to the raw name so sounddevice can attempt its own match.

        lowered = name.lower()  # Case-insensitive matching for human-friendly names.
        for idx, info in enumerate(devices):
            if (
                lowered in info.get("name", "").lower()
                and info.get("max_input_channels", 0) >= self.config.audio.channels
            ):
                self.logger.info("Using audio input #%d: %s", idx, info.get("name"))
                return idx  # First match wins so behavior stays deterministic.

        self.logger.warning(
            "Audio device containing '%s' not found; using default input", name
        )
        return None


    # --- worker placeholders; replace with real logic in later steps ---

    def capture_loop(self) -> None:
        """
        Continuously pull PCM chunks from the selected device and enqueue frames.

        The loop keeps retrying when hardware errors occur so unplugging or
        replugging the microphone does not require restarting the process.
        """
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - dependency guard
            self.logger.error(
                "sounddevice is required for audio capture. Install it and restart: %s",
                exc,
            )
            self.stop()
            return

        cfg = self.config.audio  # Shorthand alias so the code below reads cleaner.
        frame_samples = int(cfg.rate_hz * cfg.frame_ms / 1000)  # Samples contained in one frame.
        frame_bytes = frame_samples * cfg.sample_width * cfg.channels  # Byte width per frame.
        chunk_frames = max(1, cfg.chunk_frames)  # Number of frames fetched per read.
        chunk_samples = frame_samples * chunk_frames  # Total samples requested per read call.
        dtype = "int16"  # RawInputStream expects dtype names, not struct formats.
        if cfg.sample_width != 2:
            self.logger.warning(
                "Unsupported sample_width=%d; forcing 16-bit capture", cfg.sample_width
            )  # sounddevice RawInputStream only supports 16-bit in this code path.
        device = self._resolve_input_device(sd)  # Map config name/index to actual device handle.

        while not self.should_stop():  # Keeps retrying if the stream fails mid-run.
            try:
                with sd.RawInputStream(
                    samplerate=cfg.rate_hz,
                    blocksize=chunk_samples,
                    dtype=dtype,
                    channels=cfg.channels,
                    device=device,
                ) as stream:  # Open the PortAudio stream against the resolved device.
                    self.logger.info("Audio capture started (device=%s)", device or "default")
                    while not self.should_stop():  # Inner loop processes chunks until stop is requested.
                        # Read a multi-frame chunk then split into frame-sized slices.
                        data, overflowed = stream.read(chunk_samples)
                        if overflowed:
                            self.logger.warning("Audio capture overflow (%s)", overflowed)
                        for offset in range(0, len(data), frame_bytes):
                            frame = data[offset : offset + frame_bytes]  # Slice out one fixed frame.
                            if len(frame) < frame_bytes:
                                break  # Ignore partially filled tail to keep timing consistent.
                            ok = self.frame_queue.put(frame, timeout_ms=5)  # Drop-newest policy if queue is full.
                            if not ok:
                                self.logger.warning(
                                    "[%s] drop (qsize=%d)",
                                    self.frame_queue.name(),
                                    self.frame_queue.qsize(),
                                )
            except Exception as exc:  # pragma: no cover - hardware failures
                self.logger.error("Audio capture error: %s", exc)  # Log and retry after brief backoff.
                time.sleep(1.0)


    def vad_loop(self) -> None:
        """Run energy-based VAD and enqueue completed speech segments."""
        while not self.should_stop():
            ok, frame = self.frame_queue.get(timeout_ms=50)
            if not ok:
                continue

            segments = self.vad_segmenter.process_frame(frame)
            for segment in segments:
                ok = self.segment_queue.put(segment, timeout_ms=5)
                if not ok:
                    self.logger.warning(
                        "[%s] drop (qsize=%d)",
                        self.segment_queue.name(),
                        self.segment_queue.qsize(),
                    )

        # Flush any partial speech when shutting down to avoid losing audio.
        for segment in self.vad_segmenter.flush():
            ok = self.segment_queue.put(segment, timeout_ms=5)
            if not ok:
                self.logger.warning(
                    "[%s] drop during flush (qsize=%d)",
                    self.segment_queue.name(),
                    self.segment_queue.qsize(),
                )

    def inference_loop(self) -> None:
        """Run STT on segments and enqueue normalized transcripts."""
        counter = 0
        while not self.should_stop():
            ok, segment = self.segment_queue.get(timeout_ms=50)
            if not ok:
                continue
            transcript: Transcript | None
            try:
                transcript = self.asr_adapter.transcribe(segment)
            except Exception as exc:  # pragma: no cover - protect thread
                self.logger.exception("ASR adapter error: %s", exc)
                continue

            if (
                transcript is None
                or transcript.confidence < self.config.asr.min_confidence
            ):
                continue

            payload = {
                "topic": "voice.transcript",
                "ts": int(time.time() * 1000),
                "text": transcript.text,
                "confidence": transcript.confidence,
                "source": transcript.source,
            }
            self.logger.info(
                "Transcript: %s (%.2f)", transcript.text, transcript.confidence
            )
            ok = self.post_queue.put(payload, timeout_ms=5)
            if not ok:
                self.logger.warning(
                    "[%s] drop (qsize=%d)",
                    self.post_queue.name(),
                    self.post_queue.qsize(),
                )
            counter += 1
            if counter % 10 == 0:
                self.logger.debug("Inference emitted %d transcripts", counter)


    def http_loop(self) -> None:
        """Placeholder HTTP loop; posts queued payloads to the gateway."""
        while not self.should_stop():
            ok, payload = self.post_queue.get(timeout_ms=50)
            if not ok:
                continue
            self.logger.info("HTTP POST payload: %s", payload)

    
