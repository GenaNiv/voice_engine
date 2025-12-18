"""Lightweight voice activity detection (VAD) segmenter.

The segmenter consumes fixed-size PCM frames, estimates the per-frame energy,
and aggregates consecutive speech frames into byte segments. The goal is to
produce deterministic, bounded-latency chunks for downstream keyword spotting
or speech-to-text engines without blocking any capture thread.
"""

import numpy as np

from ..config import VadConfig


class VadSegmenter:
    """Stateful, energy-based VAD with hangover and min/max durations.

    The component is intentionally simple: every PCM frame is converted to RMS
    energy, compared against a static threshold, and then either appended to a
    growing speech buffer or treated as silence. This deterministic approach
    keeps CPU costs low and makes the resulting behavior easy to reason about
    when tuning thresholds in the field.

    Parameters
    ----------
    cfg:
        Full VAD configuration (thresholds, frame duration, hangover, etc.).
    sample_width:
        Size of an individual PCM sample in bytes (defaults to 16-bit audio).
    channels:
        Number of interleaved channels (defaults to mono).
    """

    def __init__(
        self,
        cfg: VadConfig,
        rate_hz: int,
        frame_ms: int,
        sample_width: int = 2,
        channels: int = 1,
    ):
        """Initialize the segmenter and pre-compute byte dimensions.

        cfg:
            Parsed VAD configuration including frame duration and thresholds.
        rate_hz:
            Sample rate of the PCM stream (e.g., 16000 for 16 kHz audio).
        frame_ms:
            Duration (milliseconds) of each capture frame fed into ``process_frame``.
        sample_width:
            PCM sample width in bytes; the default of 2 matches 16-bit audio.
        channels:
            Number of PCM channels; mono capture is assumed in the default.
        """
        self.cfg = cfg
        self.frame_ms = frame_ms
        self.sample_width = sample_width
        self.channels = channels
        self.bytes_per_ms = rate_hz * sample_width * channels // 1000
        self._current = bytearray()
        self._speech_ms = 0
        self._silence_ms = 0
        self._in_speech = False

    def _frame_energy_db(self, frame: bytes) -> float:
        """Return the RMS energy of a frame in decibels.

        A logarithmic scale matches human perception and gives us a consistent
        threshold across ambient environments. A zero-length frame is treated
        as negative infinity to ensure it never crosses the speech threshold.
        """
        if not frame:
            return -float("inf")
        samples = np.frombuffer(frame, dtype=np.int16)
        rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        # Add epsilon to avoid log(0) and keep gradients finite.
        return 20 * np.log10(rms + 1e-9)

    def process_frame(self, frame: bytes) -> list[bytes]:
        """Consume a frame and return zero or more completed speech segments.

        The method maintains internal speech/silence counters. When energy rises
        above the configured threshold, frames are appended to the current speech
        buffer. When energy falls below the threshold, hangover logic determines
        if the speech segment should be emitted or discarded. The method never
        blocks; all state is updated synchronously so callers can safely run it
        on the capture thread.

        Parameters
        ----------
        frame:
            Raw PCM bytes with the size implied by cfg.frame_ms. Any other size
            is accepted, but the timing math assumes fixed-size frames.
        """
        segments: list[bytes] = []
        energy_db = self._frame_energy_db(frame)

        if energy_db >= self.cfg.energy_thresh_db:
            # Speech onset: reset counters the moment energy crosses threshold.
            # We treat the first supra-threshold frame as the start of a new
            # utterance so previous residual bytes do not leak into it.
            if not self._in_speech:
                self._reset_state(clear_current=True)
                self._in_speech = True

            # Track how long the user has been speaking and accumulate the PCM
            # bytes in a temporary buffer. Any silence budget is cleared here so
            # hangover counting only begins once energy dips back down.
            self._speech_ms += self.frame_ms
            self._current.extend(frame)
            self._silence_ms = 0

            # Hard stop: cap segments so they never exceed max_speech_ms even if
            # a user keeps talking. This keeps downstream buffers bounded.
            if self._speech_ms >= self.cfg.max_speech_ms:
                # Only emit if we already met the minimum duration; otherwise
                # discard as noise.
                if self._speech_ms >= self.cfg.min_speech_ms:
                    segments.append(bytes(self._current))
                self._reset_state(clear_current=True)
                self._in_speech = False
        else:
            if self._in_speech:
                # Track contiguous silence to implement hangover windows.
                # Hanging on to a few silent frames prevents chopping words
                # prematurely when a speaker briefly pauses mid-sentence.
                self._silence_ms += self.frame_ms
                if self._silence_ms >= self.cfg.hangover_ms:
                    # Once silence exceeds hangover, decide whether the buffered
                    # speech was long enough to emit; short bursts are dropped.
                    if self._speech_ms >= self.cfg.min_speech_ms:
                        segments.append(bytes(self._current))
                    self._reset_state(clear_current=True)
                    self._in_speech = False

        return segments

    def flush(self) -> list[bytes]:
        """Emit any partial speech when shutting down the pipeline.

        This complements :meth:`process_frame` by ensuring that an ongoing
        utterance is not lost if the runtime stops capturing midway through a
        word. Because the runtime thread stops immediately after calling
        ``flush()``, the method never leaves ``_in_speech`` set to ``True``.
        """
        if self._in_speech and self._speech_ms >= self.cfg.min_speech_ms:
            segment = bytes(self._current)
            self._reset_state(clear_current=True)
            self._in_speech = False
            return [segment]
        self._reset_state(clear_current=True)
        self._in_speech = False
        return []

    def _reset_state(self, clear_current: bool = False) -> None:
        """Clear internal counters and optionally drop buffered audio."""
        if clear_current:
            self._current.clear()
        self._speech_ms = 0
        self._silence_ms = 0
