# voice_engine

Always-listening audio pipeline for the Roomba voice stack. Runs on a Raspberry Pi, captures PCM from the Jabra speakerphone, performs Voice Activity Detection (VAD) + keyword/Speech-To-Text (STT) inference, and POSTs canonical JSON (`voice.speaker`, `voice.transcript`) to the Roomba gateway (`http://127.0.0.1:8765/`).

## Minimal Viable Product (MVP) Scope
- Capture thread (ALSA/PortAudio) → bounded `q_frames`
- VAD worker → `q_segments`
- Inference worker → `q_posts` (emit `voice.transcript` first; speaker-ID later)
- HTTP worker with short timeouts + back-off (drop-newest on overflow)
- Config-driven (YAML) audio/VAD/ASR/gateway settings

## Next steps
1. ROOMB-11/VOICEENG-001: mic loop + queue skeleton
2. ROOMB-12/VOICEENG-002: VAD segmenter implementation
3. ROOMB-13/15: STT/KWS integration (local engine)
4. ROOMB-16: HTTP robustness & back-pressure
5. ROOMB-18/19: config/logging + systemd unit

Speaker recognition integration will reuse the existing models repo and be layered on once transcript POSTs are solid.

## Run
```bash
python -m voice_engine --config config/voice_engine.example.yaml
```
Before running on the Pi:
1. Configure audio defaults: `tools/audio_jabra_default.sh`
2. Download a Vosk model (e.g., `vosk-model-small-en-us-0.15`) and point `asr.model_path` to it in the YAML config.

Set `asr.mode` to `vosk` for full STT or `stub` to disable transcription during early bring-up. The inference worker now publishes real transcripts (subject to the configured `min_confidence`) to the HTTP worker queue.
