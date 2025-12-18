from voice_engine.config import AsrConfig, AudioConfig
from voice_engine.engine import stt


def test_stub_adapter_returns_none() -> None:
    adapter = stt.build_asr_adapter(
        AsrConfig(mode="stub"),
        AudioConfig(),
    )
    assert isinstance(adapter, stt.StubAdapter)
    assert adapter.transcribe(b"anything") is None


def test_invalid_mode_raises() -> None:
    cfg = AsrConfig(mode="does-not-exist")
    audio = AudioConfig()
    try:
        stt.build_asr_adapter(cfg, audio)
    except ValueError as exc:
        assert "Unsupported ASR mode" in str(exc)
    else:  # pragma: no cover - sanity guard
        raise AssertionError("ValueError expected")
