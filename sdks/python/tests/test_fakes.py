"""Deterministic fakes and manifest shape."""

from voicekit_sdk.fakes import (
    FAKE_FRAME_SAMPLES,
    FAKE_LLM_DELTAS,
    FAKE_REPLY,
    FAKE_TRANSCRIPT,
    FAKE_TTS_FRAMES,
    FakeLlm,
    FakeStt,
    FakeTts,
    FakeVad,
    fake_pcm,
)


def test_fake_transcript_and_deltas_are_fixed():
    stt = FakeStt()
    chunks = list(stt.transcribe(b"\x00\x00"))
    assert chunks[-1] == {"text": FAKE_TRANSCRIPT, "final": True}
    assert chunks[0]["final"] is False

    llm = FakeLlm()
    deltas = list(llm.generate("hello world"))
    assert tuple(deltas) == FAKE_LLM_DELTAS
    assert "".join(deltas) == FAKE_REPLY


def test_fake_tts_is_deterministic():
    tts = FakeTts()
    first = list(tts.synthesize(FAKE_REPLY))
    second = list(tts.synthesize(FAKE_REPLY))
    assert first == second
    assert len(first) == FAKE_TTS_FRAMES
    assert all(len(chunk) == FAKE_FRAME_SAMPLES * 2 for chunk in first)
    assert first[0] != first[1]
    assert first[0] == fake_pcm(0)


def test_fake_vad_reports_start_then_end():
    vad = FakeVad()
    events = list(vad.analyze(b"\x00\x00" * 160))
    assert [e["event"] for e in events] == ["speech_start", "speech_end"]
    assert events[0]["timestamp_ms"] == 0
    assert events[1]["timestamp_ms"] > 0


def test_all_manifests_are_selectable():
    for provider, modality in [
        (FakeStt(), "stt"),
        (FakeLlm(), "llm"),
        (FakeTts(), "tts"),
        (FakeVad(), "vad"),
    ]:
        manifest = provider.manifest
        assert manifest["name"]
        assert manifest["modality"] == modality
        assert isinstance(manifest["capabilities"], list)
        assert isinstance(manifest["config_schema"], dict)
