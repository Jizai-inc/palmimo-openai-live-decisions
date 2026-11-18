"""Model the continuous Live PCM stream at the actual song-introduction boundary."""

import asyncio
import base64
import io
import subprocess
import threading
from typing import Any

import pytest
from palmimo_sdk import Palmimo

from palmimo_openai_live_decisions import audio
from palmimo_openai_live_decisions.audio import NullPlayback
from palmimo_openai_live_decisions.config import Settings
from palmimo_openai_live_decisions.events import EventLog
from palmimo_openai_live_decisions.robot import Robot
from palmimo_openai_live_decisions.runtime import Session
from palmimo_openai_live_decisions.songs import Song, SongIntroduction


class IntroLog(EventLog):
    """Capture state transitions without generating conversation logs."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.events: list[dict[str, Any]] = []

    def write(self, kind: str, **fields: Any) -> None:
        self.events.append({"kind": kind, **fields})
        if kind == "song_started":
            self.started.set()


def test_intro_requires_text_and_actual_audio_not_initial_silence() -> None:
    now = 1.0
    intro = SongIntroduction(lambda: now)
    intro.observe_audio(b"\0\0" * 16000)
    assert intro.waiting_for() == "announcement_text"
    intro.observe_text("I'll sing.")
    now += 0.5
    assert intro.waiting_for() == "audible_onset"
    intro.observe_audio(b"\0\x08" * 320)
    assert intro.waiting_for() == "speech_end"
    now += 0.8
    assert intro.waiting_for() == "fresh_audio"
    intro.observe_audio(b"\0\0" * 9600)
    assert intro.waiting_for() is None


def test_intro_keeps_quiet_speech_and_restarts_silence_measurement() -> None:
    now = 1.0
    intro = SongIntroduction(lambda: now)
    intro.observe_text("One short introduction.")
    # Convert before abs so the negative int16 limit cannot be mistaken for silence.
    intro.observe_audio(b"\0\x80" * 320)
    intro.observe_audio(b"\0\0" * 9500)
    now += 0.5
    intro.observe_audio(b"\x40\0" * 320)
    assert intro.waiting_for() == "speech_end"
    intro.observe_audio(b"\x01\0" * 9600)
    assert intro.waiting_for() is None
    intro.observe_text("Another fragment.")
    assert intro.waiting_for() == "speech_end"
    assert intro.stats()["audible_seconds"] == 0.04


def test_intro_rejects_malformed_pcm_and_ignores_empty_packets() -> None:
    intro = SongIntroduction()
    intro.observe_audio(b"")
    assert intro.packets == 0 and intro.last_packet is None
    with pytest.raises(ValueError, match="incomplete"):
        intro.observe_audio(b"\0")


def test_drain_plays_all_queued_audio_before_reopening_speaker(monkeypatch: pytest.MonkeyPatch) -> None:
    class Pipe(io.BytesIO):
        def __init__(self) -> None:
            super().__init__()
            self.eof = threading.Event()
            self.played = b""

        def close(self) -> None:
            if not self.closed:
                self.played = self.getvalue()
                super().close()
                self.eof.set()

    class Process:
        def __init__(self, command: list[str], **kwargs: Any) -> None:
            self.stdin = Pipe()
            self.stderr = io.BytesIO()
            self.returncode: int | None = None
            self.terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = 0

        def wait(self, timeout: float) -> int:
            if self.returncode is None and not self.stdin.eof.wait(timeout):
                raise subprocess.TimeoutExpired("aplay", timeout)
            self.returncode = 0
            return 0

    monkeypatch.setattr("palmimo_sdk.resolve_alsa_device", lambda *args, **kwargs: "plughw:3,0")
    monkeypatch.setattr(audio.subprocess, "Popen", Process)
    player = audio.Playback("ReSpeaker")
    old = player.process
    pcm = b"\0\x08" * 320 + b"\0\0" * 9600
    try:
        player.submit(pcm)
        player.drain()
        assert old.stdin is not None
        assert old.stdin.played == pcm and not old.terminated  # type: ignore[attr-defined]
        assert player.process is not old and player.stats()["pending_ms"] == 0
        player.check()
    finally:
        player.close()


async def test_song_starts_when_silent_audio_packets_keep_arriving() -> None:
    song = Song("hello", "Hello", "Greeting", "English", b"\x15\0" * 1600)
    log = IntroLog()
    session = Session(Settings(songs=(song,), action_lead_seconds=0), Robot(Palmimo(), motion_enabled=False), log)
    player = NullPlayback()
    session.player = player
    announced = asyncio.Event()

    async def send(event: dict[str, Any]) -> None:
        if event.get("content", "").startswith("Action announcement:"):
            announced.set()

    async def continuous_output() -> None:
        await announced.wait()
        await session.dispatch({"type": "session.output_transcript.delta", "delta": "I'll sing a song."}, player)
        packets = 0
        while True:
            pcm = (b"\x00\x08" if packets < 10 else b"\0\0") * 320
            await session.dispatch(
                {"type": "session.output_audio.delta", "delta": base64.b64encode(pcm).decode()}, player
            )
            packets += 1
            await asyncio.sleep(0.02)

    session.send = send
    session.replace_motion(lambda: session.sing(song, "request"), origin="song")
    output = asyncio.create_task(continuous_output())
    try:
        await asyncio.wait_for(log.started.wait(), 3)
        assert session.motion is not None
        await asyncio.wait_for(session.motion, 2)
        assert any(e["kind"] == "song_ended" and e["result"] == "completed" for e in log.events)
    finally:
        output.cancel()
        await asyncio.gather(output, return_exceptions=True)
        await session.settle()


@pytest.mark.parametrize("stop_at", ["introduction", "drain"])
async def test_stop_before_music_settles_output_and_restores_conversation(stop_at: str) -> None:
    class HoldingPlayer(NullPlayback):
        def __init__(self) -> None:
            super().__init__()
            self.draining = threading.Event()
            self.release = threading.Event()
            self.resets = 0

        def drain(self) -> None:
            self.draining.set()
            assert self.release.wait(3)
            self.draining.clear()

        def interrupt(self) -> None:
            assert not self.draining.is_set()
            self.resets += 1

    song = Song("hello", "Hello", "Greeting", "English", b"\x15\0" * 1600)
    log = IntroLog()
    session = Session(Settings(songs=(song,), action_lead_seconds=0), Robot(Palmimo(), motion_enabled=False), log)
    player = HoldingPlayer()
    session.player = player
    announced = asyncio.Event()
    sent: list[dict[str, Any]] = []

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)
        if event.get("content", "").startswith("Action announcement:"):
            announced.set()

    session.send = send
    session.replace_motion(lambda: session.sing(song, "request"), origin="song")
    song_task = session.motion
    try:
        await asyncio.wait_for(announced.wait(), 1)
        await session.dispatch({"type": "session.output_transcript.delta", "delta": "I'll sing."}, player)
        await session.dispatch(
            {"type": "session.output_audio.delta", "delta": base64.b64encode(b"\0\x08" * 320).decode()}, player
        )
        if stop_at == "drain":
            await asyncio.sleep(0.5)
            await session.dispatch(
                {"type": "session.output_audio.delta", "delta": base64.b64encode(b"\0\0" * 9600).decode()},
                player,
            )
            assert await asyncio.to_thread(player.draining.wait, 1)
        await session.dispatch({"type": "session.input_transcript.delta", "delta": "Stop singing"}, player)
        if stop_at == "drain":
            await asyncio.sleep(0.05)
            assert song_task is not None and not song_task.done() and player.resets == 0
            # Repeated cancellation must not orphan the speaker drain thread.
            song_task.cancel()
            player.release.set()
        assert session.motion is not None
        await asyncio.wait_for(session.motion, 2)
        assert not log.started.is_set() and not session.singing and not session.song_output
        assert session.output_drain is None and session.song_introduction is None and player.resets == 1
        assert any("Resume normal conversation" in event.get("content", "") for event in sent)
        previous = player.bytes
        await session.dispatch({"type": "session.output_audio.delta", "delta": "AAA="}, player)
        assert player.bytes == previous + 2
    finally:
        player.release.set()
        await session.settle()
