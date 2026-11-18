"""Exercise local assets, output ownership, cancellation, and conversation recovery."""

import asyncio
import base64
import io
import json
import time
import wave
from pathlib import Path
from typing import Any

import pytest
from palmimo_sdk import Palmimo

from palmimo_openai_live_decisions import audio, cli, runtime, songs
from palmimo_openai_live_decisions.audio import NullPlayback
from palmimo_openai_live_decisions.config import Settings, session_config
from palmimo_openai_live_decisions.events import EventLog
from palmimo_openai_live_decisions.policy import Reaction
from palmimo_openai_live_decisions.robot import Robot
from palmimo_openai_live_decisions.songs import Song, load_songbook


def write_songbook(root: Path, **changes: Any) -> Path:
    """Make a real PCM WAV and editable manifest without a recorded voice."""
    with wave.open(str(root / "hello.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x01\x00" * 1600)
    entry = {
        "id": "hello",
        "title": "Hello",
        "description": "A greeting song",
        "language": "Japanese",
        "file": "hello.wav",
        **changes,
    }
    path = root / "songs.json"
    path.write_text(json.dumps({"version": 1, "songs": [entry]}))
    return path


def test_songbook_preloads_pcm_and_constrains_choices(tmp_path: Path) -> None:
    book = load_songbook(write_songbook(tmp_path))
    assert book[0].seconds == 0.1
    config = session_config(Settings(songs=book), [])
    choices = config["delegation"]["decisions"]["questions"][0]["choices"]
    assert {"value": "sing_hello"} in choices
    assert "Japanese" in config["instructions"]
    (tmp_path / "hello.wav").unlink()
    assert len(book[0].pcm) == 3200
    assert load_songbook(None) == ()


@pytest.mark.parametrize("filename", ["../escape.wav", "/tmp/song.wav"])
def test_songbook_rejects_files_outside_pack(tmp_path: Path, filename: str) -> None:
    with pytest.raises(ValueError, match="inside"):
        load_songbook(write_songbook(tmp_path, file=filename))


def test_songbook_rejects_symlink_escape(tmp_path: Path) -> None:
    path = write_songbook(tmp_path)
    (tmp_path / "link.wav").symlink_to(tmp_path.parent / "elsewhere.wav")
    data = json.loads(path.read_text())
    data["songs"][0]["file"] = "link.wav"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="inside"):
        load_songbook(path)


def test_songbook_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = write_songbook(tmp_path)
    data = json.loads(path.read_text())
    data["songs"].append(data["songs"][0])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unique"):
        load_songbook(path)


def test_songbook_cli_validates_without_credentials_or_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_songbook(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PALMIMO_SONGBOOK", str(path))
    monkeypatch.setattr("sys.argv", ["palmimo-openai-live-decisions", "check-songs"])
    cli.main()
    assert json.loads(capsys.readouterr().out)[0]["seconds"] == 0.1


def test_output_interrupt_reaps_old_process_and_accepts_fresh_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    class Process:
        def __init__(self, command: list[str], **kwargs: Any) -> None:
            self.stdin = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

        def wait(self, timeout: float) -> int:
            assert self.returncode is not None
            return self.returncode

    monkeypatch.setattr("palmimo_sdk.resolve_alsa_device", lambda *args, **kwargs: "plughw:3,0")
    monkeypatch.setattr(audio.subprocess, "Popen", Process)
    player = audio.Playback("ReSpeaker")
    old_process = player.process
    old_thread = player.thread
    player.submit(b"\x01\x02" * 640)
    player.interrupt()
    try:
        assert old_process.poll() == 0 and not old_thread.is_alive()
        assert player.process is not old_process and player.stats()["audio_seconds"] == 0
        player.submit(b"\x03\x04")
        deadline = time.monotonic() + 2
        while player.stats()["pending_ms"] and time.monotonic() < deadline:
            time.sleep(0.01)
        player.check()
        assert player.process.stdin is not None
        assert player.process.stdin.getvalue() == b"\x03\x04"  # type: ignore[attr-defined]
    finally:
        player.close()


def test_songbook_rejects_wrong_audio_format(tmp_path: Path) -> None:
    path = write_songbook(tmp_path)
    with wave.open(str(tmp_path / "hello.wav"), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(44100)
        audio.writeframes(b"\x00" * 100)
    with pytest.raises(ValueError, match="mono"):
        load_songbook(path)


async def test_song_stream_is_paced_and_preserves_all_samples() -> None:
    song = Song("hello", "Hello", "Greeting", "English", b"\x01\x02" * 1600)
    packets: list[bytes] = []
    times: list[float] = []

    class Player:
        def submit(self, pcm: bytes) -> None:
            packets.append(pcm)
            times.append(time.monotonic())

    await songs.play_song(song, Player())
    assert b"".join(packets) == song.pcm
    assert max(map(len, packets)) <= 640
    assert times[-1] - times[0] >= 0.07


def delegation(action: str, identity: str = "d1") -> dict[str, Any]:
    """Build a complete native decision event."""
    return {
        "type": "decision.event",
        "delegation_id": identity,
        "event": {
            "type": "decision.completed",
            "decision": {"answers": [{"name": "action", "type": "choice", "choice": action, "confidence": 1.0}]},
        },
    }


class RecordingPlayer(NullPlayback):
    """Record output resets without opening an audio device."""

    def __init__(self) -> None:
        super().__init__()
        self.resets = 0

    def interrupt(self) -> None:
        self.resets += 1


async def song_session(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[runtime.Session, RecordingPlayer, list[dict[str, Any]], asyncio.Event]:
    """Skip only the generated introduction; exercise the actual song handoff."""
    song = Song("hello", "Hello", "Greeting", "Japanese", b"\x01\x02" * 16000)
    session = runtime.Session(Settings(songs=(song,)), Robot(Palmimo()), EventLog())
    player = RecordingPlayer()
    session.player = player
    sent: list[dict[str, Any]] = []
    playing = asyncio.Event()

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    async def announce(*args: Any, **kwargs: Any) -> bool:
        await session.dispatch({"type": "session.output_transcript.delta", "delta": "I'll sing."}, player)
        await session.dispatch(
            {"type": "session.output_audio.delta", "delta": base64.b64encode(b"\0\x08" * 320).decode()}, player
        )
        await asyncio.sleep(0.4)
        await session.dispatch(
            {"type": "session.output_audio.delta", "delta": base64.b64encode(b"\0\0" * 9600).decode()}, player
        )
        player.bytes = 0
        return True

    async def play(song: Song, output: Any) -> None:
        output.submit(song.pcm[:640])
        playing.set()
        await asyncio.Event().wait()

    session.send = send
    monkeypatch.setattr(session, "announce_action", announce)
    monkeypatch.setattr(runtime, "play_song", play)
    session.input_events = 1
    await session.delegated(delegation("sing_hello"))
    await asyncio.wait_for(playing.wait(), 3)
    return session, player, sent, playing


@pytest.mark.parametrize("stop", ["止めて", "Please stop singing", "やめて"])
async def test_spoken_stop_flushes_song_and_allows_conversation(monkeypatch: pytest.MonkeyPatch, stop: str) -> None:
    session, player, sent, _ = await song_session(monkeypatch)
    await session.dispatch({"type": "session.output_audio.delta", "delta": "AAA="}, player)
    assert player.bytes == 640
    # A song remains stoppable even after an operator has held autonomous motion.
    session.control.motion_paused = True
    await session.dispatch({"type": "session.input_transcript.delta", "delta": stop}, player)
    assert session.motion is not None
    await asyncio.wait_for(session.motion, 3)
    assert not session.singing and not session.song_output and player.resets == 1
    assert any("Resume normal conversation" in event.get("content", "") for event in sent)
    await session.dispatch({"type": "session.output_audio.delta", "delta": "AAA="}, player)
    assert player.bytes == 642
    await session.settle()


async def test_semantic_stop_handles_other_languages_and_prevents_duplicate_song(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, player, sent, _ = await song_session(monkeypatch)
    original = session.motion
    session.input_events += 1
    await session.delegated(delegation("sing_hello", "duplicate"))
    assert session.motion is original
    await session.delegated(delegation("stop", "french_stop"))
    assert session.motion is not None
    await asyncio.wait_for(session.motion, 3)
    assert player.resets == 1 and not session.singing
    assert any("Motion stopped" in event.get("content", "") for event in sent)
    await session.settle()


async def test_connection_cleanup_cancels_song_without_replaying_it(monkeypatch: pytest.MonkeyPatch) -> None:
    session, player, _, _ = await song_session(monkeypatch)
    await session.settle()
    assert not session.singing and player.resets == 1 and session.motion is None
    assert "cancelled" in session.conversation.history()[-1]["content"]


async def test_repeated_cancellation_cannot_skip_voice_restoration(monkeypatch: pytest.MonkeyPatch) -> None:
    session, player, sent, _ = await song_session(monkeypatch)
    cleanup_started, release = asyncio.Event(), asyncio.Event()

    async def react(reaction: Reaction) -> None:
        cleanup_started.set()
        await release.wait()

    monkeypatch.setattr(session.robot, "react", react)
    task = session.motion
    assert task is not None
    task.cancel()
    await asyncio.wait_for(cleanup_started.wait(), 2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release.set()
    await session.settle()
    assert player.resets == 1 and not session.singing
    assert any("Resume normal conversation" in event.get("content", "") for event in sent)


@pytest.mark.parametrize("fail", [False, True])
async def test_finished_song_and_playback_failure_both_restore_voice(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    session, player, sent, _ = await song_session(monkeypatch)
    await session.settle()

    async def play(song: Song, output: Any) -> None:
        if fail:
            raise RuntimeError("simulated playback failure")
        output.submit(b"\x01\x02")

    monkeypatch.setattr(runtime, "play_song", play)
    await session.sing(session.settings.songs[0], "next")
    assert not session.singing and not session.song_output and player.resets == 2
    assert ("playback_failed" if fail else "completed") in session.conversation.history()[-1]["content"]
    assert any("Resume normal conversation" in event.get("content", "") for event in sent)
