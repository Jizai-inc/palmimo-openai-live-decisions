"""Exercise playback recovery with an ordered WebSocket and real speaker workers."""

import asyncio
import base64
import io
import json
import threading
from collections.abc import Buffer
from typing import Any

import pytest
from audio_fakes import acknowledge_startup
from test_recovery import RecordingLog, TestRobot
from websockets.asyncio.server import ServerConnection, serve

from palmimo_openai_live_decisions import audio, runtime
from palmimo_openai_live_decisions.config import Settings
from palmimo_openai_live_decisions.recovery import RetryableSessionError


def delta(pcm: bytes) -> str:
    return json.dumps({"type": "session.output_audio.delta", "delta": base64.b64encode(pcm).decode()})


@pytest.mark.parametrize("fault,packets", [("queue_limit", 120), ("speaker_write", 1), ("speaker_pipe", 12)])
async def test_next_reply_plays_on_same_connection_after_backlog(
    monkeypatch: pytest.MonkeyPatch, fault: str, packets: int
) -> None:
    stop = asyncio.Event()
    robot, log = TestRobot(), RecordingLog()
    processes: list[Any] = []
    players: list[audio.Playback] = []
    fresh = b"\x45\0" * 320
    stale = b"\x11\0" * 320

    class Pipe(io.BytesIO):
        def __init__(self, index: int, terminated: threading.Event) -> None:
            super().__init__()
            self.index = index
            self.terminated = terminated
            self.played = bytearray()

        def write(self, data: Buffer, /) -> int:
            pcm = bytes(data)
            if self.index == 0 and fault != "speaker_pipe":
                # One case consumes in real time; the other wedges on a single
                # packet, so the queue never has to fill to detect the stall.
                delay = len(pcm) / 32000 if fault == "queue_limit" else 5.0
                if self.terminated.wait(delay):
                    raise BrokenPipeError("old speaker closed")
            self.played.extend(pcm)
            return len(pcm)

    class Process:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.terminated = threading.Event()
            self.stdin = Pipe(len(processes), self.terminated)
            self.stderr = io.BytesIO()
            processes.append(self)

        def poll(self) -> int | None:
            return 0 if self.terminated.is_set() else None

        def terminate(self) -> None:
            self.terminated.set()

        def wait(self, timeout: float) -> int:
            assert self.terminated.wait(timeout)
            return 0

    def speaker(device: str) -> audio.Playback:
        player = audio.Playback(device, max_seconds=1.5 if fault == "queue_limit" else 0.12)
        players.append(player)
        return player

    async def server(ws: ServerConnection) -> None:
        assert json.loads(await ws.recv())["type"] == "session.start"
        await ws.send(json.dumps({"type": "session.started", "session": {"id": "same-conversation"}}))
        await acknowledge_startup(ws)
        # Wait until real microphone traffic starts.
        assert json.loads(await ws.recv())["type"] == "session.input_audio.append"
        for _ in range(packets):
            await ws.send(delta(stale))
            if fault == "speaker_pipe":
                # The producer writes promptly, but the reader consumes nothing.
                await asyncio.sleep(0.01)
        marker = ""
        async for raw in ws:
            event = json.loads(raw)
            if event["type"] == "session.close":
                await ws.send(json.dumps({"type": "session.closed", "reason": "close_requested"}))
                return
            if event["type"] == "session.update":
                assert not marker
                marker = event["event_id"]
                async with asyncio.timeout(2):
                    while len(processes) != 2:
                        await asyncio.sleep(0.005)
                await ws.send(json.dumps({"type": "session.updated", "client_event_id": "unrelated"}))
                await ws.send(delta(stale))
                await ws.send(json.dumps({"type": "session.input_transcript.delta", "delta": "Still listening?"}))
            elif marker and event["type"] == "session.input_audio.append":
                # The receiver still processes conversation while output is paused.
                if not any(e.get("text") == "Still listening?" for e in log.events):
                    continue
                assert not processes[1].stdin.played
                await ws.send(json.dumps({"type": "session.updated", "client_event_id": marker}))
                async with asyncio.timeout(2):
                    while not any(e["kind"] == "playback_resumed" for e in log.events):
                        await asyncio.sleep(0.005)
                await ws.send(delta(fresh))
                async with asyncio.timeout(2):
                    while not processes[1].stdin.played:
                        await asyncio.sleep(0.005)
                stop.set()
                marker = ""

    monkeypatch.setattr("palmimo_sdk.resolve_alsa_device", lambda *a, **kw: "test")
    monkeypatch.setattr(audio.subprocess, "Popen", Process)
    monkeypatch.setattr(runtime, "Playback", speaker)
    if fault == "speaker_pipe":
        monkeypatch.setattr(
            audio.Playback, "_pipe_bytes", lambda p: len(processes[0].stdin.played) if p.process is processes[0] else 0
        )
    async with serve(server, "127.0.0.1", 0) as listener:
        monkeypatch.setattr(runtime, "LIVE_URL", f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}")
        async with asyncio.timeout(8):
            await runtime.run(Settings(idle_enabled=False), "test", robot, log, stop, asyncio.Queue())  # type: ignore[arg-type]
    assert len(players) == 1 and len(processes) == 2
    assert processes[1].stdin.played == fresh
    assert all(p.terminated.is_set() for p in processes)
    assert not players[0].thread.is_alive() and not players[0].monitor.is_alive()
    assert robot.connects == robot.closes == 1 and robot.stops == 0
    stalls = [e for e in log.events if e["kind"] == "playback_stalled"]
    assert len(stalls) == 1 and stalls[0]["reason"] == fault
    assert stalls[0]["pending_ms"] <= 1500
    if fault == "speaker_pipe":
        assert stalls[0]["pending_ms"] == 0 and stalls[0]["pipe_pending_ms"] >= 100
    resumed = [e for e in log.events if e["kind"] == "playback_resumed"]
    assert len(resumed) == 1 and resumed[0]["discarded_stream_ms"] >= 20
    assert not any(e["kind"] in {"reconnected", "connection_lost"} for e in log.events)


async def test_early_ack_never_enables_output_while_old_worker_is_closing() -> None:
    release = threading.Event()

    class Player(audio.NullPlayback):
        def interrupt(self) -> None:
            assert release.wait(3)

        def stats(self) -> dict[str, Any]:
            return {}

    session = runtime.Session(Settings(), TestRobot(), RecordingLog())  # type: ignore[arg-type]
    player = Player()
    session.player = player
    sent = []

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    session.send = send
    await session.recover_playback("queue_limit")
    recovery = session.playback_recovery
    assert recovery is not None
    try:
        await session.dispatch({"type": "session.updated", "client_event_id": sent[-1]["event_id"]}, player)
        await session.dispatch(json.loads(delta(b"\x21\0")), player)
        assert player.bytes == 0 and not recovery.ready.is_set()
    finally:
        release.set()
        await recovery.reset
    await session.dispatch(json.loads(delta(b"\x22\0")), player)
    assert player.bytes == 2 and recovery.ready.is_set()


async def test_recurring_backlog_requests_reconnect() -> None:
    log = RecordingLog()
    session = runtime.Session(Settings(), TestRobot(), log)  # type: ignore[arg-type]

    class Player(audio.NullPlayback):
        def stats(self) -> dict[str, Any]:
            return {}

    async def send(event: dict[str, Any]) -> None:
        pass

    session.send, session.player = send, Player()
    await session.recover_playback("queue_limit")
    recovery = session.playback_recovery
    assert recovery is not None
    await recovery.reset
    await session.dispatch({"type": "session.updated", "client_event_id": recovery.marker}, session.player)
    with pytest.raises(RetryableSessionError, match="again"):
        await session.recover_playback("queue_limit")


async def test_stop_joins_reset_before_releasing_the_speaker(monkeypatch: pytest.MonkeyPatch) -> None:
    stop = asyncio.Event()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    robot, log = TestRobot(), RecordingLog()
    closes = []

    class Player(audio.NullPlayback):
        def __init__(self, device: str) -> None:
            super().__init__()

        def submit(self, pcm: bytes) -> None:
            raise audio.PlaybackBacklog("synthetic backlog")

        def interrupt(self) -> None:
            entered.set()
            assert release.wait(5)
            finished.set()

        def stats(self) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            assert finished.is_set()
            closes.append(True)

    async def server(ws: ServerConnection) -> None:
        assert json.loads(await ws.recv())["type"] == "session.start"
        await ws.send(json.dumps({"type": "session.started", "session": {"id": "closing"}}))
        await acknowledge_startup(ws)
        assert json.loads(await ws.recv())["type"] == "session.input_audio.append"
        await ws.send(delta(b"\x21\0"))
        async for raw in ws:
            if json.loads(raw)["type"] == "session.close":
                await ws.send(json.dumps({"type": "session.closed", "reason": "close_requested"}))
                return

    monkeypatch.setattr(runtime, "Playback", Player)
    async with serve(server, "127.0.0.1", 0) as listener:
        monkeypatch.setattr(runtime, "LIVE_URL", f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}")
        task = asyncio.create_task(runtime.run(Settings(idle_enabled=False), "test", robot, log, stop, asyncio.Queue()))  # type: ignore[arg-type]
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            stop.set()
            async with asyncio.timeout(2):
                while robot.mic.subscribers:
                    await asyncio.sleep(0.005)
            assert not task.done() and not closes
        finally:
            stop.set()
            release.set()
            await asyncio.wait_for(task, 4)
    assert closes == [True] and robot.closes == 1
    assert not any(e["kind"] in {"reconnect_wait", "reconnected"} for e in log.events)
