"""Exercise capture gaps and transport backpressure through the real runtime."""

import asyncio
import base64
import json
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import pytest
from audio_fakes import FakeMic
from test_recovery import RecordingLog, TestRobot
from websockets.asyncio.server import ServerConnection, serve

from palmimo_openai_live_decisions import runtime
from palmimo_openai_live_decisions.config import Settings
from palmimo_openai_live_decisions.input_audio import AudioInput, CaptureRecovery


async def test_capture_continues_during_event_loop_pause_and_drops_stale_audio() -> None:
    mic = FakeMic()
    capture = AudioInput(mic, buffer_seconds=0.05)
    capture.start()
    try:
        await asyncio.sleep(0.04)
        before = capture.stats()["captured_chunks"]
        time.sleep(0.2)
        stats = capture.stats()
        assert stats["captured_chunks"] > before + 5
        assert stats["capture_age_seconds"] < 0.05
        assert stats["queued_ms"] <= 50
        assert stats["dropped_chunks"] > 0
        mic.stuck = True
        await asyncio.sleep(0.1)
        assert capture.take() is None
    finally:
        await asyncio.to_thread(capture.close)
    assert mic.subscribers == 0 and not capture.thread.is_alive()


async def test_capture_reopen_budget_and_device_failure_are_terminal() -> None:
    mic = FakeMic()
    recovery = CaptureRecovery()
    stop = asyncio.Event()
    assert await recovery.restart(mic, stop)
    assert await recovery.restart(mic, stop)
    with pytest.raises(RuntimeError, match="exhausted"):
        await recovery.restart(mic, stop)
    assert mic.opens == mic.closes == 2
    mic.fail_open = True
    with pytest.raises(RuntimeError, match="reopen failed"):
        await CaptureRecovery().restart(mic, stop)


async def test_audio_recovery_does_not_retry_failed_motion_hardware() -> None:
    robot = TestRobot()

    async def failed_stop() -> str:
        raise OSError("Synthetic servo failure")

    robot.stop = failed_stop  # type: ignore[method-assign]
    session = runtime.Session(Settings(), robot, RecordingLog())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="Motion hold failed"):
        await session.hold_for_input("capture")
    assert session.control.motion_paused


@pytest.mark.parametrize("cancel", [False, True])
async def test_stop_during_reopen_settles_worker_and_does_not_reopen(cancel: bool) -> None:
    mic = FakeMic()
    entered, release = threading.Event(), threading.Event()

    def close() -> None:
        entered.set()
        assert release.wait(2)

    mic.close = close  # type: ignore[method-assign]
    stop = asyncio.Event()
    task = asyncio.create_task(CaptureRecovery().restart(mic, stop))
    assert await asyncio.to_thread(entered.wait, 1)
    if cancel:
        task.cancel()
        await asyncio.sleep(0)
    else:
        stop.set()
    release.set()
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert not await task
    assert mic.opens == 0


@pytest.mark.parametrize(
    "mode",
    [
        "capture_short",
        "capture_restart",
        "send_short",
        "send_restart",
        "event_loop",
        "stop",
        "stop_send",
        "device_failure",
    ],
)
async def test_audio_stalls_recover_without_replaying_motion_or_losing_quiet(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    robot, log = TestRobot(), RecordingLog()
    stop = asyncio.Event()
    configs: list[dict[str, Any]] = []
    players: list[Any] = []
    received = 0
    injected = False
    samples_before_gap = 0

    class Player:
        def __init__(self, device: str) -> None:
            self.bytes = 0
            self.closed = False
            players.append(self)

        def submit(self, pcm: bytes) -> None:
            self.bytes += len(pcm)

        def check(self) -> None:
            pass

        def stats(self) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            self.closed = True

    async def server(ws: ServerConnection) -> None:
        nonlocal received, injected, samples_before_gap
        start = json.loads(await ws.recv())
        configs.append(start["session"])
        index = len(configs)
        await ws.send(json.dumps({"type": "session.started", "session": {"id": f"live_{index}"}}))
        if index == 1:
            await ws.send(json.dumps({"type": "session.input_transcript.delta", "delta": "Keep speaking English."}))
        async for raw in ws:
            event = json.loads(raw)
            if event["type"] == "session.close":
                await ws.send(json.dumps({"type": "session.closed", "reason": "close_requested"}))
                return
            if event["type"] != "session.input_audio.append":
                continue
            received += 1
            if not injected and received >= 3:
                injected = True
                samples_before_gap = robot.mic.blocks
                if mode in {"capture_restart", "stop"}:
                    robot.mic.stuck = True
                elif mode == "capture_short":
                    robot.mic.gap_until = time.monotonic() + 0.25
                elif mode == "device_failure":
                    robot.mic.broken = True
                elif mode == "event_loop":
                    time.sleep(0.25)
            recovered = any(e["kind"] == "input_recovered" for e in log.events) or index > 1
            if recovered:
                await ws.send(
                    json.dumps({"type": "session.output_audio.delta", "delta": base64.b64encode(b"\1\0" * 32).decode()})
                )
                await ws.send(json.dumps({"type": "session.output_transcript.delta", "delta": "I can hear you again."}))
                async with asyncio.timeout(2):
                    while players[-1].bytes == 0:
                        await asyncio.sleep(0.01)
                stop.set()

    real_connect = runtime.websockets.connect

    @asynccontextmanager
    async def connect(*args: Any, **kwargs: Any) -> Any:
        async with real_connect(*args, **kwargs) as ws:
            original = ws.send
            sends = 0

            async def send(message: Any, **options: Any) -> None:
                nonlocal sends
                if json.loads(message).get("type") == "session.input_audio.append":
                    sends += 1
                    if (mode.startswith("send") or mode == "stop_send") and len(configs) == 1 and sends == 4:
                        await asyncio.sleep(0.25 if mode == "send_short" else 2)
                await original(message, **options)

            ws.send = send  # type: ignore[method-assign]
            yield ws

    original_run = runtime.run_connection

    async def connection(ws: Any, session: runtime.Session, *args: Any) -> None:
        if len(configs) == 1:
            session.policy.quiet = True
        await original_run(ws, session, *args)

    async def stopper() -> None:
        if mode not in {"stop", "stop_send"}:
            return
        async with asyncio.timeout(2):
            while not any(e["kind"] == "input_stalled" for e in log.events):
                await asyncio.sleep(0.01)
        stop.set()

    monkeypatch.setattr(runtime, "Playback", Player)
    monkeypatch.setattr(runtime.websockets, "connect", connect)
    monkeypatch.setattr(runtime, "run_connection", connection)
    async with serve(server, "127.0.0.1", 0) as listener:
        port = listener.sockets[0].getsockname()[1]
        monkeypatch.setattr(runtime, "LIVE_URL", f"ws://127.0.0.1:{port}")
        settings = Settings(idle_enabled=False, stream_stall_seconds=0.08, audio_recovery_seconds=0.4)
        async with asyncio.timeout(8):
            task = asyncio.gather(runtime.run(settings, "test", robot, log, stop, asyncio.Queue()), stopper())  # type: ignore[arg-type]
            if mode == "device_failure":
                with pytest.raises(RuntimeError, match="Microphone capture device failed"):
                    await task
            else:
                await task
    assert len(configs) == (2 if mode.endswith("restart") else 1)
    assert robot.closes == 1
    if mode != "device_failure":
        assert robot.stops >= 1
    assert robot.mic.subscribers == 0
    assert robot.mic.opens == (1 if mode == "capture_restart" else 0)
    assert all(p.closed for p in players)
    if mode not in {"stop", "stop_send", "device_failure"}:
        assert players[-1].bytes > 0
    if mode.startswith("send"):
        assert robot.mic.blocks > samples_before_gap + 10
        assert not any(e["kind"] == "capture_restart_requested" for e in log.events)
    if mode.endswith("restart"):
        assert any(e["kind"] == "reconnected" and e["quiet"] and e["motion_paused"] for e in log.events)
        assert any("Keep speaking English" in item["content"][0]["text"] for item in configs[1]["input"])
    else:
        assert not any(e["kind"] == "reconnected" for e in log.events)
