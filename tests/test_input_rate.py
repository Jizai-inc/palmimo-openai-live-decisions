"""Verify rate-limited Live audio remains paced and recoverable without devices."""

import asyncio
import base64
import json
import time
from typing import Any

import numpy as np
import pytest
from audio_fakes import FakeMic
from test_recovery import RecordingLog, TestRobot
from websockets.asyncio.server import ServerConnection, serve

from palmimo_openai_live_decisions import input_rate, runtime
from palmimo_openai_live_decisions.config import Settings
from palmimo_openai_live_decisions.input_rate import INPUT_RATE_ERROR, AudioSendRate


def test_pacing_uses_actual_pcm_duration_without_catchup_credit(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [10.0]
    monkeypatch.setattr(input_rate.time, "monotonic", lambda: now[0])
    rate = AudioSendRate()
    rate.sent(1024, 10.0, 10.001)
    now[0] = 10.01
    assert rate.delay() == pytest.approx(0.022)
    now[0] = 20.0
    assert rate.delay() == 0
    rate.sent(3200, 20.0, 20.002)
    now[0] = 20.01
    assert rate.delay() == pytest.approx(0.09)
    rate.sent(1024, 20.1, 23.0)
    now[0] = 23.0
    assert rate.delay() == 0
    rate.sent(1024, 23.0, 23.001)
    now[0] = 23.002
    assert rate.delay() == pytest.approx(0.03)


def test_rate_pause_window_resume_and_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    monkeypatch.setattr(input_rate.time, "monotonic", lambda: now[0])
    rate = AudioSendRate()
    assert rate.limited() == 1
    assert rate.paused() and rate.delay() == 1
    now[0] = 100.5
    assert not rate.sent(1024, 99.9, 100.5)
    now[0] = 101.0
    assert not rate.paused()
    assert rate.sent(1024, 101.0, 101.001)
    assert not rate.sent(1024, 101.032, 101.033)
    assert rate.limited() == 2 and rate.delay() == 2
    now[0] = 103.0
    assert rate.limited() == 3
    now[0] = 134.0
    assert rate.limited() == 1
    stats = rate.stats()
    assert stats["sent_audio_ms"] == 96
    assert stats["sent_audio_ms_last_5s"] == stats["send_rate_1s"] == 0
    assert stats["rate_limit_events"] == 4
    now[0] = 135.0
    rate.sent(3200, 135.0, 135.0)
    assert rate.stats()["send_rate_1s"] == 0.1
    assert rate.stats()["send_rate_5s"] == 0.02


@pytest.mark.parametrize("mode", ["single", "repeated", "stop", "muted"])
async def test_input_rate_errors_preserve_voice_and_only_reconnect_when_needed(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    robot, log = TestRobot(), RecordingLog()
    stop = asyncio.Event()
    commands: asyncio.Queue[str] = asyncio.Queue()
    configurations: list[dict[str, Any]] = []
    sessions: list[runtime.Session] = []
    players: list[Any] = []
    resumed_frames: list[int] = []
    pcm = b"\1\0" * 160

    class Player:
        def __init__(self, device: str) -> None:
            self.audio = bytearray()
            self.closed = False
            players.append(self)

        def submit(self, chunk: bytes) -> None:
            self.audio.extend(chunk)

        def check(self) -> None:
            pass

        def stats(self) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            self.closed = True

    original_run = runtime.run_connection

    async def connection(ws: Any, session: runtime.Session, *args: Any) -> None:
        sessions.append(session)
        if len(sessions) == 1:
            session.policy.quiet = True
        await original_run(ws, session, *args)

    async def server(ws: ServerConnection) -> None:
        configuration = json.loads(await ws.recv())["session"]
        configurations.append(configuration)
        index = len(configurations)
        await ws.send(json.dumps({"type": "session.started", "session": {"id": f"live_{index}"}}))
        sent_error = False
        paused_at = 0.0
        async for raw in ws:
            event = json.loads(raw)
            if event["type"] == "session.close":
                await ws.send(json.dumps({"type": "session.closed", "reason": "close_requested"}))
                return
            if event["type"] == "session.update":
                await ws.send(json.dumps({"type": "session.updated", "client_event_id": event["event_id"]}))
                # The receiver and playback must stay responsive while input is paused.
                await ws.send(
                    json.dumps({"type": "session.output_audio.delta", "delta": base64.b64encode(pcm).decode()})
                )
                async with asyncio.timeout(1):
                    while not players[-1].audio:
                        await asyncio.sleep(0.005)
                if mode == "stop":
                    stop.set()
                if mode == "muted":
                    commands.put_nowait("m")
            if event["type"] == "session.input_audio.mute" and mode == "muted":
                await asyncio.sleep(1.1)
                assert sessions[-1].control.muted
                assert sessions[-1].audio_rate.sent_bytes == 320
                stop.set()
            if event["type"] != "session.input_audio.append":
                continue
            if index == 1 and not sent_error:
                sent_error = True
                paused_at = time.monotonic()
                await ws.send(
                    json.dumps({"type": "session.input_transcript.delta", "delta": "Remember our quiet conversation."})
                )
                for i in range(3 if mode == "repeated" else 1):
                    await ws.send(
                        json.dumps(
                            {
                                "type": "error",
                                "event_id": f"limit_{i}",
                                "error": {
                                    "code": INPUT_RATE_ERROR,
                                    "type": "invalid_request_error",
                                    "param": "audio",
                                },
                            }
                        )
                    )
            elif index > 1 or (mode == "single" and time.monotonic() - paused_at >= 0.9):
                resumed_frames.append(int(np.frombuffer(base64.b64decode(event["audio"]), dtype="<i2")[0]))
                assert resumed_frames[-1] >= robot.mic.blocks - 5
                await ws.send(
                    json.dumps({"type": "session.output_audio.delta", "delta": base64.b64encode(pcm).decode()})
                )
                async with asyncio.timeout(1):
                    while not players[-1].audio:
                        await asyncio.sleep(0.005)
                stop.set()

    monkeypatch.setattr(runtime, "Playback", Player)
    monkeypatch.setattr(runtime, "run_connection", connection)
    async with serve(server, "127.0.0.1", 0) as listener:
        port = listener.sockets[0].getsockname()[1]
        monkeypatch.setattr(runtime, "LIVE_URL", f"ws://127.0.0.1:{port}")
        async with asyncio.timeout(8):
            await runtime.run(Settings(idle_enabled=False), "test", robot, log, stop, commands)  # type: ignore[arg-type]
    assert len(configurations) == (2 if mode == "repeated" else 1)
    assert robot.connects == robot.closes == 1 and robot.mic.opens == 0
    assert robot.mic.subscribers == 0 and all(player.closed for player in players)
    assert sessions[-1].policy.quiet
    assert any(e["kind"] == "input_rate_limited" and "sent_audio_ms" in e for e in log.events)
    if mode == "repeated":
        assert any(
            "Remember our quiet conversation." in item["content"][0]["text"] for item in configurations[1]["input"]
        )
        assert sessions[-1].control.motion_paused
    if mode == "single":
        assert any(e["kind"] == "input_rate_resumed" for e in log.events)
        assert not any(e["kind"] == "connection_lost" for e in log.events)


async def test_backlogged_microphone_is_paced_at_pcm_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    class FloodMic(FakeMic):
        def get(self, timeout: float) -> Any:
            time.sleep(0.005)
            self.blocks += 1
            return np.full(640, self.blocks, dtype=np.int16)

    robot, log = TestRobot(), RecordingLog()
    robot.mic = FloodMic()
    session = runtime.Session(Settings(idle_enabled=False), robot, log)  # type: ignore[arg-type]
    stop = asyncio.Event()
    writes: list[tuple[float, int]] = []

    class Socket:
        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> Any:
            await asyncio.Future()

        async def send(self, raw: str) -> None:
            event = json.loads(raw)
            if event["type"] == "session.input_audio.append":
                writes.append((time.monotonic(), len(base64.b64decode(event["audio"]))))
                if len(writes) == 15:
                    stop.set()

    class Player:
        def check(self) -> None:
            pass

        def stats(self) -> dict[str, Any]:
            return {}

    async with asyncio.timeout(3):
        await runtime.run_connection(Socket(), session, Player(), stop, asyncio.Queue())
    for (started, size), (following, _) in zip(writes, writes[1:], strict=False):
        assert following - started >= size / 32000 - 0.002
    final = next(e for e in log.events if e["kind"] == "input_audio_final")
    assert final["sent_audio_ms"] == 600
    assert final["dropped_chunks"] > 0
