"""Discard queued pre-ready output without relaxing normal playback limits."""

import asyncio
import base64
import json
from typing import Any

import pytest
from audio_fakes import acknowledge_startup
from test_recovery import RecordingLog, TestRobot
from websockets.asyncio.server import ServerConnection, serve

from palmimo_openai_live_decisions import runtime
from palmimo_openai_live_decisions.config import Settings
from palmimo_openai_live_decisions.recovery import ProtocolError


async def test_slow_device_startup_discards_queued_audio_then_plays_fresh_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log, robot = RecordingLog(), TestRobot()
    stop = asyncio.Event()
    heard: list[bytes] = []
    startup_pcm, fresh_pcm = b"\1\0" * 1600, b"\2\0" * 160

    class Player:
        def __init__(self, device: str) -> None:
            pass

        def submit(self, pcm: bytes) -> None:
            assert pcm == fresh_pcm, "No queued startup output may enter the speaker"
            heard.append(pcm)
            stop.set()

        def check(self) -> None:
            pass

        def stats(self) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            pass

    connect = robot.connect

    async def slow_connect() -> None:
        await asyncio.sleep(0.1)
        await connect()

    async def server(ws: ServerConnection) -> None:
        assert json.loads(await ws.recv())["type"] == "session.start"
        await ws.send(json.dumps({"type": "session.started", "session": {"id": "startup-test"}}))
        for _ in range(40):
            await ws.send(
                json.dumps({"type": "session.output_audio.delta", "delta": base64.b64encode(startup_pcm).decode()})
            )
        await acknowledge_startup(ws)
        sent = False
        async for raw in ws:
            event = json.loads(raw)
            if event["type"] == "session.close":
                await ws.send(json.dumps({"type": "session.closed", "reason": "close_requested"}))
                return
            if event["type"] == "session.input_audio.append" and not sent:
                sent = True
                await ws.send(
                    json.dumps({"type": "session.output_audio.delta", "delta": base64.b64encode(fresh_pcm).decode()})
                )

    monkeypatch.setattr(robot, "connect", slow_connect)
    monkeypatch.setattr(runtime, "Playback", Player)
    async with serve(server, "127.0.0.1", 0) as listener:
        port = listener.sockets[0].getsockname()[1]
        monkeypatch.setattr(runtime, "LIVE_URL", f"ws://127.0.0.1:{port}")
        async with asyncio.timeout(3):
            await runtime.run(Settings(idle_enabled=False), "test", robot, log, stop, asyncio.Queue())  # type: ignore[arg-type]
    assert heard == [fresh_pcm]
    assert any(
        e["kind"] == "startup_output_drained" and e["packets"] == 40 and e["audio_seconds"] == 4 for e in log.events
    )
    assert robot.closes == 1


@pytest.mark.parametrize("failure", ["error", "closed", "no_ack"])
async def test_startup_boundary_requires_a_matching_ack_and_surfaces_failure(failure: str) -> None:
    class Socket:
        async def send(self, value: str) -> None:
            pass

        async def recv(self) -> str:
            if failure == "no_ack":
                await asyncio.Event().wait()
            if failure == "error":
                return json.dumps({"type": "error", "error": {"code": "invalid_request"}})
            return json.dumps({"type": "session.closed", "reason": "close_requested"})

    with pytest.raises((ProtocolError, TimeoutError)):
        await runtime.drain_startup_output(Socket(), RecordingLog(), timeout=0.02)
