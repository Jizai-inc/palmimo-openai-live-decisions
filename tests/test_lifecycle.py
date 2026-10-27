"""Verify device ownership, recovery, and playback byte integrity."""

import asyncio
import io
import time
from contextlib import asynccontextmanager
from typing import Any

import pytest
from palmimo_sdk import Palmimo

from palmimo_openai_live_decisions import audio, runtime
from palmimo_openai_live_decisions.config import Settings
from palmimo_openai_live_decisions.events import EventLog
from palmimo_openai_live_decisions.robot import Robot


class FakePlayer:
    """Observe cleanup without opening ALSA."""

    def __init__(self, device: str) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def stats(self) -> dict[str, Any]:
        return {}


class LifecycleRobot:
    """Count resource transitions and optionally fail during startup."""

    def __init__(self, fail_start: bool = False, fail_cancel: bool = False) -> None:
        self.connects = 0
        self.closes = 0
        self.fail_start = fail_start
        self.fail_cancel = fail_cancel

    async def connect(self) -> None:
        self.connects += 1
        if self.fail_start:
            raise OSError("serial device unavailable")

    async def cancel(self) -> None:
        if self.fail_cancel:
            raise RuntimeError("cancel failed")

    async def close(self) -> None:
        self.closes += 1


async def test_reconnect_keeps_history_and_mute_without_reopening_robot(monkeypatch: pytest.MonkeyPatch) -> None:
    stop = asyncio.Event()
    robot = LifecycleRobot()
    sessions: list[runtime.Session] = []
    players: list[FakePlayer] = []
    histories: list[list[dict[str, str]]] = []

    @asynccontextmanager
    async def connect(*args: Any, **kwargs: Any) -> Any:
        yield object()

    async def start(ws: Any, settings: Settings, history: list[dict[str, str]]) -> None:
        histories.append(history)

    async def close(*args: Any) -> None:
        return

    async def connection(ws: Any, session: runtime.Session, player: FakePlayer, *args: Any) -> None:
        sessions.append(session)
        players.append(player)
        if len(sessions) == 1:
            session.conversation.append("user", "Remember this conversation.")
            session.control.muted = True
            session.policy.quiet = True
            raise ConnectionError("simulated network loss")
        stop.set()

    async def backoff(awaitable: Any, timeout: float) -> None:
        awaitable.close()
        raise TimeoutError

    monkeypatch.setattr(runtime.websockets, "connect", connect)
    monkeypatch.setattr(runtime, "start_connection", start)
    monkeypatch.setattr(runtime, "close_connection", close)
    monkeypatch.setattr(runtime, "run_connection", connection)
    monkeypatch.setattr(runtime, "Playback", FakePlayer)
    monkeypatch.setattr(runtime.asyncio, "wait_for", backoff)
    await runtime.run(Settings(), "test", robot, EventLog(), stop, asyncio.Queue())  # type: ignore[arg-type]
    assert robot.connects == 1 and robot.closes == 1
    assert len(sessions) == 2 and sessions[1].control.muted and sessions[1].control.motion_paused
    assert sessions[1].policy is sessions[0].policy and sessions[1].policy.quiet
    assert any(part["content"] == "Remember this conversation." for part in histories[1])
    assert "Motion was stopped" in histories[1][-1]["content"]
    assert all(player.closed for player in players)


async def test_device_startup_failure_parks_without_reconnecting(monkeypatch: pytest.MonkeyPatch) -> None:
    robot = LifecycleRobot(fail_start=True)

    @asynccontextmanager
    async def connect(*args: Any, **kwargs: Any) -> Any:
        yield object()

    async def start(*args: Any) -> None:
        return

    monkeypatch.setattr(runtime.websockets, "connect", connect)
    monkeypatch.setattr(runtime, "start_connection", start)
    with pytest.raises(RuntimeError, match="Device startup failed"):
        await runtime.run(Settings(), "test", robot, EventLog(), asyncio.Event(), asyncio.Queue())  # type: ignore[arg-type]
    assert robot.connects == 1 and robot.closes == 1


async def test_parking_runs_even_when_tool_cancellation_fails() -> None:
    robot = LifecycleRobot(fail_cancel=True)
    stop = asyncio.Event()
    stop.set()
    with pytest.raises(RuntimeError, match="cancel failed"):
        await runtime.run(Settings(), "test", robot, EventLog(), stop, asyncio.Queue())  # type: ignore[arg-type]
    assert robot.closes == 1


async def test_audio_only_does_not_report_a_successful_motion() -> None:
    robot = Robot(Palmimo(), motion_enabled=False)
    assert "no physical action" in await robot.action("nod")


def test_playback_preserves_pcm_and_reports_underruns(monkeypatch: pytest.MonkeyPatch) -> None:
    class Process:
        def __init__(self, command: list[str], **kwargs: Any) -> None:
            self.command = command
            self.stdin = io.BytesIO()
            self.stderr = io.BytesIO(b"underrun!!! (at least 10 ms long)\n")
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
    first, second = b"\x01\x02" * 320, b"\x03\x04" * 640
    player.submit(first)
    player.submit(second)
    deadline = time.monotonic() + 2
    while player.stats()["pending_ms"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert player.process.stdin is not None
    assert player.process.stdin.getvalue() == first + second  # type: ignore[attr-defined]
    assert player.stats()["underruns"] == 1
    assert player.stats()["audio_seconds"] == 0.06
    with pytest.raises(ValueError, match="incomplete"):
        player.submit(b"\x00")
    with pytest.raises(RuntimeError, match="backlog"):
        player.submit(b"\x00\x00" * 32000)
    player.close()
    assert not player.thread.is_alive() and not player.monitor.is_alive()
