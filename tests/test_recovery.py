"""Exercise failed Decisions through real local WebSockets without hardware."""

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection, serve

from palmimo_openai_live_decisions import audio, recovery, runtime
from palmimo_openai_live_decisions.config import Settings
from palmimo_openai_live_decisions.events import EventLog
from palmimo_openai_live_decisions.recovery import DecisionRecovery, ProtocolError, RetryableSessionError


class RecordingLog(EventLog):
    """Retain diagnostics without printing conversation payloads in tests."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def write(self, kind: str, **fields: Any) -> None:
        self.events.append({"kind": kind, **fields})


def failure(identity: str = "trg_failed", code: str = "decision_failed") -> dict[str, Any]:
    return {
        "type": "error",
        "event_id": "evt_error",
        "error": {"code": code, "type": "server_error", "message": f"Decision sampling failed for {identity}."},
    }


def test_single_failure_recovers_and_drops_late_failed_result() -> None:
    log = RecordingLog()
    policy = DecisionRecovery(Settings(), log)
    policy.session_id = "live_test"
    source = {
        "trigger_id": "trg_failed",
        "source_event_id": "evt_input",
        "source_event_type": "session.input_transcript.delta",
        "event": {"type": "decision.created"},
    }
    assert policy.observe_decision(source)
    policy.handle_error(failure())
    assert not policy.observe_decision({**source, "event": {"type": "decision.completed"}})
    assert policy.observe_decision({"trigger_id": "trg_next", "event": {"type": "decision.completed"}})
    error = next(e for e in log.events if e["kind"] == "api_error")
    assert error["session_id"] == "live_test" and error["source_event_id"] == "evt_input"
    assert any(e["kind"] == "decision_recovered" for e in log.events)


def test_failure_window_expires_and_burst_requires_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    monkeypatch.setattr(recovery.time, "monotonic", lambda: now[0])
    policy = DecisionRecovery(Settings(), RecordingLog())
    policy.handle_error(failure())
    policy.handle_error(failure("trg_second"))
    now[0] += 31
    policy.handle_error(failure("trg_later"))
    assert len(policy.failures) == 1
    policy.handle_error(failure("trg_later_2"))
    with pytest.raises(RetryableSessionError):
        policy.handle_error(failure("trg_later_3"))


@pytest.mark.parametrize("code", ["invalid_api_key", "unknown_parameter", "permission_denied", "unknown_error"])
def test_other_api_errors_never_become_retryable(code: str) -> None:
    with pytest.raises(ProtocolError, match=code):
        DecisionRecovery(Settings(), RecordingLog()).handle_error(failure(code=code))


@pytest.mark.parametrize("reason", ["content", None, "unknown"])
def test_terminal_closure_does_not_restart(reason: str | None) -> None:
    with pytest.raises(ProtocolError):
        recovery.session_closed_error(reason)


@pytest.mark.parametrize("reason", ["expired", "connection_lost"])
def test_recoverable_closure_replaces_session(reason: str) -> None:
    with pytest.raises(RetryableSessionError):
        recovery.session_closed_error(reason)


@pytest.mark.parametrize("reason", ["close_requested", "remote_hangup"])
def test_requested_closure_is_respected(reason: str) -> None:
    recovery.session_closed_error(reason)


class TestRobot:
    """Observe ownership and motion holds while using a fake microphone."""

    __test__ = False

    def __init__(self) -> None:
        self.mic = object()
        self.walking = False
        self.face_tracking = False
        self.connects = 0
        self.closes = 0
        self.stops = 0

    async def connect(self) -> None:
        self.connects += 1

    async def cancel(self) -> None:
        self.walking = False

    async def stop(self) -> str:
        self.stops += 1
        self.walking = False
        return "Motion held."

    async def close(self) -> None:
        self.closes += 1

    def take_wave(self) -> bool:
        return False


@pytest.mark.parametrize("mode", ["single", "burst", "closed", "fatal"])
async def test_voice_pipeline_recovers_over_real_websocket(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    robot = TestRobot()
    log = RecordingLog()
    stop = asyncio.Event()
    heard = asyncio.Event()
    configs: list[dict[str, Any]] = []
    players: list[Any] = []
    pcm = b"\x01\x02" * 32

    class Player:
        def __init__(self, device: str) -> None:
            self.closed = False
            self.received = bytearray()
            players.append(self)

        def submit(self, audio: bytes) -> None:
            self.received.extend(audio)
            heard.set()

        def check(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

        def stats(self) -> dict[str, Any]:
            return {}

    async def microphone(mic: Any) -> AsyncIterator[bytes]:
        while True:
            yield b"\0\0" * 32
            await asyncio.sleep(0.01)

    async def server(ws: ServerConnection) -> None:
        start = json.loads(await ws.recv())
        configs.append(start["session"])
        index = len(configs)
        await ws.send(json.dumps({"type": "session.started", "session": {"id": f"live_{index}"}}))
        await ws.send(
            json.dumps(
                {
                    "type": "session.input_transcript.delta",
                    "event_id": "evt_context",
                    "delta": "Please keep speaking English about our robot demo.",
                }
            )
        )
        if index == 1:
            if mode == "closed":
                await ws.send(
                    json.dumps({"type": "session.closed", "reason": "connection_lost", "usage": {"seconds": 1}})
                )
                await ws.wait_closed()
                return
            for i in range(3 if mode == "burst" else 1):
                await ws.send(
                    json.dumps(failure(f"trg_{i}", "invalid_api_key" if mode == "fatal" else "decision_failed"))
                )
        continuing = index > 1 or mode == "single"
        if continuing:
            await ws.send(json.dumps({"type": "session.output_audio.delta", "delta": base64.b64encode(pcm).decode()}))
            await ws.send(json.dumps({"type": "session.output_transcript.delta", "delta": "Continuing in English."}))
            await ws.send(
                json.dumps(
                    {"type": "decision.event", "trigger_id": "trg_success", "event": {"type": "decision.completed"}}
                )
            )
            await asyncio.wait_for(heard.wait(), 2)
            if mode != "single":
                stop.set()
        async for raw in ws:
            command = json.loads(raw)
            if command["type"] == "session.update":
                await ws.send(json.dumps({"type": "session.updated", "client_event_id": command["event_id"]}))
                if mode == "single":
                    async with asyncio.timeout(2):
                        while not any(e["kind"] == "session_probe_succeeded" for e in log.events):
                            await asyncio.sleep(0)
                    stop.set()
            if command["type"] == "session.close":
                await ws.send(
                    json.dumps({"type": "session.closed", "reason": "close_requested", "usage": {"seconds": 2}})
                )
                return

    monkeypatch.setattr(runtime, "Playback", Player)
    monkeypatch.setattr(runtime, "microphone_frames", microphone)
    async with serve(server, "127.0.0.1", 0) as listener:
        port = listener.sockets[0].getsockname()[1]
        monkeypatch.setattr(runtime, "LIVE_URL", f"ws://127.0.0.1:{port}")
        task = runtime.run(Settings(idle_enabled=False), "test", robot, log, stop, asyncio.Queue())  # type: ignore[arg-type]
        async with asyncio.timeout(8):
            if mode == "fatal":
                with pytest.raises(ProtocolError, match="invalid_api_key"):
                    await task
            else:
                await task
    assert len(configs) == (2 if mode in {"burst", "closed"} else 1)
    assert robot.connects == 1 and robot.closes == 1
    assert all(p.closed for p in players)
    if mode != "fatal":
        assert bytes(players[-1].received) == pcm
    if mode in {"burst", "closed"}:
        assert robot.stops == 1
        assert any("Please keep speaking English" in p["content"] for p in configs[1]["input"])
        assert any(e["kind"] == "reconnected" and e["motion_paused"] for e in log.events)
    if mode == "single":
        assert robot.stops == 0
        assert any(e["kind"] == "decision_recovered" for e in log.events)
        assert not any(e["kind"] == "reconnected" for e in log.events)


async def test_repeated_connection_failures_cool_down_and_stop_promptly(monkeypatch: pytest.MonkeyPatch) -> None:
    stop = asyncio.Event()
    robot = TestRobot()
    log = RecordingLog()
    waits: list[float] = []

    @asynccontextmanager
    async def connect(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        raise OSError("synthetic network outage")
        yield  # pragma: no cover

    async def wait(awaitable: Any, timeout: float) -> None:
        waits.append(timeout)
        awaitable.close()
        if len(waits) == 3:
            stop.set()
            return
        raise TimeoutError

    monkeypatch.setattr(runtime.websockets, "connect", connect)
    monkeypatch.setattr(runtime.asyncio, "wait_for", wait)
    await runtime.run(Settings(reconnect_attempts=1), "test", robot, log, stop, asyncio.Queue())  # type: ignore[arg-type]
    assert waits == [2, 30, 30]
    assert robot.connects == 0 and robot.closes == 1
    assert len([e for e in log.events if e["kind"] == "reconnect_wait"]) == 3


async def test_motion_io_error_is_not_a_network_retry() -> None:
    session = runtime.Session(Settings(), TestRobot(), RecordingLog())  # type: ignore[arg-type]

    async def broken_motion() -> None:
        raise OSError("synthetic servo failure")

    session.motion = asyncio.create_task(broken_motion())
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="Motion device failed"):
        session.check_motion()


@pytest.mark.parametrize("acknowledged", [False, True])
async def test_health_probe_distinguishes_quiet_listening_from_unresponsive_session(acknowledged: bool) -> None:
    log = RecordingLog()
    session = runtime.Session(Settings(), TestRobot(), log)  # type: ignore[arg-type]
    session.policy.quiet = True
    sent: list[dict[str, Any]] = []

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    session.send = send
    await session.dispatch(failure(), None)
    assert sent[0]["type"] == "session.update" and sent[0]["session"] == {}
    assert session.health_probe is not None
    identity, started = session.health_probe
    session.health_probe = (identity, started - 11)
    if acknowledged:
        await session.dispatch({"type": "session.updated", "client_event_id": identity}, None)
        session.check_streams()
        assert session.health_probe is None
    else:
        with pytest.raises(RetryableSessionError, match="health probe"):
            session.check_streams()
    assert session.policy.quiet
    assert session.conversation.history() == []


async def test_microphone_io_error_is_not_a_network_retry() -> None:
    class BrokenMic:
        def stream(self) -> Any:
            raise OSError("synthetic microphone failure")

    with pytest.raises(RuntimeError, match="Microphone capture device failed"):
        async for _ in audio.microphone_frames(BrokenMic()):
            pytest.fail("A broken microphone cannot produce frames")


async def test_stop_during_connection_setup_does_not_open_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    stop = asyncio.Event()
    robot = TestRobot()

    @asynccontextmanager
    async def connect(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield object()

    async def start(*args: Any) -> dict[str, Any]:
        stop.set()
        return {"id": "live_stopped"}

    async def close(*args: Any) -> None:
        pass

    monkeypatch.setattr(runtime.websockets, "connect", connect)
    monkeypatch.setattr(runtime, "start_connection", start)
    monkeypatch.setattr(runtime, "close_connection", close)
    await runtime.run(Settings(), "test", robot, RecordingLog(), stop, asyncio.Queue())  # type: ignore[arg-type]
    assert robot.connects == 0 and robot.closes == 1
