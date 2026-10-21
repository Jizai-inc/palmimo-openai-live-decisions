"""Exercise continuous motion cancellation before any hardware run."""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from palmimo_sdk import Palmimo

from palmimo_openai_live_decisions import runtime
from palmimo_openai_live_decisions.actions import ACTIONS, CONTINUOUS_ACTIONS
from palmimo_openai_live_decisions.audio import NullPlayback
from palmimo_openai_live_decisions.config import Settings
from palmimo_openai_live_decisions.events import EventLog
from palmimo_openai_live_decisions.robot import Robot
from palmimo_openai_live_decisions.runtime import Session


@pytest.mark.parametrize("action", sorted(CONTINUOUS_ACTIONS))
async def test_walking_stays_active_until_sdk_cancel_and_returns_to_neutral(action: str) -> None:
    frames: list[dict[str, int]] = []

    def observe(positions: dict[str, int]) -> None:
        assert all(200 <= value <= 3900 for value in positions.values())
        frames.append(positions)

    robot = Robot(Palmimo(on_step=observe))
    robot.palmimo.return_to_neutral()
    neutral = robot.palmimo.positions
    started = asyncio.Event()

    async def report() -> None:
        started.set()

    task = asyncio.create_task(robot.action(action, report))
    try:
        await asyncio.wait_for(started.wait(), 3)
        await asyncio.sleep(1.6)
        assert robot.walking and not task.done()
    finally:
        await robot.cancel()
        await asyncio.wait_for(task, 3)
    assert frames and not robot.walking
    assert robot.palmimo.motion == "idle"
    assert robot.palmimo.positions == neutral
    await robot.close()


@pytest.mark.parametrize("motion", ["forward", "body_tilt", "look_left"])
@pytest.mark.parametrize("stop_source", ["spoken", "delegated", "operator"])
async def test_stop_returns_legs_and_neck_to_neutral_before_acknowledgment(motion: str, stop_source: str) -> None:
    frames: list[dict[str, int]] = []
    robot = Robot(Palmimo(on_step=frames.append))
    robot.palmimo.return_to_neutral()
    neutral = robot.palmimo.positions
    await robot.call("look", {"pitch": -10.0, "yaw": 15.0})
    session = Session(Settings(), robot, EventLog())
    acknowledgments: list[dict[str, int]] = []

    async def send(event: dict[str, Any]) -> None:
        if "停止しました" in event.get("content", ""):
            acknowledgments.append(robot.palmimo.positions)

    async def move() -> None:
        await robot.action(motion)

    session.send = send
    session.schedule(move)
    await asyncio.sleep(0.4)
    assert robot.palmimo.positions != neutral
    player = NullPlayback()
    if stop_source == "spoken":
        await session.dispatch({"type": "session.input_transcript.delta", "delta": "止まって"}, player)
    elif stop_source == "operator":
        session.stop_motion("operator_stop")
    else:
        await session.delegated(
            {
                "delegation_id": "stop",
                "event": {
                    "type": "decision.completed",
                    "decision": {
                        "answers": [{"type": "choice", "name": "action", "choice": "stop", "confidence": 1.0}],
                    },
                },
            }
        )
    await session.dispatch({"type": "session.output_audio.delta", "delta": "AAA="}, player)
    assert player.bytes == 2
    assert session.motion is not None
    await asyncio.wait_for(session.motion, 3)
    assert acknowledgments == [neutral]
    assert robot.palmimo.positions == neutral and session.control.motion_paused
    assert all(200 <= value <= 3900 for frame in frames for value in frame.values())
    await session.settle()


async def test_spoken_stop_interrupts_walking_while_output_audio_continues() -> None:
    robot = Robot(Palmimo())
    session = Session(Settings(), robot, EventLog())
    sent: list[dict[str, Any]] = []

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    session.send = send
    started = asyncio.Event()

    async def report() -> None:
        started.set()

    async def walk() -> None:
        await robot.action("forward", report)

    session.schedule(walk)
    await asyncio.wait_for(started.wait(), 3)
    player = NullPlayback()
    for fragment in ("止ま", "って"):
        await session.dispatch({"type": "session.input_transcript.delta", "delta": fragment}, player)
    await session.dispatch({"type": "session.output_audio.delta", "delta": "AAA="}, player)
    assert player.bytes == 2
    assert session.motion is not None
    await asyncio.wait_for(session.motion, 3)
    assert not robot.walking and "停止しました" in sent[-1]["content"]
    await session.settle()


def test_capture_and_server_watchdogs_reject_stalled_control_paths() -> None:
    robot = Robot(Palmimo())
    session = Session(Settings(), robot, EventLog())
    session.last_microphone_frame = time.monotonic() - 3
    with pytest.raises(RuntimeError, match="Microphone capture stalled"):
        session.check_streams()
    session.last_microphone_frame = time.monotonic()
    session.last_server_event = time.monotonic() - 3
    robot.walking = True
    with pytest.raises(ConnectionError, match="Live event stream stalled"):
        session.check_streams()


def test_dance_uses_the_agreed_four_seconds() -> None:
    assert ACTIONS["dance"] == ("dance", {"seconds": 4.0})


async def test_network_loss_settles_actual_sdk_walking(monkeypatch: pytest.MonkeyPatch) -> None:
    lost = asyncio.Event()
    robot = Robot(Palmimo())
    session = Session(Settings(), robot, EventLog())
    started = asyncio.Event()

    class Socket:
        async def send(self, text: str) -> None:
            return

        def __aiter__(self) -> "Socket":
            return self

        async def __anext__(self) -> str:
            await lost.wait()
            raise ConnectionError("simulated transport failure")

    async def microphone(mic: Any) -> AsyncIterator[bytes]:
        while True:
            yield b"\0\0" * 160
            await asyncio.sleep(0.01)

    async def report() -> None:
        started.set()

    async def walk() -> None:
        await robot.action("forward", report)

    monkeypatch.setattr(runtime, "microphone_frames", microphone)
    session.schedule(walk)
    controller = asyncio.create_task(
        runtime.run_connection(Socket(), session, NullPlayback(), asyncio.Event(), asyncio.Queue())
    )
    await asyncio.wait_for(started.wait(), 3)
    lost.set()
    with pytest.raises(ConnectionError, match="simulated transport failure"):
        await asyncio.wait_for(controller, 3)
    assert not robot.walking and robot.palmimo.motion == "idle" and session.motion is None


async def test_replaced_handoff_never_starts_the_cancelled_walk() -> None:
    robot = Robot(Palmimo())
    session = Session(Settings(), robot, EventLog())
    first_running = asyncio.Event()
    settling = asyncio.Event()
    release = asyncio.Event()
    executed: list[str] = []

    async def first_motion() -> None:
        first_running.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            settling.set()
            await release.wait()
            raise

    async def walk() -> None:
        executed.append("walk")

    async def stop() -> None:
        executed.append("stop")

    session.schedule(first_motion)
    await first_running.wait()
    session.replace_motion(walk)
    await settling.wait()
    session.replace_motion(stop)
    release.set()
    assert session.motion is not None
    await asyncio.wait_for(session.motion, 2)
    assert executed == ["stop"]
    await session.settle()
