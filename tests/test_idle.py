"""Exercise autonomous movement without sacrificing audio or stop semantics."""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from palmimo_sdk import Palmimo

from palmimo_live_agent import idle, runtime
from palmimo_live_agent.audio import NullPlayback
from palmimo_live_agent.config import Settings
from palmimo_live_agent.events import EventLog
from palmimo_live_agent.idle import IDLE_ACTIONS, IdleDecider
from palmimo_live_agent.robot import Robot
from palmimo_live_agent.runtime import InputControl, Session


async def discard(event: dict[str, Any]) -> None:
    """Receive protocol output without starting a server."""


def test_idle_options_cannot_get_stuck_in_a_gaze_and_expression_cycle() -> None:
    decider = IdleDecider("test")
    decider.history.extend(["tilt", "attend", "curious", "look_left", "smile", "nod"])
    offered = decider.options(True)
    assert offered and all(IDLE_ACTIONS[name].family == "body" for name in offered)
    assert "tilt" not in offered


@pytest.mark.parametrize("history", [["tilt"], ["tilt", "attend", "curious"]])
def test_idle_body_motion_leaves_room_for_smaller_gestures(history: list[str]) -> None:
    decider = IdleDecider("test")
    decider.history.extend(history)
    offered = decider.options(True)
    assert offered and all(IDLE_ACTIONS[name].family != "body" for name in offered)


def test_idle_body_motion_becomes_optional_after_three_smaller_gestures() -> None:
    decider = IdleDecider("test")
    decider.history.extend(["tilt", "attend", "curious", "look_left"])
    families = {IDLE_ACTIONS[name].family for name in decider.options(False)}
    assert "body" in families and "display" in families


def delegation(action: str) -> dict[str, Any]:
    """Represent an explicit native Decisions request."""
    return {
        "delegation_id": action,
        "event": {
            "type": "decision.completed",
            "decision": {
                "answers": [{"type": "choice", "name": "action", "choice": action, "confidence": 1}],
            },
        },
    }


@pytest.mark.parametrize("choice", ["tilt", "forward", None])
async def test_decider_accepts_only_offered_choices_and_bounds_context(
    choice: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "answers": [
                    {
                        "name": "idle_action",
                        "type": "choice" if choice else "refusal",
                        "choice": choice,
                    }
                ]
            },
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(idle.httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(handle), **kw))
    decider = IdleDecider("test-key")
    decider.history.append("look_left")
    answer = await decider.choose("a" * 5000, "b" * 5000, False)
    assert answer == ("tilt" if choice == "tilt" else None)
    context = json.loads(requests[0]["input"])
    assert len(context["conversation"]) == len(context["scene"]) == 1200
    choices = {item["value"] for item in requests[0]["questions"][0]["choices"]}
    assert "attend" not in choices and "look_left" not in choices and "forward" not in choices


async def test_slow_idle_choice_keeps_audio_available_and_is_cancelled_by_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    started, cancelled = asyncio.Event(), asyncio.Event()
    decider = IdleDecider("test")

    async def choose(*args: Any) -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return "tilt"

    monkeypatch.setattr(decider, "choose", choose)
    session = Session(Settings(), Robot(Palmimo()), EventLog(), idle=decider)
    session.send = discard
    loop = asyncio.create_task(session.idle_loop())
    try:
        await asyncio.wait_for(started.wait(), 1)
        player = NullPlayback()
        await session.dispatch({"type": "session.output_audio.delta", "delta": "AAA="}, player)
        await session.dispatch({"type": "session.input_transcript.delta", "delta": "止まって"}, player)
        await asyncio.wait_for(cancelled.wait(), 1)
        assert session.motion is not None
        await session.motion
        assert player.bytes == 2 and session.control.motion_paused and not decider.history
        assert not session.idle_available()
        await session.dispatch({"type": "session.input_transcript.delta", "delta": "自由に動いて"}, player)
        await session.delegated(delegation("resume_idle"))
        assert session.idle_available()
    finally:
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)
        await session.settle()


async def test_idle_starts_during_speech_and_yields_to_reaction_without_voice_notifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started, cancelled, reacted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    robot = Robot(Palmimo())
    decider = IdleDecider("test")
    sent: list[dict[str, Any]] = []

    async def choose(*args: Any) -> str:
        return "tilt"

    async def move(name: str) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def react() -> None:
        reacted.set()

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    monkeypatch.setattr(decider, "choose", choose)
    monkeypatch.setattr(robot, "idle", move)
    session = Session(Settings(), robot, EventLog(), idle=decider)
    session.send = send
    player = NullPlayback()
    await session.dispatch({"type": "session.input_transcript.delta", "delta": "いま話している途中で"}, player)
    loop = asyncio.create_task(session.idle_loop())
    try:
        await asyncio.wait_for(started.wait(), 0.5)
        await session.dispatch({"type": "session.output_audio.delta", "delta": "AAA="}, player)
        assert player.bytes == 2 and not sent
        assert not session.schedule(react, origin="expression")
        assert session.schedule(react)
        await asyncio.wait_for(reacted.wait(), 1)
        assert cancelled.is_set() and list(decider.history) == ["tilt"]
    finally:
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)
        await session.settle()


async def test_idle_pause_preserves_audio_reactions_and_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 100.0
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: now))
    monkeypatch.setattr(runtime.random, "uniform", lambda low, high: 3.0)
    completed, reacted = asyncio.Event(), asyncio.Event()
    calls: list[str] = []
    robot = Robot(Palmimo())
    decider = IdleDecider("test")

    async def choose(*args: Any) -> str:
        return "curious"

    async def move(name: str) -> None:
        calls.append(name)
        completed.set()

    async def react() -> None:
        reacted.set()

    monkeypatch.setattr(decider, "choose", choose)
    monkeypatch.setattr(robot, "idle", move)
    session = Session(Settings(), robot, EventLog(), idle=decider)
    session.send = discard
    loop = asyncio.create_task(session.idle_loop())
    try:
        await asyncio.wait_for(completed.wait(), 1)
        completed.clear()
        assert not session.idle_available()
        player = NullPlayback()
        await session.dispatch({"type": "session.output_audio.delta", "delta": "AAA="}, player)
        assert player.bytes == 2
        assert session.schedule(react)
        await asyncio.wait_for(reacted.wait(), 1)
        now = 102.99
        await asyncio.sleep(0.12)
        assert len(calls) == 1
        now = 103.0
        await session.dispatch({"type": "session.input_transcript.delta", "delta": "話している途中"}, player)
        await asyncio.wait_for(completed.wait(), 1)
        assert len(calls) == 2
        await session.dispatch({"type": "session.input_transcript.delta", "delta": "止まって"}, player)
        assert session.control.motion_paused
        assert session.motion is not None
        await session.motion
        now = 1000.0
        assert not session.idle_available()
        assert len(calls) == 2
    finally:
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)
        await session.settle()


async def test_stop_holds_all_spontaneous_motion_across_sessions_until_new_request() -> None:
    control = InputControl(motion_paused=True)
    robot = Robot(Palmimo())
    for _ in range(2):
        session = Session(
            Settings(action_ack_timeout=0.001), robot, EventLog(), control=control, idle=IdleDecider("test")
        )
        session.send = discard
        assert not session.idle_available()
        assert not session.schedule(robot.cancel)
        await session.settle()
    await session.dispatch({"type": "session.input_transcript.delta", "delta": "うなずいて"}, NullPlayback())
    await session.delegated(delegation("nod"))
    assert not control.motion_paused and session.motion is not None
    await session.motion
    assert session.idle_available()
    await session.settle()


async def test_stop_rejects_a_late_action_from_the_pre_stop_input() -> None:
    session = Session(Settings(), Robot(Palmimo()), EventLog())
    session.send = discard
    await session.dispatch({"type": "session.input_transcript.delta", "delta": "前に歩いて"}, NullPlayback())
    session.stop_motion("operator_stop")
    assert session.motion is not None
    await session.motion
    await session.delegated(delegation("forward"))
    assert session.control.motion_paused and not session.robot.walking
    assert session.action_speech is None
    await session.settle()


async def test_stale_idle_plan_cannot_run_after_a_preempting_request(monkeypatch: pytest.MonkeyPatch) -> None:
    started, finish = asyncio.Event(), asyncio.Event()
    decider = IdleDecider("test")

    async def choose(*args: Any) -> str:
        started.set()
        try:
            await finish.wait()
        except asyncio.CancelledError:
            return "tilt"
        return "tilt"

    monkeypatch.setattr(decider, "choose", choose)
    session = Session(Settings(), Robot(Palmimo()), EventLog(), idle=decider)
    session.send = discard
    loop = asyncio.create_task(session.idle_loop())
    try:
        await asyncio.wait_for(started.wait(), 1)
        session.stop_motion("test")
        assert session.motion is not None
        await session.motion
        await asyncio.sleep(0.05)
        assert not decider.history
    finally:
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)
        await session.settle()


@pytest.mark.parametrize("name", [name for name, action in IDLE_ACTIONS.items() if action.tool and name != "attend"])
async def test_idle_sdk_motions_stay_in_safe_range_and_finish_at_idle(name: str) -> None:
    positions: list[dict[str, int]] = []
    robot = Robot(Palmimo(on_step=positions.append))
    await robot.idle(name)
    assert positions and all(200 <= value <= 3900 for frame in positions for value in frame.values())
    assert robot.palmimo.motion == "idle"
    await robot.cancel()


async def test_idle_motion_failure_reaches_supervisor(monkeypatch: pytest.MonkeyPatch) -> None:
    robot = Robot(Palmimo())

    async def fail(name: str) -> None:
        raise RuntimeError("bus failure")

    monkeypatch.setattr(robot, "idle", fail)
    session = Session(Settings(), robot, EventLog())
    session.schedule(lambda: robot.idle("tilt"), origin="idle")
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="bus failure"):
        session.check_motion()
