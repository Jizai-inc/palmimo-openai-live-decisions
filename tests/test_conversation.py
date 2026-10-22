"""Verify social policy, continuous speech, cancellation, and SDK integration."""

import asyncio
import base64
from pathlib import Path
from typing import Any

import pytest
from palmimo_sdk import Palmimo

from palmimo_openai_live_decisions.actions import ACTION_DESCRIPTIONS, ACTIONS, CONTINUOUS_ACTIONS
from palmimo_openai_live_decisions.audio import NullPlayback
from palmimo_openai_live_decisions.config import Settings, session_config
from palmimo_openai_live_decisions.events import EventLog
from palmimo_openai_live_decisions.policy import Conversation, Reaction, ReactionPolicy
from palmimo_openai_live_decisions.robot import Robot
from palmimo_openai_live_decisions.runtime import Session


def source(identity: str = "one", direction: str = "input") -> dict[str, Any]:
    """Build a source fragment with an explicit direction."""
    return {"event_id": identity, "type": f"session.{direction}_transcript.delta", "delta": "hello"}


def result(identity: str = "one", direction: str = "input", confidence: float = 0.95, **choices: str) -> dict[str, Any]:
    """Build a native trigger result tied to a source event."""
    return {
        "type": "decision.event",
        "source_event_id": identity,
        "source_event_type": f"session.{direction}_transcript.delta",
        "delegation_id": None,
        "event": {
            "type": "decision.completed",
            "decision": {
                "answers": [
                    {"type": "choice", "name": name, "choice": value, "confidence": confidence}
                    for name, value in choices.items()
                ]
            },
        },
    }


def test_policy_discards_unknown_expired_duplicate_and_out_of_order_results() -> None:
    now = [0.0]
    policy = ReactionPolicy(Settings(), lambda: now[0])
    assert policy.evaluate(result()) is None
    policy.observe(source("one"))
    policy.observe(source("two"))
    assert policy.evaluate(result("two", direction="output")) is None
    assert policy.evaluate(result("two", gesture="nod")) is not None
    assert policy.evaluate(result("one", gesture="shake")) is None
    assert policy.evaluate(result("two", gesture="nod")) is None
    policy.observe(source("three"))
    now[0] = 4
    assert policy.evaluate(result("three", gesture="nod")) is None


def test_policy_low_confidence_never_moves_or_steers() -> None:
    policy = ReactionPolicy(Settings())
    policy.observe(source())
    reaction = policy.evaluate(result(confidence=0.2, gesture="nod", participation="contribute"))
    assert reaction is not None
    assert reaction.gesture is None and reaction.instruction is None


def test_policy_cooldown_only_counts_accepted_motions() -> None:
    policy = ReactionPolicy(Settings())
    policy.observe(source("one"))
    reaction = policy.evaluate(result("one", gesture="nod"))
    assert reaction is not None and reaction.gesture == "nod"
    policy.applied_motion(reaction)
    policy.observe(source("two"))
    reaction = policy.evaluate(result("two", gesture="nod", participation="contribute"))
    assert reaction is not None and reaction.gesture is None and reaction.instruction


def test_quiet_request_suppresses_reactions_until_directly_addressed() -> None:
    policy = ReactionPolicy(Settings())
    for identity, participation in [("one", "yield"), ("two", "contribute"), ("three", "answer")]:
        policy.observe(source(identity))
        reaction = policy.evaluate(result(identity, participation=participation, gesture="nod"))
        assert reaction is not None
        if participation == "yield":
            assert reaction.instruction and reaction.gesture is None
        elif participation == "contribute":
            assert reaction.instruction is None and reaction.gesture is None
        else:
            assert reaction.gesture == "nod"


def test_history_is_bounded_without_invented_speaker_labels() -> None:
    conversation = Conversation(limit=12)
    conversation.append("user", "first")
    conversation.append("assistant", "response")
    conversation.append("user", "two people")
    history = conversation.history()
    assert sum(len(p["content"]) for p in history[1:]) == 12
    assert history[-1] == {"role": "user", "content": "two people"}
    history[-1]["content"] = "mutated"
    assert conversation.history()[-1]["content"] == "two people"


def test_reconnect_preserves_multilingual_turns_without_rewriting_them() -> None:
    conversation = Conversation()
    turns = [
        ("user", "これから英語で話してね"),
        ("assistant", "Sure, let's speak English."),
        ("user", "Could you dance?"),
        ("assistant", "I'll dance for four seconds."),
        ("user", "Ahora en español, por favor."),
    ]
    for role, text in turns:
        conversation.append(role, text)
    config = session_config(Settings(), conversation.history())
    assert config["input"][1:] == [{"role": role, "content": text} for role, text in turns]


def test_celebratory_clap_does_not_repeat_on_adjacent_transcript_fragments() -> None:
    now = [0.0]
    policy = ReactionPolicy(Settings(), lambda: now[0])
    for index, seconds in enumerate((0, 5, 16)):
        now[0] = seconds
        identity = str(index)
        policy.observe(source(identity))
        reaction = policy.evaluate(result(identity, gesture="clap"))
        assert reaction is not None
        if index == 1:
            assert reaction.gesture is None
        else:
            assert reaction.gesture == "clap"
            policy.applied_motion(reaction)


def test_config_uses_native_triggers_without_turn_gating_or_duration_limit() -> None:
    config = session_config(Settings(), [])
    assert ACTION_DESCRIPTIONS.keys() == ACTIONS.keys()
    assert config["model"] == "gpt-live-1"
    assert config["delegation"]["type"] == "decisions"
    assert len(config["triggers"]) == 2
    assert "turn_detection" not in config
    assert config["store"] is False and Settings().seconds == 0


class FakeRobot:
    """A motor task that settles through cancellation."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()
        self.actions: list[str] = []
        self.walking = False

    async def cancel(self) -> None:
        """Observe the SDK cancellation path."""
        self.stopped.set()

    async def stop(self) -> str:
        """Observe a completed return to the neutral stance."""
        self.stopped.set()
        return "Motion stopped. Holding the startup stance."

    async def react(self, reaction: Reaction) -> None:
        """Keep the motor busy while audio events are processed."""
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.stopped.set()

    async def action(self, name: str) -> str:
        """Record a completed action."""
        self.actions.append(name)
        return "completed"


async def test_audio_continues_during_human_speech_and_motor_work() -> None:
    robot = FakeRobot()
    session = Session(Settings(), robot, EventLog())  # type: ignore[arg-type]
    player = NullPlayback()
    assert session.schedule(lambda: robot.react(Reaction(gesture="nod")))
    await robot.started.wait()
    await session.dispatch(source(), player)
    await session.dispatch(
        {"type": "session.output_audio.delta", "delta": base64.b64encode(b"\0\0" * 10).decode()}, player
    )
    assert player.bytes == 20 and not robot.stopped.is_set()
    assert not session.schedule(lambda: robot.react(Reaction(gesture="nod")))
    await session.settle()
    assert robot.stopped.is_set() and session.motion is None


@pytest.mark.parametrize("acknowledgment", ["うなずくね", "I'll nod.", "Voy a asentir."])
async def test_delegation_executes_once_and_reports_only_after_execution(acknowledgment: str) -> None:
    robot = FakeRobot()
    session = Session(Settings(action_lead_seconds=0), robot, EventLog())  # type: ignore[arg-type]
    sent: list[dict[str, Any]] = []
    announced = asyncio.Event()

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)
        announced.set()

    session.send = send
    await session.dispatch(source(), NullPlayback())
    event = {
        "delegation_id": "request-one",
        "event": {
            "type": "decision.completed",
            "decision": {"answers": [{"name": "action", "type": "choice", "choice": "nod", "confidence": 0.99}]},
        },
    }
    await session.delegated(event)
    await session.delegated(event)
    await asyncio.wait_for(announced.wait(), 1)
    assert not robot.actions and "has not started yet" in sent[0]["content"]
    await session.dispatch({"type": "session.output_transcript.delta", "delta": acknowledgment}, NullPlayback())
    assert session.motion is not None
    await session.motion
    assert robot.actions == ["nod"] and "completed" in sent[-1]["content"]
    assert sent[-1]["type"] == "session.thinking.append"
    await session.settle()


async def test_explicit_action_preempts_gesture_without_blocking_audio() -> None:
    robot = FakeRobot()
    session = Session(Settings(action_lead_seconds=0), robot, EventLog())  # type: ignore[arg-type]
    sent: list[dict[str, Any]] = []
    announced = asyncio.Event()

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)
        announced.set()

    session.send = send
    await session.dispatch(source(), NullPlayback())
    session.schedule(lambda: robot.react(Reaction(gesture="nod")))
    await robot.started.wait()
    previous = session.motion
    event = result(gesture="nod")
    event["delegation_id"] = "explicit-nod"
    event["event"]["decision"]["answers"] = [{"name": "action", "type": "choice", "choice": "nod", "confidence": 1.0}]
    await session.delegated(event)
    player = NullPlayback()
    await session.dispatch({"type": "session.output_audio.delta", "delta": "AAA="}, player)
    assert player.bytes == 2
    await asyncio.wait_for(announced.wait(), 1)
    assert not robot.actions
    await session.dispatch({"type": "session.output_transcript.delta", "delta": "うなずくね"}, player)
    assert session.motion is not None
    await session.motion
    assert previous is not None and previous.done()
    assert robot.actions == ["nod"] and "completed" in sent[-1]["content"]
    await session.settle()


async def test_missing_spoken_acknowledgment_skips_motion() -> None:
    robot = FakeRobot()
    session = Session(Settings(action_ack_timeout=0.01), robot, EventLog())  # type: ignore[arg-type]
    sent: list[dict[str, Any]] = []

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    session.send = send
    await session.dispatch(source(), NullPlayback())
    event = result(action="dance")
    event["delegation_id"] = "dance"
    await session.delegated(event)
    assert session.motion is not None
    await asyncio.wait_for(session.motion, 1)
    assert not robot.actions and "not executed" in sent[-1]["content"]
    await session.settle()


@pytest.mark.parametrize("stop_source", ["delegation", "transcript"])
async def test_stop_cancels_pending_ack_without_waiting_for_speech(stop_source: str) -> None:
    robot = FakeRobot()
    session = Session(Settings(action_lead_seconds=0), robot, EventLog())  # type: ignore[arg-type]
    sent: list[dict[str, Any]] = []
    announced = asyncio.Event()

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)
        announced.set()

    session.send = send
    player = NullPlayback()
    await session.dispatch(source(), player)
    event = result(action="dance")
    event["delegation_id"] = "dance"
    await session.delegated(event)
    await asyncio.wait_for(announced.wait(), 1)
    if stop_source == "delegation":
        stop = result(action="stop")
        stop["delegation_id"] = "stop"
        await session.delegated(stop)
    else:
        await session.dispatch({"type": "session.input_transcript.delta", "delta": "止まって"}, player)
    assert session.motion is not None
    await asyncio.wait_for(session.motion, 1)
    await session.dispatch({"type": "session.output_transcript.delta", "delta": "踊るね"}, player)
    assert not robot.actions and robot.stopped.is_set()
    assert sum("has not started yet" in item["content"] for item in sent) == 1
    assert "Motion stopped" in sent[-1]["content"]
    await session.settle()


@pytest.mark.parametrize("action", list(ACTIONS))
async def test_all_registered_sdk_actions_stay_in_safe_range_during_dry_run(action: str) -> None:
    frames: list[dict[str, int]] = []

    def observe(positions: dict[str, int]) -> None:
        assert all(200 <= value <= 3900 for value in positions.values())
        frames.append(positions)

    robot = Robot(Palmimo(on_step=observe))
    if action in CONTINUOUS_ACTIONS:
        tool, arguments = ACTIONS[action]
        result_text = await robot.call(tool, arguments)
    else:
        result_text = await robot.action(action)
    assert frames and "interrupted" not in result_text
    await robot.cancel()
    await robot.close()


async def test_motor_failure_reaches_supervisor() -> None:
    session = Session(Settings(), FakeRobot(), EventLog())  # type: ignore[arg-type]

    async def fail() -> None:
        raise RuntimeError("device failed")

    session.schedule(fail)
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="device failed"):
        session.check_motion()


async def test_existing_sdk_neck_motions_complete_without_hardware() -> None:
    robot = Robot(Palmimo())
    assert "nodded" in await robot.call("nod", {"seconds": 0.5})
    assert "shook" in await robot.call("head_shake", {"seconds": 0.5})
    await robot.cancel()
    await robot.close()


def test_logs_are_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.write("status", ready=True)
    log.close()
    assert path.stat().st_mode & 0o777 == 0o600
