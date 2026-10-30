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
from palmimo_openai_live_decisions.idle import IdleDecider
from palmimo_openai_live_decisions.policy import QUIET_RELEASE, QUIET_RESTORED, Conversation, Reaction, ReactionPolicy
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


def test_neck_cooldown_keeps_expressions_and_applause_available() -> None:
    now = [0.0]
    policy = ReactionPolicy(Settings(), lambda: now[0])
    policy.applied_neck_motion()
    for identity, seconds, direction, gesture in [
        ("early", 4.0, "input", "nod"),
        ("reply", 10.0, "output", "shake"),
        ("celebrate", 15.0, "input", "clap"),
        ("ready", 20.0, "output", "nod"),
    ]:
        now[0] = seconds
        policy.observe(source(identity, direction))
        reaction = policy.evaluate(result(identity, direction, gesture=gesture, expression="curious"))
        assert reaction is not None and reaction.expression == "curious"
        assert reaction.gesture == (gesture if identity in {"celebrate", "ready"} else None)


async def test_idle_gaze_and_transcript_reactions_use_one_neck_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    started = asyncio.Event()
    robot = Robot(Palmimo())
    decider = IdleDecider("test")
    decider.history.extend(["curious", "smile", "rest"])

    async def choose(conversation: str, scene: str, tracking: bool, neck_available: bool) -> str:
        assert neck_available
        return "attend"

    async def move(name: str) -> None:
        assert name == "attend"
        started.set()

    monkeypatch.setattr(decider, "choose", choose)
    monkeypatch.setattr(robot, "idle", move)
    session = Session(Settings(), robot, EventLog(), idle=decider)
    session.policy.clock = lambda: now[0]
    loop = asyncio.create_task(session.idle_loop())
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert not session.policy.neck_available()
        now[0] = 110.0
        session.policy.observe(source())
        reaction = session.policy.evaluate(result(gesture="nod", expression="happy"))
        assert reaction == Reaction(expression="happy")
        now[0] = 120.0
        session.policy.observe(source("later"))
        reaction = session.policy.evaluate(result("later", gesture="nod"))
        assert reaction is not None and reaction.gesture == "nod"
        session.policy.applied_motion(reaction)
        assert all(name not in decider.options(True, session.policy.neck_available()) for name in ("nod", "attend"))
    finally:
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)
        await session.settle()


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


def quiet_step(policy: ReactionPolicy, identity: str, participation: str) -> Reaction:
    """Evaluate one fresh participation decision."""
    policy.observe(source(identity))
    reaction = policy.evaluate(result(identity, participation=participation))
    assert reaction is not None
    return reaction


def test_quiet_release_allows_a_new_quiet_request_immediately() -> None:
    policy = ReactionPolicy(Settings(), lambda: 1.0)
    assert quiet_step(policy, "quiet", "yield").instruction and policy.quiet
    assert quiet_step(policy, "resume", "answer").instruction == QUIET_RELEASE
    assert quiet_step(policy, "quiet-again", "yield").instruction and policy.quiet
    assert quiet_step(policy, "resume-again", "answer").instruction == QUIET_RELEASE
    assert quiet_step(policy, "answer-again", "answer").instruction is None


def test_operator_release_rejects_pending_quiet_results() -> None:
    policy = ReactionPolicy(Settings())
    quiet_step(policy, "quiet", "yield")
    policy.observe(source("old"))
    assert policy.release() == QUIET_RELEASE
    reaction = policy.evaluate(result("old", participation="yield"))
    assert reaction is not None and reaction.instruction is None and not policy.quiet
    assert quiet_step(policy, "new", "yield").instruction and policy.quiet


def test_pending_new_quiet_request_survives_an_earlier_answer_result() -> None:
    policy = ReactionPolicy(Settings())
    quiet_step(policy, "quiet", "yield")
    policy.observe(source("answer"))
    policy.observe(source("new-quiet"))
    response = policy.evaluate(result("answer", participation="answer"))
    assert response is not None and response.instruction == QUIET_RELEASE
    response = policy.evaluate(result("new-quiet", participation="yield"))
    assert response is not None and response.instruction and policy.quiet


def test_assistant_speech_cannot_release_quiet() -> None:
    policy = ReactionPolicy(Settings())
    quiet_step(policy, "quiet", "yield")
    policy.observe(source("output", "output"))
    reaction = policy.evaluate(result("output", "output", participation="answer"))
    assert reaction is not None and reaction.instruction is None and policy.quiet


def test_reconnect_forgets_old_decisions_but_keeps_quiet_and_neck_interval() -> None:
    policy = ReactionPolicy(Settings())
    quiet_step(policy, "quiet", "yield")
    policy.observe(source("pending"))
    policy.applied_neck_motion()
    policy.reconnect()
    assert policy.quiet and not policy.neck_available()
    assert policy.evaluate(result("pending", participation="answer")) is None
    assert quiet_step(policy, "new", "answer").instruction == QUIET_RELEASE


async def test_name_mentions_do_not_release_quiet_without_a_direct_address_decision() -> None:
    session = Session(Settings(), FakeRobot(), EventLog())  # type: ignore[arg-type]
    session.policy.quiet = True
    for identity, text in [("mention", "Palmimo is a friendly robot."), ("name", "パルミーモってかわいいよね")]:
        await session.dispatch({**source(identity), "delta": text}, NullPlayback())
        await session.dispatch(result(identity, participation="listen"), NullPlayback())
        assert session.policy.quiet


async def test_explicit_resume_sends_a_release_instruction() -> None:
    session = Session(Settings(), FakeRobot(), EventLog())  # type: ignore[arg-type]
    sent: list[dict[str, Any]] = []

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    session.send = send
    session.policy.quiet = True
    await session.dispatch({**source("idle"), "delta": "自由に動いて"}, NullPlayback())
    event = result(action="resume_idle")
    event["delegation_id"] = "resume"
    await session.delegated(event)
    assert not session.policy.quiet
    assert sent[0]["type"] == "session.instructions.append" and sent[0]["content"] == QUIET_RELEASE
    await session.settle()


async def test_answer_decision_while_quiet_sends_the_release() -> None:
    session = Session(Settings(), FakeRobot(), EventLog())  # type: ignore[arg-type]
    sent: list[dict[str, Any]] = []

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    session.send = send
    session.policy.quiet = True
    await session.dispatch(source("asked"), NullPlayback())
    await session.dispatch(result("asked", participation="answer"), NullPlayback())
    assert not session.policy.quiet
    assert len(sent) == 3
    assert sent[0]["type"] == "session.instructions.append"
    assert sent[0]["content"] == QUIET_RELEASE and sent[0]["event_id"]
    assert sent[1]["type"] == "decision.item.create"
    assert "state: active" in sent[1]["item"]["content"][0]["text"]
    assert sent[2]["type"] == "session.commentary.append"
    assert sent[2]["event_id"] and sent[2]["delegation_id"] is None
    await session.dispatch(
        {"type": "session.instructions.appended", "client_event_id": sent[0]["event_id"]}, NullPlayback()
    )
    assert not session.pending_steering
    assert session.awaiting_quiet_reply
    await session.dispatch({**source("reply", "output"), "delta": "Yes, I can hear you."}, NullPlayback())
    assert not session.awaiting_quiet_reply
    assert session.quiet_reply_request is None


async def test_quiet_reply_is_requested_once_per_fresh_release_and_cancelled_by_new_quiet() -> None:
    session = Session(Settings(), FakeRobot(), EventLog())  # type: ignore[arg-type]
    sent: list[dict[str, Any]] = []

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    session.send = send
    session.policy.quiet = True
    await session.restore_quiet()
    assert not any(e["type"] == "session.commentary.append" for e in sent)
    for identity in ("heard", "question-tail"):
        await session.dispatch(source(identity), NullPlayback())
        await session.dispatch(result(identity, participation="answer"), NullPlayback())
    assert len([e for e in sent if e["type"] == "session.commentary.append"]) == 1
    assert session.awaiting_quiet_reply
    await session.dispatch(source("quiet-again"), NullPlayback())
    await session.dispatch(result("quiet-again", participation="yield"), NullPlayback())
    assert session.policy.quiet and not session.awaiting_quiet_reply
    assert session.quiet_reply_request is None
    await session.dispatch(result("heard", participation="answer"), NullPlayback())
    assert len([e for e in sent if e["type"] == "session.commentary.append"]) == 1
    await session.dispatch(source("new-question"), NullPlayback())
    await session.dispatch(result("new-question", participation="answer"), NullPlayback())
    assert len([e for e in sent if e["type"] == "session.commentary.append"]) == 2


async def test_reconnect_restores_quiet_listening_only_when_quiet() -> None:
    policy = ReactionPolicy(Settings())
    sent: list[dict[str, Any]] = []

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    for quiet in (False, True):
        policy.quiet = quiet
        session = Session(Settings(), FakeRobot(), EventLog(), policy=policy)  # type: ignore[arg-type]
        session.send = send
        await session.restore_quiet()
    assert len(sent) == 2
    assert sent[0]["type"] == "session.instructions.append" and sent[0]["content"] == QUIET_RESTORED
    assert sent[1]["type"] == "decision.item.create"
    assert "state: quiet" in sent[1]["item"]["content"][0]["text"]


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
    assert config["input"][1:] == [
        {
            "type": "message",
            "role": role,
            "content": [{"type": "output_text" if role == "assistant" else "input_text", "text": text}],
        }
        for role, text in turns
    ]


def test_reconnect_history_fits_live_limits_and_keeps_latest_text() -> None:
    conversation = Conversation()
    for index in range(160):
        conversation.append("assistant" if index % 2 else "user", str(index))
    items = session_config(Settings(), conversation.history())["input"]
    assert len(items) == 128
    assert items[0]["role"] == "developer"
    assert items[-1]["content"][0]["text"] == "159"

    conversation.append("user", "日本語の長い会話" * 1000 + "最新の話題")
    items = session_config(Settings(), conversation.history())["input"]
    assert sum(len(item["content"][0]["text"].encode("utf-8")) for item in items) <= 6000
    assert items[-1]["content"][0]["text"].endswith("最新の話題")
    assert "\ufffd" not in items[-1]["content"][0]["text"]


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
    session.policy.applied_neck_motion()
    assert not session.policy.neck_available()
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


async def test_expression_reaction_updates_the_display_without_motor_frames() -> None:
    class Display:
        def __init__(self) -> None:
            self.expressions: list[str] = []

        def set_expression(self, name: str, hold_ms: int = 0) -> str:
            self.expressions.append(name)
            return f"OK {name}"

    display = Display()
    frames: list[dict[str, int]] = []
    robot = Robot(Palmimo(display=display, on_step=frames.append))  # type: ignore[arg-type]
    await robot.react(Reaction(expression="curious"))
    await robot.react(Reaction(expression="thinking"))
    await robot.react(Reaction(expression="happy"))
    assert display.expressions == ["CURIOUS", "THINKING", "HAPPY"]
    assert frames == []


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
