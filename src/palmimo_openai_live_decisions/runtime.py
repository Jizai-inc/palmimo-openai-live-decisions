"""Supervise a Live connection, robot ownership, and graceful recovery."""

import asyncio
import base64
import contextlib
import json
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any

import websockets

from .actions import ACTION_DESCRIPTIONS, ACTIONS, CONTINUOUS_ACTIONS
from .audio import NullPlayback, Playback, microphone_frames
from .config import LIVE_URL, Settings, session_config
from .events import EventLog
from .idle import IDLE_PAUSE_SECONDS, IdleDecider
from .policy import QUIET_RESTORED, Conversation, ReactionPolicy
from .robot import Robot
from .vision import SceneVision


class ProtocolError(RuntimeError):
    """An API configuration error must not start a hardware retry loop."""


@dataclass
class InputControl:
    """Keep microphone and motion holds across connection replacements."""

    muted: bool = False
    motion_paused: bool = False


class Session:
    """Dispatch network events without blocking speech on motor execution."""

    def __init__(
        self,
        settings: Settings,
        robot: Robot,
        log: EventLog,
        conversation: Conversation | None = None,
        control: InputControl | None = None,
        vision: SceneVision | None = None,
        idle: IdleDecider | None = None,
        policy: ReactionPolicy | None = None,
    ) -> None:
        self.settings = settings
        self.robot = robot
        self.log = log
        self.conversation = conversation if conversation is not None else Conversation()
        self.policy = policy if policy is not None else ReactionPolicy(settings)
        self.motion: asyncio.Task[None] | None = None
        self.seen_delegations: deque[str] = deque(maxlen=256)
        self.events = 0
        self.transcript_events = 0
        self.input_events = 0
        self.last_action_input = 0
        self.decision_events = 0
        self.control = control if control is not None else InputControl()
        self.final_usage = False
        self.input_tail = ""
        self.last_server_event = time.monotonic()
        self.last_microphone_frame = time.monotonic()
        self.action_speech: asyncio.Event | None = None
        self.vision = vision
        self.visual_task: asyncio.Task[None] | None = None
        self.idle = idle
        self.idle_request: asyncio.Task[str | None] | None = None
        self.next_idle_at = 0.0
        self.motion_origin = "reaction"
        self.motion_epoch = 0
        self.scene = ""
        self.scene_time = 0.0
        self.send: Callable[[dict[str, Any]], Awaitable[None]]

    def schedule(self, work: Callable[[], Coroutine[Any, Any, None]], *, origin: str = "reaction") -> bool:
        """Let reactions interrupt idle movement; drop other competing gestures."""
        self.check_motion()
        if self.control.motion_paused or self.control.muted:
            return False
        if self.motion is not None and not self.motion.done():
            if self.motion_origin == "idle" and origin == "reaction":
                self.replace_motion(work, origin=origin)
                return True
            return False
        self.invalidate_idle()
        self.motion_origin = origin
        self.motion = asyncio.create_task(work())
        return True

    def invalidate_idle(self) -> None:
        """Discard a plan as soon as higher-priority work or a stop arrives."""
        self.motion_epoch += 1
        if self.idle_request is not None:
            self.idle_request.cancel()

    def check_motion(self) -> None:
        """Surface robot errors to the owning runtime."""
        if self.motion is not None and self.motion.done():
            self.motion.result()
            self.motion = None

    def replace_motion(self, work: Callable[[], Coroutine[Any, Any, None]], *, origin: str = "explicit") -> None:
        """Prioritize a user request while settling the previous task off the receiver."""
        self.check_motion()
        self.invalidate_idle()
        self.motion_origin = origin
        previous = self.motion

        async def handoff() -> None:
            if previous is not None:
                await self.robot.cancel()
                previous.cancel()
                outcome = (await asyncio.gather(previous, return_exceptions=True))[0]
                if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError):
                    raise outcome
            await work()

        self.motion = asyncio.create_task(handoff())

    def stop_motion(self, reason: str) -> None:
        """Settle before confirming a local or transcript-triggered stop."""
        self.control.motion_paused = True
        self.last_action_input = self.input_events

        async def stopped() -> None:
            result = await self.robot.stop()
            self.log.write("motion_stopped", reason=reason)
            await self.send(
                {
                    "type": "session.commentary.append",
                    "delegation_id": None,
                    "content": result + " A new motion request or permission to move freely resumes movement. "
                    "Confirm briefly in the current conversation language.",
                }
            )

        self.replace_motion(stopped)

    async def steer(self, content: str) -> None:
        """Append conversation steering to the Live session instructions."""
        await self.send({"type": "session.instructions.append", "delegation_id": None, "content": content})

    async def lift_quiet(self, release: str | None, reason: str) -> None:
        """Tell the voice model that quiet listening ended; the local flag alone cannot."""
        if release is None:
            return
        self.log.write("quiet_released", reason=reason)
        await self.steer(release)

    async def restore_quiet(self) -> None:
        """Re-establish quiet listening on a new connection, which starts without appended instructions."""
        if not self.policy.quiet:
            return
        self.log.write("quiet_restored")
        await self.steer(QUIET_RESTORED)

    def check_streams(self) -> None:
        """Stop when capture fails, or walking loses its remote voice control path."""
        now = time.monotonic()
        if now - self.last_microphone_frame > self.settings.stream_stall_seconds:
            raise RuntimeError("Microphone capture stalled")
        if self.robot.walking and now - self.last_server_event > self.settings.stream_stall_seconds:
            raise ConnectionError("Live event stream stalled while walking")

    async def announce_action(self, action: str, delegation: str) -> bool:
        """Wait for voice onset and the playback margin before starting a requested action."""
        speech = asyncio.Event()
        self.action_speech = speech
        try:
            await self.send(
                {
                    "type": "session.commentary.append",
                    "delegation_id": delegation,
                    "content": f"Action announcement: {ACTION_DESCRIPTIONS[action]}. The action has not started yet. "
                    "Say a brief, natural acknowledgment in the language of the latest spoken request now, "
                    "even if this short request changed language. Honor an explicit language preference if present. "
                    "Convey the intent; do not read this English notice verbatim or switch language because of it. "
                    "The body will start with your speech.",
                }
            )
            try:
                await asyncio.wait_for(speech.wait(), self.settings.action_ack_timeout)
            except TimeoutError:
                self.log.write("action_skipped", action=action, reason="voice_onset_timeout")
                await self.send(
                    {
                        "type": "session.thinking.append",
                        "delegation_id": delegation,
                        "content": "The action was not executed because voice onset was not confirmed. Do not claim execution.",
                    }
                )
                return False
            self.log.write("action_voice_onset", action=action)
            await asyncio.sleep(self.settings.action_lead_seconds)
            return True
        finally:
            self.action_speech = None

    async def settle(self) -> None:
        """Wait for SDK cancellation to finish before disconnecting hardware."""
        self.invalidate_idle()
        if self.idle_request is not None:
            await asyncio.gather(self.idle_request, return_exceptions=True)
            self.idle_request = None
        if self.visual_task is not None:
            self.visual_task.cancel()
            await asyncio.gather(self.visual_task, return_exceptions=True)
            self.visual_task = None
        await self.robot.cancel()
        if self.motion is not None:
            if not self.motion.done():
                self.motion.cancel()
            try:
                await self.motion
            except asyncio.CancelledError:
                pass
            finally:
                self.motion = None

    def check_wave(self) -> None:
        """Schedule a silent local wave reflex without preempting explicit motion."""
        if not self.robot.take_wave() or self.policy.quiet or self.control.muted or self.control.motion_paused:
            return
        self.log.write("wave_detected")

        async def wave() -> None:
            await self.robot.wave_back()
            self.log.write("wave_completed")

        if not self.schedule(wave):
            self.log.write("wave_skipped", reason="motion_busy")

    async def observe_scene(self, delegation: str | None = None) -> None:
        """Return fresh visual context while leaving speech and motor control available."""
        try:
            if self.vision is None:
                raise RuntimeError("Camera unavailable in this mode")
            question = (
                "\n".join(item["content"] for item in self.conversation.history() if item["role"] == "user")[-800:]
                if delegation
                else ""
            )
            description, age = await self.vision.describe(question)
            self.scene = description
            self.scene_time = time.monotonic() - age
            content = f"Visual observation (captured {age:.1f} seconds ago): {description}"
            self.log.write("vision_observed", requested=delegation is not None, age_seconds=round(age, 2))
        except Exception as exc:
            self.scene = ""
            content = (
                "Visual observation: the current image could not be checked. "
                "Do not describe an earlier observation as the current scene."
            )
            self.log.write("vision_unavailable", error=type(exc).__name__)
        if delegation is not None:
            content += "\nAnswer naturally in the current conversation language, not the observation's language."
        else:
            content += "\nBackground context only; do not speak or move solely because of this update."
        await self.send(
            {
                "type": "session.commentary.append" if delegation else "session.thinking.append",
                "delegation_id": delegation,
                "content": content,
            }
        )

    async def dispatch(self, event: dict[str, Any], player: Any) -> None:
        """Handle continuous audio independently of the transcript policy."""
        self.events += 1
        self.last_server_event = time.monotonic()
        kind = event.get("type")
        self.check_motion()
        if self.visual_task is not None and self.visual_task.done():
            self.visual_task.result()
            self.visual_task = None
        if kind == "session.output_audio.delta":
            player.submit(base64.b64decode(event["delta"], validate=True))
        elif kind in {"session.input_transcript.delta", "session.output_transcript.delta"}:
            self.transcript_events += 1
            self.policy.observe(event)
            role = "user" if kind == "session.input_transcript.delta" else "assistant"
            self.conversation.append(role, event.get("delta", ""))
            if role == "assistant" and event.get("delta", "").strip() and self.action_speech is not None:
                self.action_speech.set()
            if role == "user":
                self.input_events += 1
                self.input_tail = (self.input_tail + event.get("delta", ""))[-32:]
                if not self.control.motion_paused and any(
                    word in self.input_tail.lower()
                    for word in ("止まって", "止めて", "とまって", "とめて", "ストップ", "停止して", "stop")
                ):
                    self.input_tail = ""
                    self.stop_motion("spoken_stop")
                await self.lift_quiet(self.policy.heard(event.get("delta", "")), "name_heard")
            self.log.write(
                "transcript",
                role=role,
                text=event.get("delta", ""),
                start_ms=event.get("start_ms"),
                end_ms=event.get("end_ms"),
            )
        elif kind == "decision.event":
            self.decision_events += 1
            inner = event.get("event", {})
            if inner.get("type") == "decision.completed":
                self.log.write(
                    "decision",
                    delegated=bool(event.get("delegation_id")),
                    source=event.get("source_event_type"),
                    answers=[
                        {key: answer.get(key) for key in ("name", "choice", "confidence")}
                        for answer in inner.get("decision", {}).get("answers", [])
                        if answer.get("type") == "choice"
                    ],
                )
            if event.get("delegation_id"):
                await self.delegated(event)
            else:
                was_quiet = self.policy.quiet
                reaction = self.policy.evaluate(event)
                if reaction is None:
                    self.log.write("decision_ignored")
                    return
                if self.policy.quiet != was_quiet:
                    self.log.write("quiet_entered" if self.policy.quiet else "quiet_released", reason="decision")
                if reaction.expression or reaction.gesture or reaction.instruction:
                    self.log.write(
                        "reaction",
                        expression=reaction.expression,
                        gesture=reaction.gesture,
                        steering=bool(reaction.instruction),
                    )
                if reaction.instruction:
                    await self.steer(reaction.instruction)

                async def react() -> None:
                    await self.robot.react(reaction)
                    self.log.write("reaction_completed", expression=reaction.expression, gesture=reaction.gesture)
                    if reaction.gesture and not self.settings.audio_only:
                        description = {
                            "nod": "nodded briefly",
                            "shake": "shook the head briefly",
                            "clap": "clapped",
                        }[reaction.gesture]
                        await self.send(
                            {
                                "type": "session.thinking.append",
                                "delegation_id": None,
                                "content": f"Motion result: spontaneously {description}. This was a conversational gesture. "
                                "Do not speak or request another action solely because of this notice.",
                            }
                        )

                if self.policy.quiet and self.motion_origin == "idle":
                    self.invalidate_idle()
                    if self.motion is not None:
                        self.replace_motion(self.robot.cancel, origin="reaction")
                if (reaction.expression or reaction.gesture) and self.schedule(
                    react, origin="reaction" if reaction.gesture else "expression"
                ):
                    self.policy.applied_motion(reaction)
        elif kind == "session.input_audio.muted":
            self.log.write("microphone", muted=True)
        elif kind == "session.input_audio.unmuted":
            self.log.write("microphone", muted=False)
        elif kind == "session.closed":
            self.final_usage = True
            self.log.write("session_closed", usage=event.get("usage"))
        elif kind == "error":
            error = event.get("error", {})
            self.log.write("api_error", code=error.get("code"), message=error.get("message"))
            raise ProtocolError(str(error.get("code", "unknown_api_error")))

    async def delegated(self, event: dict[str, Any]) -> None:
        """Execute each explicit request at most once within a session."""
        inner = event.get("event", {})
        if inner.get("type") != "decision.completed":
            return
        delegation = event["delegation_id"]
        if delegation in self.seen_delegations:
            return
        self.seen_delegations.append(delegation)
        choices = [
            a
            for a in inner.get("decision", {}).get("answers", [])
            if a.get("name") == "action"
            and a.get("type") == "choice"
            and a.get("confidence", 0) >= self.settings.confidence
        ]
        action = choices[0].get("choice", "none") if choices else "none"
        if action in {*ACTIONS, "observe", "resume_idle"}:
            if self.input_events <= self.last_action_input:
                action = "none"
            else:
                self.last_action_input = self.input_events
        if action in CONTINUOUS_ACTIONS:
            self.input_tail = ""
        if action == "resume_idle":
            self.control.motion_paused = False
            release = self.policy.release()
            self.input_tail = ""
            self.log.write("idle_resumed")
            await self.lift_quiet(release, "resume_idle")
            await self.send(
                {
                    "type": "session.commentary.append",
                    "delegation_id": delegation,
                    "content": "Autonomous gestures have resumed. Confirm briefly in the current conversation language.",
                }
            )
            return
        if action == "observe":
            if self.visual_task is None or self.visual_task.done():
                self.visual_task = asyncio.create_task(self.observe_scene(delegation))
            else:
                await self.send(
                    {
                        "type": "session.thinking.append",
                        "delegation_id": delegation,
                        "content": "A fresh visual observation is in progress. Wait for that result.",
                    }
                )
            return

        async def started() -> None:
            self.log.write("action_started", action=action)
            await self.send(
                {
                    "type": "session.thinking.append",
                    "delegation_id": delegation,
                    "content": "Walking has started and continues until stopped. It is not complete. Do not repeat the start announcement.",
                }
            )

        async def perform() -> None:
            if action == "stop":
                result = await self.robot.stop()
            else:
                if not self.settings.audio_only and not await self.announce_action(action, delegation):
                    return
                self.log.write("action_execution_started", action=action)
                result = (
                    await self.robot.action(action, started)
                    if action in CONTINUOUS_ACTIONS
                    else await self.robot.action(action)
                )
            self.log.write("action", action=action, result=result)
            await self.send(
                {
                    "type": "session.commentary.append" if action == "stop" else "session.thinking.append",
                    "delegation_id": delegation,
                    "content": result + " Confirm briefly in the current conversation language."
                    if action == "stop"
                    else f"Execution result: {result}. Do not repeat a start announcement because of this completion notice.",
                }
            )

        if action not in {*ACTIONS, "stop"}:
            await self.send(
                {
                    "type": "session.thinking.append",
                    "delegation_id": delegation,
                    "content": "There is no action to execute. Continue the conversation in the person's language.",
                }
            )
        else:
            self.control.motion_paused = action == "stop"
            if action == "stop":
                self.last_action_input = self.input_events
            self.input_tail = ""
            self.replace_motion(perform)

    def idle_available(self) -> bool:
        """Allow silent movement during speech, but never compete with requested work."""
        self.check_motion()
        return bool(
            self.idle is not None
            and self.settings.idle_enabled
            and not self.settings.audio_only
            and not self.control.motion_paused
            and not self.control.muted
            and not self.policy.quiet
            and self.motion is None
            and time.monotonic() >= self.next_idle_at
        )

    async def idle_loop(self) -> None:
        """Leave a short pause between gestures without delaying speech or requests."""
        assert self.idle is not None
        while True:
            if not self.idle_available():
                await asyncio.sleep(0.1)
                continue
            epoch = self.motion_epoch
            started = time.monotonic()
            context = "\n".join(f"{part['role']}: {part['content']}" for part in self.conversation.history())
            scene = self.scene if started - self.scene_time < 20 else ""
            self.idle_request = asyncio.create_task(self.idle.choose(context, scene, self.robot.face_tracking))
            try:
                name = await self.idle_request
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise
                continue
            except Exception as exc:
                self.log.write("idle_unavailable", error=type(exc).__name__)
                await asyncio.sleep(2)
                continue
            finally:
                self.idle_request = None
            if epoch != self.motion_epoch or not self.idle_available() or time.monotonic() - started > 3:
                self.log.write("idle_discarded", reason="superseded_or_expired")
                continue
            if name is None:
                await asyncio.sleep(2)
                continue

            async def perform(name: str = name, started: float = started) -> None:
                assert self.idle is not None
                self.idle.history.append(name)
                self.log.write("idle_started", action=name, decision_ms=round((time.monotonic() - started) * 1000))
                try:
                    await self.robot.idle(name)
                    self.log.write("idle_completed", action=name)
                finally:
                    self.next_idle_at = time.monotonic() + random.uniform(*IDLE_PAUSE_SECONDS)

            self.schedule(perform, origin="idle")


async def start_connection(ws: Any, settings: Settings, history: list[dict[str, str]]) -> None:
    """Require an accepted session before touching peripherals."""
    await ws.send(
        json.dumps({"type": "session.start", "session": session_config(settings, history)}, ensure_ascii=False)
    )
    async with asyncio.timeout(20):
        while True:
            event = json.loads(await ws.recv())
            if event.get("type") == "session.started":
                return
            if event.get("type") == "error":
                error = event.get("error", {})
                raise ProtocolError(f"{error.get('code')}: {error.get('message')}")


async def close_connection(ws: Any, session: Session, player: Any) -> None:
    """Wait briefly for final usage while discarding closing speech."""
    try:
        await ws.send(json.dumps({"type": "session.close"}))
        async with asyncio.timeout(5):
            while not session.final_usage:
                event = json.loads(await ws.recv())
                if event.get("type") == "session.closed":
                    await session.dispatch(event, NullPlayback())
    except (TimeoutError, websockets.ConnectionClosed):
        session.log.write("usage_unconfirmed")


async def run_connection(
    ws: Any, session: Session, player: Any, stop: asyncio.Event, commands: asyncio.Queue[str]
) -> None:
    """Cancel and await every producer before draining final session events."""

    async def send(event: dict[str, Any]) -> None:
        await ws.send(json.dumps(event, ensure_ascii=False))

    session.send = send
    session.last_microphone_frame = time.monotonic()
    session.last_server_event = time.monotonic()
    if session.control.muted:
        await send({"type": "session.input_audio.mute"})
    await session.restore_quiet()

    async def receive() -> None:
        async for raw in ws:
            event = json.loads(raw)
            await session.dispatch(event, player)
            if event.get("type") == "session.closed":
                raise ConnectionError("Live session ended")
        raise ConnectionError("Live connection ended")

    async def microphone() -> None:
        async for pcm in microphone_frames(session.robot.mic):
            session.last_microphone_frame = time.monotonic()
            if not session.control.muted:
                await send({"type": "session.input_audio.append", "audio": base64.b64encode(pcm).decode()})
        raise RuntimeError("Microphone stream ended")

    async def supervise() -> None:
        last_gaze = time.monotonic()
        last_audio_log = last_gaze
        while True:
            session.check_motion()
            session.check_streams()
            session.check_wave()
            player.check()
            if time.monotonic() - last_audio_log >= 5:
                session.log.write("audio", **player.stats())
                last_audio_log = time.monotonic()
            if session.idle is None and session.robot.face_tracking and time.monotonic() - last_gaze >= 8:

                async def gaze() -> None:
                    await session.robot.call("look_at_face", {"seconds": 1.0})

                if session.schedule(gaze):
                    last_gaze = time.monotonic()
            await asyncio.sleep(0.1)

    async def operator() -> None:
        while True:
            command = await commands.get()
            if command == "q":
                stop.set()
            elif command == "x":
                session.stop_motion("operator_stop")
            elif command in {"m", "l"}:
                session.control.muted = command == "m"
                if command == "m":
                    session.stop_motion("input_muted")
                await send({"type": "session.input_audio.mute" if command == "m" else "session.input_audio.unmute"})
            elif command == "s":
                session.log.write(
                    "status",
                    model="gpt-live-1",
                    muted=session.control.muted,
                    events=session.events,
                    decisions=session.decision_events,
                    audio=player.stats(),
                    walking=session.robot.walking,
                    motion_paused=session.control.motion_paused,
                )
            elif command == "i":
                session.control.motion_paused = False
                release = session.policy.release()
                session.input_tail = ""
                session.log.write("idle_resumed", reason="operator")
                await session.lift_quiet(release, "operator")

    async def scene_updates() -> None:
        while True:
            await asyncio.sleep(session.settings.vision_interval)
            if session.vision is not None and not session.vision.lock.locked():
                await session.observe_scene()

    tasks = [asyncio.create_task(coro) for coro in (receive(), microphone(), supervise(), operator(), stop.wait())]
    if session.vision is not None:
        tasks.append(asyncio.create_task(scene_updates()))
    if session.idle is not None:
        tasks.append(asyncio.create_task(session.idle_loop()))
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await session.settle()


async def run(
    settings: Settings, key: str, robot: Robot, log: EventLog, stop: asyncio.Event, commands: asyncio.Queue[str]
) -> None:
    """Run until stopped, recovering boundedly from network failures only."""
    conversation = Conversation()
    policy = ReactionPolicy(settings)
    control = InputControl()
    failures = 0
    deadline = time.monotonic() + settings.seconds if settings.seconds else float("inf")
    connected = False
    frames = getattr(robot, "camera_frames", None)
    vision = SceneVision(frames, key) if frames is not None else None
    idle = IdleDecider(key) if settings.idle_enabled and not settings.audio_only else None
    try:
        while not stop.is_set() and time.monotonic() < deadline:
            player: Playback | None = None
            session = Session(settings, robot, log, conversation, control, vision, idle, policy)
            started = time.monotonic()
            try:
                async with websockets.connect(
                    LIVE_URL,
                    additional_headers={"Authorization": f"Bearer {key}"},
                    open_timeout=20,
                    max_size=4 * 1024 * 1024,
                    ping_interval=10,
                    ping_timeout=15,
                ) as ws:
                    await start_connection(ws, settings, conversation.history())
                    try:
                        if not connected:
                            await robot.connect()
                            connected = True
                        player = Playback(settings.speaker_device)
                    except OSError as exc:
                        raise RuntimeError("Device startup failed") from exc
                    log.write(
                        "ready",
                        model="gpt-live-1",
                        audio_only=settings.audio_only,
                        microphone=settings.mic_device,
                        speaker=settings.speaker_device,
                    )
                    try:
                        remaining = deadline - time.monotonic()
                        if settings.seconds:
                            async with asyncio.timeout(max(0, remaining)):
                                await run_connection(ws, session, player, stop, commands)
                        else:
                            await run_connection(ws, session, player, stop, commands)
                    except TimeoutError:
                        stop.set()
                    finally:
                        await close_connection(ws, session, player)
            except (websockets.ConnectionClosed, OSError, ConnectionError, TimeoutError) as exc:
                control.motion_paused = True
                conversation.append(
                    "developer",
                    "Motion was stopped after a connection interruption. Do not resume without a new request.",
                )
                failures = 1 if time.monotonic() - started > 30 else failures + 1
                log.write("connection_lost", error=type(exc).__name__, attempt=failures)
                if failures > settings.reconnect_attempts:
                    raise RuntimeError("Reconnect attempts exhausted") from exc
                if player:
                    log.write("audio_final", **player.stats())
                    await asyncio.to_thread(player.close)
                    player = None
                if not stop.is_set():
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(stop.wait(), min(2**failures, 15))
            finally:
                if player:
                    log.write("audio_final", **player.stats())
                    await asyncio.to_thread(player.close)
    finally:
        log.write("parking")
        try:
            await robot.cancel()
        finally:
            await robot.close()
        log.write("stopped")
