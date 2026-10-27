"""Reject stale decisions and keep reactions from competing with conversation."""

import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .config import Settings

QUIET_RELEASE = (
    "Quiet listening has ended. Respond again when addressed and join with brief contributions as usual. "
    "If you were just spoken to, answer now. Keep the conversation language."
)
QUIET_RESTORED = (
    "The connection recovered during quiet listening. Keep listening without speech or backchannels "
    "until directly addressed. Keep the conversation language."
)


@dataclass(frozen=True)
class Reaction:
    """A small application-owned action, never raw servo instructions."""

    expression: str | None = None
    gesture: str | None = None
    instruction: str | None = None


class Conversation:
    """Keep a bounded transcript in memory for reconnect context."""

    def __init__(self, limit: int = 12000) -> None:
        self.parts: deque[dict[str, str]] = deque()
        self.limit = limit

    def append(self, role: str, text: str) -> None:
        """Preserve fragments exactly, without inventing speaker identities."""
        if self.parts and self.parts[-1]["role"] == role:
            self.parts[-1]["content"] += text
        else:
            self.parts.append({"role": role, "content": text})
        while sum(len(p["content"]) for p in self.parts) > self.limit:
            excess = sum(len(p["content"]) for p in self.parts) - self.limit
            if len(self.parts[0]["content"]) <= excess:
                self.parts.popleft()
            else:
                self.parts[0]["content"] = self.parts[0]["content"][excess:]

    def history(self) -> list[dict[str, str]]:
        """Return detached history, including an explicit continuity caveat."""
        if not self.parts:
            return []
        return [
            {
                "role": "developer",
                "content": "The connection has recovered. Below is part of the preceding conversation. "
                "Continue its language and any explicit language preference. "
                "Do not invent missed speech or speaker identities, and do not repeat past actions.",
            },
            *[dict(p) for p in self.parts],
        ]


class ReactionPolicy:
    """Own per-session ordering, expiry, confidence, and reaction frequency."""

    def __init__(self, settings: Settings, clock: Callable[[], float] = time.monotonic) -> None:
        self.settings = settings
        self.clock = clock
        self.sources: OrderedDict[str, tuple[str, int, float]] = OrderedDict()
        self.applied: dict[str, int] = {}
        self.sequence = 0
        self.last_gesture = float("-inf")
        self.last_neck_motion = float("-inf")
        self.last_clap = float("-inf")
        self.last_contribution = float("-inf")
        self.last_expression: str | None = None
        self.quiet = False
        self.listening_sequence = 0

    def release(self, sequence: int | None = None) -> str | None:
        """Lift quiet listening and invalidate decisions from before this request."""
        self.listening_sequence = self.sequence if sequence is None else sequence
        if not self.quiet:
            return None
        self.quiet = False
        return QUIET_RELEASE

    def reconnect(self) -> None:
        """Keep listening/motion state but discard source events from the old connection."""
        self.sources.clear()
        self.applied.clear()
        self.listening_sequence = self.sequence

    def observe(self, event: dict[str, Any]) -> None:
        """Remember source events so late results cannot move the robot."""
        event_id = event.get("event_id")
        if not isinstance(event_id, str):
            return
        self.sequence += 1
        self.sources[event_id] = (event["type"], self.sequence, self.clock())
        while len(self.sources) > 256:
            self.sources.popitem(last=False)

    def evaluate(self, event: dict[str, Any]) -> Reaction | None:
        """Convert a fresh trigger result to at most one reaction."""
        source = self.sources.get(event.get("source_event_id", ""))
        if source is None or event.get("delegation_id") is not None:
            return None
        direction, sequence, observed = source
        if direction != event.get("source_event_type"):
            return None
        if self.clock() - observed > self.settings.decision_ttl:
            return None
        if sequence <= self.applied.get(direction, -1):
            return None
        inner = event.get("event", {})
        if inner.get("type") != "decision.completed":
            return None
        self.applied[direction] = sequence
        answers = {
            answer.get("name"): answer.get("choice")
            for answer in inner.get("decision", {}).get("answers", [])
            if answer.get("type") == "choice"
            and isinstance(answer.get("confidence"), (int, float))
            and answer["confidence"] >= self.settings.confidence
        }
        instruction = None
        participation = answers.get("participation")
        # Only fresh human input can change listening state, never assistant speech
        # or a decision that predates the latest explicit/operator transition.
        if direction != "session.input_transcript.delta" or sequence <= self.listening_sequence:
            participation = None
        if participation == "yield" and not self.quiet:
            self.quiet = True
            self.listening_sequence = sequence
            instruction = "Listen without speaking or backchannels until directly addressed again. Keep the conversation language."
        elif participation == "answer":
            instruction = self.release(sequence)
        elif (
            participation == "contribute"
            and not self.quiet
            and self.clock() - self.last_contribution >= self.settings.contribution_cooldown
        ):
            self.last_contribution = self.clock()
            instruction = (
                "There may be a useful opening to contribute. Wait while people are speaking, then offer "
                "one brief relevant thought in the conversation language if helpful. Do not repeat a recent answer."
            )
        expression = answers.get("expression")
        if expression not in {"neutral", "happy", "sad", "thinking", "surprised", "curious"}:
            expression = None
        if expression == self.last_expression:
            expression = None
        gesture = answers.get("gesture")
        if (
            gesture not in {"nod", "shake", "clap"}
            or self.quiet
            or self.clock() - self.last_gesture < self.settings.gesture_cooldown
        ):
            gesture = None
        if gesture == "clap" and self.clock() - self.last_clap < self.settings.clap_cooldown:
            gesture = None
        if gesture in {"nod", "shake"} and not self.neck_available():
            gesture = None
        return Reaction(expression, gesture, instruction)

    def neck_available(self) -> bool:
        """Share one interval between transcript gestures and autonomous gaze."""
        return self.clock() - self.last_neck_motion >= self.settings.neck_cooldown

    def applied_neck_motion(self) -> None:
        """Reserve the next interval when a neck action actually starts or is accepted."""
        self.last_neck_motion = self.clock()

    def applied_motion(self, reaction: Reaction) -> None:
        """Consume cooldown only after an action is accepted by the scheduler."""
        if reaction.gesture:
            self.last_gesture = self.clock()
        if reaction.gesture in {"nod", "shake"}:
            self.applied_neck_motion()
        if reaction.gesture == "clap":
            self.last_clap = self.clock()
        if reaction.expression:
            self.last_expression = reaction.expression
