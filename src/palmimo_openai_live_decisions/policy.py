"""Reject stale decisions and keep reactions from competing with conversation."""

import re
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
# A name can be mentioned rather than addressed, so this release does not prompt an answer.
NAME_RELEASE = (
    "Quiet listening has ended because your name was heard. If that was addressed to you, answer briefly; "
    "if people were only talking about you, keep listening until addressed. Keep the conversation language."
)
# Speech recognition often bends the name (アルミモ, パルビーモ, カルビーモ); match those spellings too.
NAME_PATTERN = re.compile(r"[パバアカ]ル[ミビ]ー?モ|パルミ|palmimo", re.IGNORECASE)


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
        self.last_clap = float("-inf")
        self.last_contribution = float("-inf")
        self.last_expression: str | None = None
        self.quiet = False
        self.quiet_since = float("-inf")
        self.released_at = float("-inf")
        self.quiet_heard = ""

    def release(self) -> str | None:
        """Lift quiet listening and return the steering that tells the voice model so."""
        if not self.quiet:
            return None
        self.quiet = False
        self.released_at = self.clock()
        self.quiet_heard = ""
        return QUIET_RELEASE

    def heard(self, text: str) -> str | None:
        """Treat the robot's name, heard after the quiet request itself, as being addressed."""
        if not self.quiet:
            return None
        heard = self.quiet_heard + text
        if self.clock() - self.quiet_since < self.settings.quiet_name_grace:
            # Keep only a name head that a fragment boundary could split, never a whole name.
            self.quiet_heard = heard[-4:]
            return None
        self.quiet_heard = heard[-32:]
        if not NAME_PATTERN.search(heard):
            return None
        self.release()
        return NAME_RELEASE

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
        was_quiet = self.quiet
        release = self.release() if participation == "answer" else None
        # Transcript context keeps an earlier quiet request visible; do not re-enter right after a release.
        if participation == "yield" and self.clock() - self.released_at < self.settings.quiet_rearm_seconds:
            participation = None
        if participation == "yield" and not self.quiet:
            self.quiet = True
            instruction = "Listen without speaking or backchannels until directly addressed again. Keep the conversation language."
        elif participation == "answer":
            self.quiet = False
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
        if expression not in {"neutral", "happy", "sad", "thinking", "surprised"}:
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
        if self.quiet and not was_quiet:
            self.quiet_since = self.clock()
            self.quiet_heard = ""
        return Reaction(expression, gesture, instruction or release)

    def applied_motion(self, reaction: Reaction) -> None:
        """Consume cooldown only after an action is accepted by the scheduler."""
        if reaction.gesture:
            self.last_gesture = self.clock()
        if reaction.gesture == "clap":
            self.last_clap = self.clock()
        if reaction.expression:
            self.last_expression = reaction.expression
