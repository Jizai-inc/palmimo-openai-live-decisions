"""Isolate failed decisions from voice and keep bounded recovery diagnostics."""

import re
import time
from collections import OrderedDict, deque
from typing import Any

from .config import Settings
from .events import EventLog


class ProtocolError(RuntimeError):
    """A configuration or terminal session error must not start a retry loop."""


class RetryableSessionError(ConnectionError):
    """Replace an unhealthy session without restarting device ownership."""


def bounded(value: Any) -> str | None:
    """Keep only small diagnostic scalars, never complete protocol payloads."""
    return value[:1024] if isinstance(value, str) else None


class DecisionRecovery:
    """Tolerate isolated failures, retiring a session after a burst of failures."""

    def __init__(self, settings: Settings, log: EventLog) -> None:
        self.settings = settings
        self.log = log
        self.session_id: str | None = None
        self.failures: deque[float] = deque(maxlen=settings.decision_failure_limit)
        self.failed_ids: deque[str] = deque(maxlen=256)
        self.sources: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.pending_failure: float | None = None

    def observe_decision(self, event: dict[str, Any]) -> bool:
        """Associate errors with their source and reject late failed results."""
        identity = bounded(event.get("trigger_id")) or bounded(event.get("delegation_id"))
        source = {
            key: bounded(event.get(key))
            for key in ("trigger_id", "delegation_id", "source_event_id", "source_event_type")
        }
        if identity:
            self.sources[identity] = source
            while len(self.sources) > 256:
                self.sources.popitem(last=False)
        if identity in self.failed_ids:
            self.log.write("failed_decision_discarded", session_id=self.session_id, **source)
            return False
        if event.get("event", {}).get("type") == "decision.completed" and self.pending_failure is not None:
            self.log.write(
                "decision_recovered",
                session_id=self.session_id,
                elapsed_ms=round((time.monotonic() - self.pending_failure) * 1000),
                **source,
            )
            self.pending_failure = None
        return True

    def handle_error(self, event: dict[str, Any]) -> None:
        """Skip one failed evaluation; all other API errors remain terminal."""
        error = event.get("error", {})
        code = bounded(error.get("code"))
        message = bounded(error.get("message"))
        match = re.search(r"\b(?:trg|del)_[A-Za-z0-9_-]+", message or "")
        identity = match.group() if match else None
        source = self.sources.get(identity or "", {})
        self.log.write(
            "api_error",
            session_id=self.session_id,
            event_id=bounded(event.get("event_id")),
            client_event_id=bounded(error.get("client_event_id")) or bounded(event.get("client_event_id")),
            code=code,
            error_type=bounded(error.get("type")),
            param=bounded(error.get("param")),
            message=message,
            decision_id=identity,
            **source,
        )
        if code != "decision_failed":
            raise ProtocolError(code or "unknown_api_error")
        now = time.monotonic()
        if identity:
            self.failed_ids.append(identity)
        while self.failures and now - self.failures[0] > self.settings.decision_failure_window:
            self.failures.popleft()
        self.failures.append(now)
        if self.pending_failure is None:
            self.pending_failure = now
        if len(self.failures) >= self.settings.decision_failure_limit:
            self.log.write("decision_reconnect_required", session_id=self.session_id, failures=len(self.failures))
            raise RetryableSessionError("Repeated decision_failed events")
        self.log.write(
            "decision_failed_ignored", session_id=self.session_id, decision_id=identity, failures=len(self.failures)
        )


def session_closed_error(reason: str | None) -> None:
    """Recover lost/expired sessions without overriding explicit or safety closure."""
    if reason in {"expired", "connection_lost"}:
        raise RetryableSessionError(f"Live session ended: {reason}")
    if reason not in {"close_requested", "remote_hangup"}:
        raise ProtocolError(f"Live session ended: {reason or 'unknown'}")
