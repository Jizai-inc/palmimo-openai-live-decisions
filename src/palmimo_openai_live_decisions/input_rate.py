"""Pace PCM by sample duration and bound recovery from rejected input audio."""

import time
from collections import deque
from typing import Any

from .config import RATE

INPUT_RATE_ERROR = "input_audio_rate_limit_exceeded"


class AudioSendRate:
    """Never accumulate send credit across capture gaps or transport stalls."""

    def __init__(self) -> None:
        self.next_send_at = 0.0
        self.pause_until = 0.0
        self.first_send_at: float | None = None
        self.sent_bytes = 0
        self.recent: deque[tuple[float, int]] = deque()
        self.errors: deque[float] = deque(maxlen=3)
        self.error_count = 0
        self.resume_pending = False

    def delay(self) -> float:
        """Wait before taking a frame, so queued speech can still expire."""
        return max(0.0, max(self.next_send_at, self.pause_until) - time.monotonic())

    def paused(self) -> bool:
        """Discard capture throughout an API-requested sending pause."""
        return time.monotonic() < self.pause_until

    def limited(self) -> int:
        """Pause for one/two seconds; three errors in thirty seconds retire a session."""
        now = time.monotonic()
        while self.errors and now - self.errors[0] > 30:
            self.errors.popleft()
        self.errors.append(now)
        self.error_count += 1
        self.pause_until = now + min(len(self.errors), 2)
        self.resume_pending = True
        return len(self.errors)

    def sent(self, size: int, started: float, finished: float) -> bool:
        """Count completed writes, not server acceptance; return a local resume signal."""
        duration = size / (RATE * 2)
        self.next_send_at = max(started + duration, finished)
        if self.first_send_at is None:
            self.first_send_at = started
        self.sent_bytes += size
        self.recent.append((finished, size))
        self._trim(finished)
        resumed = self.resume_pending and started >= self.pause_until
        if resumed:
            self.resume_pending = False
        return resumed

    def _trim(self, now: float) -> None:
        while self.recent and now - self.recent[0][0] >= 5:
            self.recent.popleft()

    def stats(self) -> dict[str, Any]:
        """Report client write rates; TCP/server arrival timing is not observable here."""
        now = time.monotonic()
        self._trim(now)
        bytes_1s = sum(size for at, size in self.recent if now - at < 1)
        bytes_5s = sum(size for _, size in self.recent)
        return {
            "sent_audio_ms": round(self.sent_bytes / (RATE * 2) * 1000, 1),
            "send_elapsed_seconds": round(now - self.first_send_at, 3) if self.first_send_at is not None else 0.0,
            "sent_audio_ms_last_1s": round(bytes_1s / (RATE * 2) * 1000, 1),
            "sent_audio_ms_last_5s": round(bytes_5s / (RATE * 2) * 1000, 1),
            "send_rate_1s": round(bytes_1s / (RATE * 2), 3),
            "send_rate_5s": round(bytes_5s / (RATE * 2 * 5), 3),
            "pacing_wait_ms": round(self.delay() * 1000, 1),
            "rate_limit_events": self.error_count,
            "rate_pause_remaining_ms": round(max(0.0, self.pause_until - now) * 1000, 1),
        }
