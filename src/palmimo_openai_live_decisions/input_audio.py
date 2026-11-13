"""Bound microphone buffering independently of network and event-loop progress."""

import asyncio
import threading
import time
from collections import deque
from typing import Any

from .config import RATE
from .recovery import RetryableSessionError


class CaptureStalled(RetryableSessionError):
    """A persistent capture gap requires a bounded microphone restart."""


class SendStalled(RetryableSessionError):
    """A blocked network write requires a fresh Live connection."""


class AudioInput:
    """Drain one SDK subscription without waiting for WebSocket writes."""

    def __init__(self, mic: Any, buffer_seconds: float = 0.5) -> None:
        self.mic = mic
        self.buffer_seconds = buffer_seconds
        self.limit = int(buffer_seconds * RATE * 2)
        self.frames: deque[tuple[float, bytes]] = deque()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.last_capture = time.monotonic()
        self.pending = 0
        self.chunks = 0
        self.dropped = 0
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._capture, name="live-input", daemon=True)

    def start(self) -> None:
        """Subscribe only after the robot has opened its microphone."""
        self.last_capture = time.monotonic()
        self.thread.start()

    def _capture(self) -> None:
        try:
            with self.mic.stream(maxsize=8) as subscription:
                while not self.stop.is_set():
                    chunk = subscription.get(0.2)
                    if chunk is None:
                        continue
                    pcm = chunk.astype("<i2", copy=False).tobytes()
                    if not pcm:
                        continue
                    now = time.monotonic()
                    with self.lock:
                        self.last_capture = now
                        self.chunks += 1
                        self.frames.append((now, pcm))
                        self.pending += len(pcm)
                        self._trim(now)
        except Exception as exc:
            with self.lock:
                self.error = exc

    def _trim(self, now: float) -> None:
        while self.frames and (self.pending > self.limit or now - self.frames[0][0] > self.buffer_seconds):
            _, pcm = self.frames.popleft()
            self.pending -= len(pcm)
            self.dropped += 1

    def take(self) -> bytes | None:
        """Discard stale speech rather than replaying it after a network pause."""
        with self.lock:
            self._trim(time.monotonic())
            if not self.frames:
                return None
            _, pcm = self.frames.popleft()
            self.pending -= len(pcm)
            return pcm

    def check(self) -> None:
        """Keep explicit device failures terminal rather than retrying hardware blindly."""
        with self.lock:
            if self.error is not None:
                raise RuntimeError("Microphone capture device failed") from self.error

    def discard(self) -> None:
        """Drop pending speech during mute or a rate-limit pause, without stopping capture."""
        with self.lock:
            self.dropped += len(self.frames)
            self.frames.clear()
            self.pending = 0

    def stats(self) -> dict[str, Any]:
        """Expose counters and ages only; never log microphone samples."""
        with self.lock:
            now = time.monotonic()
            result = {
                "capture_age_seconds": round(now - self.last_capture, 3),
                "captured_chunks": self.chunks,
                "queued_ms": round(self.pending / (RATE * 2) * 1000, 1),
                "oldest_age_seconds": round(now - self.frames[0][0], 3) if self.frames else 0.0,
                "dropped_chunks": self.dropped,
            }
        for name in ("blocks", "overflows", "late_blocks", "process_seconds"):
            value = getattr(self.mic, name, None)
            if isinstance(value, (int, float)):
                result[f"sdk_{name}"] = round(value, 3)
        return result

    def close(self) -> None:
        """Join the subscriber before reopening or releasing the SDK device."""
        self.stop.set()
        self.thread.join(timeout=1)
        if self.thread.is_alive():
            raise RuntimeError("Microphone subscriber did not stop")
        with self.lock:
            self.frames.clear()
            self.pending = 0


class CaptureRecovery:
    """Bound device reopen attempts across replacement Live sessions."""

    def __init__(self) -> None:
        self.attempts: deque[float] = deque()

    async def restart(self, mic: Any, stop: asyncio.Event) -> bool:
        """Serialize SDK close/open and finish an in-flight operation before parking."""
        now = time.monotonic()
        while self.attempts and now - self.attempts[0] > 60:
            self.attempts.popleft()
        if len(self.attempts) >= 2:
            raise RuntimeError("Microphone recovery exhausted: two restarts within 60 seconds")
        if stop.is_set():
            return False
        self.attempts.append(now)
        cancelled = threading.Event()

        def reopen() -> bool:
            mic.close()
            if not cancelled.is_set() and not stop.is_set():
                mic.open()
                return True
            return False

        worker = asyncio.create_task(asyncio.to_thread(reopen))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled.set()
            await worker
            raise
        except Exception as exc:
            raise RuntimeError("Microphone reopen failed") from exc
