"""Continuous PCM playback with bounded buffering and observable failures."""

import queue
import subprocess
import threading
import time
from collections import deque
from typing import Any

from .config import RATE


class Playback:
    """Keep audio writes off the event loop and reject a growing backlog."""

    def __init__(self, device: str, max_seconds: float = 1.5) -> None:
        from palmimo_sdk import resolve_alsa_device

        resolved = resolve_alsa_device(device, kind="playback")
        if not resolved:
            raise RuntimeError(f"Playback device not found: {device}")
        self.process = subprocess.Popen(
            [
                "aplay",
                "-q",
                "-D",
                resolved,
                "-t",
                "raw",
                "-f",
                "S16_LE",
                "-r",
                str(RATE),
                "-c",
                "1",
                "--buffer-time=500000",
                "--period-time=20000",
                "--start-delay=400000",
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.queue: queue.Queue[bytes | None] = queue.Queue()
        self.limit = int(max_seconds * RATE * 2)
        self.pending = 0
        self.lock = threading.Lock()
        self.error: Exception | None = None
        self.closed = False
        self.underruns = 0
        self.diagnostics: deque[str] = deque(maxlen=4)
        self.packets = 0
        self.audio_bytes = 0
        self.last_packet: float | None = None
        self.max_gap_ms = 0.0
        self.max_pending = 0
        self.thread = threading.Thread(target=self._write, name="live-playback", daemon=True)
        self.monitor = threading.Thread(target=self._monitor, name="live-audio-diagnostics", daemon=True)
        self.thread.start()
        self.monitor.start()

    def submit(self, pcm: bytes) -> None:
        """Fail fast instead of replaying old audio seconds after an interruption."""
        if len(pcm) % 2:
            raise ValueError("PCM16 packet has an incomplete sample")
        with self.lock:
            if self.closed:
                raise RuntimeError("Playback is closed")
            if self.error:
                raise RuntimeError("Playback failed") from self.error
            if self.pending + len(pcm) > self.limit:
                raise RuntimeError("Playback backlog exceeded the latency limit")
            self.pending += len(pcm)
            now = time.monotonic()
            if self.last_packet is not None:
                self.max_gap_ms = max(self.max_gap_ms, (now - self.last_packet) * 1000)
            self.last_packet = now
            self.packets += 1
            self.audio_bytes += len(pcm)
            self.max_pending = max(self.max_pending, self.pending)
            self.queue.put_nowait(pcm)

    def _monitor(self) -> None:
        """Drain ALSA diagnostics so underruns cannot go unnoticed or block the pipe."""
        assert self.process.stderr is not None
        for line in self.process.stderr:
            message = line.decode(errors="replace").strip()
            with self.lock:
                if "underrun" in message.lower():
                    self.underruns += 1
                self.diagnostics.append(message[:300])

    def stats(self) -> dict[str, Any]:
        """Expose timing counters without recording microphone or generated audio."""
        with self.lock:
            return {
                "packets": self.packets,
                "audio_seconds": round(self.audio_bytes / (RATE * 2), 3),
                "max_gap_ms": round(self.max_gap_ms, 1),
                "pending_ms": round(self.pending / (RATE * 2) * 1000, 1),
                "max_pending_ms": round(self.max_pending / (RATE * 2) * 1000, 1),
                "underruns": self.underruns,
                "diagnostics": list(self.diagnostics),
            }

    def _write(self) -> None:
        """Own the pipe until shutdown; report failures to the supervisor."""
        try:
            while (pcm := self.queue.get()) is not None:
                if self.closed:
                    return
                assert self.process.stdin is not None
                self.process.stdin.write(pcm)
                self.process.stdin.flush()
                with self.lock:
                    self.pending -= len(pcm)
        except Exception as exc:
            with self.lock:
                self.error = exc

    def check(self) -> None:
        """Detect failures even when the model is silent."""
        if self.error or self.process.poll() is not None:
            raise RuntimeError("Speaker process stopped unexpectedly")

    def close(self) -> None:
        """Discard queued speech and reap this application's audio process."""
        with self.lock:
            self.closed = True
        self.queue.put_nowait(None)
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            # Only the audio subprocess is killed, never a robot controller.
            self.process.kill()
            self.process.wait(timeout=3)
        self.thread.join(timeout=3)
        self.monitor.join(timeout=3)
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stderr:
            self.process.stderr.close()
        if self.thread.is_alive() or self.monitor.is_alive():
            raise RuntimeError("Audio worker did not stop")


class NullPlayback:
    """Count returned PCM during non-hardware API checks."""

    def __init__(self) -> None:
        self.bytes = 0

    def submit(self, pcm: bytes) -> None:
        """Observe output without opening an audio device."""
        self.bytes += len(pcm)

    def check(self) -> None:
        """No subprocess is owned."""

    def close(self) -> None:
        """No device needs closing."""


async def microphone_frames(mic: Any) -> Any:
    """Keep the SDK's existing echo-cancelled 16 kHz capture continuous."""
    import asyncio

    try:
        with mic.stream() as subscription:
            while True:
                chunk = await asyncio.to_thread(subscription.get, 0.2)
                if chunk is not None:
                    yield chunk.astype("<i2", copy=False).tobytes()
    except OSError as exc:
        raise RuntimeError("Microphone capture device failed") from exc
