"""Continuous PCM playback with bounded buffering and observable failures."""

import contextlib
import queue
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from .config import RATE

if sys.platform == "linux" or TYPE_CHECKING:
    import fcntl
    import termios


class PlaybackBacklog(RuntimeError):
    """The speaker has fallen behind; queued Live audio is no longer timely."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Playback backlog exceeded the latency limit ({reason})")
        self.reason = reason


class Playback:
    """Keep audio writes off the event loop and reject a growing backlog."""

    def __init__(self, device: str, max_seconds: float = 1.5) -> None:
        self.device = device
        self.max_seconds = max_seconds
        self.resetting = False
        self._open()

    def _open(self) -> None:
        """Start a new speaker process after the previous one has been reaped."""
        from palmimo_sdk import resolve_alsa_device

        resolved = resolve_alsa_device(self.device, kind="playback")
        if not resolved:
            raise RuntimeError(f"Playback device not found: {self.device}")
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
        self.limit = int(self.max_seconds * RATE * 2)
        self.pending = 0
        self.lock = threading.Lock()
        self.error: Exception | None = None
        self.closed = False
        self.underruns = 0
        self.diagnostics: deque[str] = deque(maxlen=4)
        self.packets = 0
        self.audio_bytes = 0
        self.last_packet: float | None = None
        self.write_started: float | None = None
        self.writing_bytes = 0
        self.written_packets = 0
        self.max_gap_ms = 0.0
        self.max_pending = 0
        self.thread = threading.Thread(target=self._write, name="live-playback", daemon=True)
        self.monitor = threading.Thread(target=self._monitor, name="live-audio-diagnostics", daemon=True)
        self.thread.start()
        self.monitor.start()

    def _pipe_bytes(self) -> int | None:
        """Linux exposes unread bytes on the write end of aplay's stdin pipe."""
        if sys.platform != "linux" or self.process.stdin is None:
            return None
        try:
            return int(struct.unpack("i", fcntl.ioctl(self.process.stdin, termios.FIONREAD, bytes(4)))[0])
        except (OSError, ValueError, TypeError):
            # In-memory sinks and the closing pipe do not expose this counter.
            return None

    def _check_backlog(self, incoming: int = 0) -> None:
        """Called under the writer lock; in-flight bytes can already be inside the pipe."""
        if self.pending + incoming > self.limit:
            raise PlaybackBacklog("queue_limit")
        pipe = self._pipe_bytes()
        if pipe is not None and self.pending - self.writing_bytes + pipe + incoming > self.limit:
            raise PlaybackBacklog("speaker_pipe")

    def submit(self, pcm: bytes) -> None:
        """Fail fast instead of replaying old audio seconds after an interruption."""
        if len(pcm) % 2:
            raise ValueError("PCM16 packet has an incomplete sample")
        with self.lock:
            if self.closed:
                raise RuntimeError("Playback is closed")
            if self.error:
                raise RuntimeError("Playback failed") from self.error
            self._check_backlog(len(pcm))
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
            now = time.monotonic()
            pipe = self._pipe_bytes()
            return {
                "packets": self.packets,
                "written_packets": self.written_packets,
                "audio_seconds": round(self.audio_bytes / (RATE * 2), 3),
                "max_gap_ms": round(self.max_gap_ms, 1),
                "last_packet_age_ms": round((now - self.last_packet) * 1000, 1)
                if self.last_packet is not None
                else None,
                "write_age_ms": round((now - self.write_started) * 1000, 1) if self.write_started is not None else None,
                "pending_ms": round(self.pending / (RATE * 2) * 1000, 1),
                "pipe_pending_ms": round(pipe / (RATE * 2) * 1000, 1) if pipe is not None else None,
                "max_pending_ms": round(self.max_pending / (RATE * 2) * 1000, 1),
                "underruns": self.underruns,
                "diagnostics": list(self.diagnostics),
            }

    def _write(self) -> None:
        """Own the pipe until shutdown; report failures to the supervisor."""
        try:
            while (pcm := self.queue.get()) is not None:
                with self.lock:
                    if self.closed:
                        return
                    self.write_started = time.monotonic()
                    self.writing_bytes = len(pcm)
                assert self.process.stdin is not None
                self.process.stdin.write(pcm)
                self.process.stdin.flush()
                with self.lock:
                    self.write_started = None
                    self.writing_bytes = 0
                    self.written_packets += 1
                    self.pending -= len(pcm)
            if not self.closed and self.process.stdin:
                self.process.stdin.close()
        except Exception as exc:
            with self.lock:
                self.error = exc

    def check(self) -> None:
        """Detect failures even when the model is silent."""
        if self.resetting:
            return
        if self.error or self.process.poll() is not None:
            raise RuntimeError("Speaker process stopped unexpectedly")
        with self.lock:
            if self.write_started is not None and time.monotonic() - self.write_started > self.max_seconds:
                raise PlaybackBacklog("speaker_write")
            self._check_backlog()

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
        if self.thread.is_alive() or self.monitor.is_alive():
            raise RuntimeError("Audio worker did not stop")
        if self.process.stdin:
            # A blocked BufferedWriter can still have bytes to flush after we
            # deliberately terminated its reader. That pipe is being discarded.
            with contextlib.suppress(BrokenPipeError):
                self.process.stdin.close()
        if self.process.stderr:
            self.process.stderr.close()

    def interrupt(self) -> None:
        """Drop queued and ALSA-buffered audio before handing the speaker back."""
        self.resetting = True
        try:
            self.close()
            self._open()
        finally:
            self.resetting = False

    def drain(self) -> None:
        """Play the complete queued introduction before opening a fresh song stream."""
        self.resetting = True
        try:
            self.queue.put_nowait(None)
            try:
                result = self.process.wait(timeout=self.max_seconds + 3)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Speaker did not drain the introduction") from exc
            self.thread.join(timeout=1)
            if result != 0 or self.error or self.thread.is_alive():
                raise RuntimeError("Speaker failed while draining the introduction")
            self.close()
            self._open()
        finally:
            self.resetting = False


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

    def interrupt(self) -> None:
        """No buffered audio needs discarding."""

    def drain(self) -> None:
        """No physical playback buffer needs draining."""
