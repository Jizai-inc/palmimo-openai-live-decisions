"""Validated local songs and bounded playback, independent of the music provider."""

import asyncio
import json
import re
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

SONG_RATE = 16000


class SongIntroduction:
    """Distinguish generated speech from Live's continuously streamed silence."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.last_packet: float | None = None
        self.last_text: float | None = None
        self.audible_samples = 0
        self.quiet_samples = 0
        self.received_samples = 0
        self.packets = 0

    def observe_audio(self, pcm: bytes) -> None:
        """Measure near-digital silence, not gaps in network delivery."""
        if len(pcm) % 2:
            raise ValueError("PCM16 packet has an incomplete sample")
        if not pcm:
            return
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
        self.last_packet = self.clock()
        self.packets += 1
        self.received_samples += len(samples)
        for offset in range(0, len(samples), SONG_RATE // 50):
            frame = samples[offset : offset + SONG_RATE // 50]
            # Generated silence is normally zero. A low floor retains quiet speech.
            if int(np.max(np.abs(frame))) > 32:
                self.audible_samples += len(frame)
                self.quiet_samples = 0
            else:
                self.quiet_samples += len(frame)

    def observe_text(self, text: str) -> None:
        """Use text only as corroboration, never as proof that audio has played."""
        if text.strip():
            self.last_text = self.clock()

    def waiting_for(self) -> str | None:
        """Require fresh audio, actual voice onset, and a measured silent tail."""
        if self.last_text is None:
            return "announcement_text"
        if not self.audible_samples:
            return "audible_onset"
        if self.last_packet is None or self.clock() - self.last_packet > 0.5:
            return "fresh_audio"
        if self.quiet_samples < int(0.6 * SONG_RATE) or self.clock() - self.last_text < 0.4:
            return "speech_end"
        return None

    def stats(self) -> dict[str, str | int | float | None]:
        """Report timing evidence without storing speech samples or text."""
        return {
            "waiting_for": self.waiting_for(),
            "packets": self.packets,
            "received_seconds": round(self.received_samples / SONG_RATE, 3),
            "audible_seconds": round(self.audible_samples / SONG_RATE, 3),
            "quiet_seconds": round(self.quiet_samples / SONG_RATE, 3),
            "packet_age_seconds": round(self.clock() - self.last_packet, 3) if self.last_packet is not None else None,
        }


@dataclass(frozen=True)
class Song:
    """One preloaded PCM song; no network request is needed to perform it."""

    id: str
    title: str
    description: str
    language: str
    pcm: bytes

    @property
    def action(self) -> str:
        """Expose an allowlisted Decisions choice."""
        return f"sing_{self.id}"

    @property
    def seconds(self) -> float:
        """Return the actual audio duration."""
        return len(self.pcm) / (SONG_RATE * 2)


def load_songbook(path: Path | None) -> tuple[Song, ...]:
    """Validate every asset before opening devices, with a bounded memory footprint."""
    if path is None:
        return ()
    root = path.resolve().parent
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("Songbook must use version 1")
    entries = data.get("songs")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 8:
        raise ValueError("Songbook must contain 1 to 8 songs")
    songs: list[Song] = []
    identities: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Song entry must be an object")
        values = [entry.get(key) for key in ("id", "title", "description", "language", "file")]
        if any(not isinstance(value, str) or not value.strip() or len(value) > 300 for value in values):
            raise ValueError("Song metadata must contain nonempty strings of at most 300 characters")
        identity, title, description, language, filename = (str(value) for value in values)
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", identity) or identity in identities:
            raise ValueError("Song IDs must be unique lowercase identifiers")
        asset = (root / filename).resolve()
        if Path(filename).is_absolute() or not asset.is_relative_to(root):
            raise ValueError("Song files must stay inside the songbook directory")
        if asset.stat().st_size > SONG_RATE * 2 * 61:
            raise ValueError("Song asset is too large")
        with wave.open(str(asset), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (
                1,
                2,
                SONG_RATE,
                "NONE",
            ):
                raise ValueError("Songs must be mono 16 kHz PCM16 WAV files")
            frames = audio.getnframes()
            if not 0 < frames <= SONG_RATE * 60:
                raise ValueError("Songs must be between 0 and 60 seconds")
            pcm = audio.readframes(frames)
            if len(pcm) != frames * 2:
                raise ValueError("Song WAV data is truncated")
        identities.add(identity)
        songs.append(Song(identity, title, description, language, pcm))
    return tuple(songs)


class SongOutput(Protocol):
    """Use the same speaker owner as the conversation."""

    def submit(self, pcm: bytes) -> None: ...


async def play_song(song: Song, output: SongOutput) -> None:
    """Feed short PCM packets without overflowing the conversation's bounded queue."""
    started = time.monotonic()
    packet_bytes = SONG_RATE * 2 // 50
    for offset in range(0, len(song.pcm), packet_bytes):
        due = started + offset / (SONG_RATE * 2)
        if time.monotonic() - due > 0.5:
            raise RuntimeError("Song playback missed its timing deadline")
        output.submit(song.pcm[offset : offset + packet_bytes])
        await asyncio.sleep(max(0, due + 0.02 - time.monotonic()))
    # Include ALSA's 500 ms buffer before switching the output owner back to speech.
    await asyncio.sleep(0.6)
