"""Synthetic SDK-shaped capture for transport tests without audio devices."""

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np


async def acknowledge_startup(ws: Any) -> None:
    """Mirror the Live boundary after authenticated device initialization."""
    event = json.loads(await ws.recv())
    assert event["type"] == "session.update" and event["event_id"].startswith("startup_")
    await ws.send(json.dumps({"type": "session.updated", "client_event_id": event["event_id"]}))


class FakeMic:
    """Produce PCM at a steady cadence with controllable gaps and reopen failures."""

    def __init__(self) -> None:
        self.gap_until = 0.0
        self.stuck = False
        self.broken = False
        self.fail_open = False
        self.opens = 0
        self.closes = 0
        self.subscribers = 0
        self.blocks = 0

    @contextmanager
    def stream(self, maxsize: int = 8) -> Iterator[Any]:
        self.subscribers += 1
        try:
            yield self
        finally:
            self.subscribers -= 1

    def get(self, timeout: float) -> Any:
        time.sleep(min(timeout, 0.01))
        if self.broken:
            raise OSError("Synthetic device failure")
        if self.stuck or time.monotonic() < self.gap_until:
            return None
        self.blocks += 1
        return np.full(160, self.blocks % 30000, dtype=np.int16)

    def close(self) -> None:
        assert self.subscribers == 0, "Subscribers must settle before closing the device"
        self.closes += 1

    def open(self) -> None:
        self.opens += 1
        if self.fail_open:
            raise RuntimeError("Synthetic reopen failure")
        self.stuck = False
