"""Private event logging without audio bytes, credentials, or session tokens."""

import json
import os
import time
from pathlib import Path
from typing import Any, TextIO


class EventLog:
    """Write bounded structured diagnostics to an optional owner-only file."""

    def __init__(self, path: Path | None = None) -> None:
        self.file: TextIO | None = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            os.fchmod(fd, 0o600)
            self.file = os.fdopen(fd, "a", buffering=1)

    def write(self, kind: str, **fields: Any) -> None:
        """Log caller-selected fields rather than complete protocol payloads."""
        event = {"time": time.time(), "kind": kind, **fields}
        if self.file:
            self.file.write(json.dumps(event, ensure_ascii=False) + "\n")
        if kind not in {"audio", "decision", "decision_ignored"}:
            print(json.dumps(event, ensure_ascii=False), flush=True)

    def close(self) -> None:
        """Release the file descriptor."""
        if self.file:
            self.file.close()
