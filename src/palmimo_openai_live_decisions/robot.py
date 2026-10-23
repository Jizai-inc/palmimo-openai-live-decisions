"""Adapt SDK motions and the public companion example's face tracking."""

import asyncio
import contextlib
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

from palmimo_sdk import Palmimo
from palmimo_sdk.agent.toolset import AgentToolSet

from .actions import ACTIONS, CONTINUOUS_ACTIONS
from .config import Settings
from .idle import IDLE_ACTIONS
from .policy import Reaction
from .vision import CameraFrames
from .walking import make_walking_tool

FACES = {
    "neutral": "IDLE",
    "happy": "HAPPY",
    "sad": "SAD",
    "thinking": "THINKING",
    "surprised": "SURPRISE",
    "curious": "CURIOUS",
}


class Robot:
    """One SDK owner with serialized, cancellation-aware tool execution."""

    def __init__(
        self, palmimo: Palmimo, *, hardware: bool = False, mic: Any = None, motion_enabled: bool = True
    ) -> None:
        self.palmimo = palmimo
        self.hardware = hardware
        self.mic = mic
        self.motion_enabled = motion_enabled
        self.tools = AgentToolSet(palmimo, include=[*(tool for tool, _ in ACTIONS.values()), "set_face", "stop"])
        self.walking = False
        self.walk_stop = threading.Event()
        self.walk_started = threading.Event()
        self.tools.register(make_walking_tool(self.walk_stop, self.walk_started))
        self.face_tracking = False
        self.camera_frames = CameraFrames(palmimo.camera) if palmimo.camera is not None else None
        self.wave_watch: Any = None
        self.wave_task: asyncio.Task[None] | None = None
        self.last_wave: float | None = None

    async def connect(self) -> None:
        """Open attached resources only after API configuration succeeds."""
        if self.hardware:
            await asyncio.to_thread(self.palmimo.connect)
        if self.wave_watch is not None:
            self.wave_task = asyncio.create_task(self._watch_waves())

    async def _watch_waves(self) -> None:
        """Keep visual detections fresh across voice-session reconnects."""
        async for _ in self.wave_watch.watch():
            self.last_wave = time.monotonic()

    def take_wave(self) -> bool:
        """Consume a recent detection, dropping anything missed while busy."""
        if self.wave_task is not None and self.wave_task.done():
            self.wave_task.result()
        observed, self.last_wave = self.last_wave, None
        return observed is not None and time.monotonic() - observed < 1

    async def wave_back(self) -> None:
        """Reuse the original reflex's happy expression and both-hand wave."""
        await self.call("set_face", {"name": "HAPPY"})
        await self.call("wave_both", {"seconds": 3.0})

    async def idle(self, name: str) -> None:
        """Run a bounded silent gesture through the same SDK cancellation path."""
        if not self.motion_enabled:
            return
        action = IDLE_ACTIONS[name]
        started = time.monotonic()
        if action.face:
            await self.call("set_face", {"name": action.face})
        if action.tool:
            await self.call(action.tool, dict(action.arguments))
        await asyncio.sleep(max(0, action.seconds - (time.monotonic() - started)))

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Propagate device errors so the runtime parks rather than continuing."""
        result = await self.tools.call(name, arguments)
        if result.is_error:
            raise RuntimeError(f"Robot action failed: {result.text}")
        return result.text

    async def react(self, reaction: Reaction) -> None:
        """Use the existing neck-only nod without an agreement glyph."""
        if not self.motion_enabled:
            return
        if reaction.expression:
            await self.call("set_face", {"name": FACES[reaction.expression]})
        if reaction.gesture:
            if reaction.gesture == "clap":
                await self.call("set_face", {"name": "HAPPY"})
                await self.call("clap", {"seconds": 3.0})
            else:
                await self.call("nod" if reaction.gesture == "nod" else "head_shake", {"seconds": 1.8})

    async def action(self, name: str, started: Callable[[], Awaitable[None]] | None = None) -> str:
        """Run requested motions while preserving SDK stop/neutral behavior."""
        if not self.motion_enabled:
            return "Audio-only mode: no physical action was executed."
        if name not in ACTIONS:
            return "No supported motion was requested."
        if name in CONTINUOUS_ACTIONS:
            self.walk_stop.clear()
            self.walk_started.clear()
            self.walking = True
            try:
                async with asyncio.TaskGroup() as group:
                    task = group.create_task(self.call("continuous_walk", {"direction": name}))

                    async def report_start() -> None:
                        while not task.done() and not self.walk_started.is_set():
                            await asyncio.sleep(0.02)
                        if self.walk_started.is_set() and started is not None:
                            await started()

                    group.create_task(report_start())
                return task.result()
            finally:
                self.walking = False
        tool, arguments = ACTIONS[name]
        return await self.call(tool, dict(arguments))

    async def cancel(self) -> None:
        """Cancel through the SDK before awaiting the owning motion task."""
        self.walk_stop.set()
        await self.tools.cancel_running()

    async def stop(self) -> str:
        """Return legs and gaze to the SDK's neutral stance after the previous task settles."""
        if not self.motion_enabled:
            return "Audio-only mode: no physical action was executed."
        await self.call("stop", {})
        return "Motion stopped. Legs and neck returned to the startup stance and are holding it."

    async def close(self) -> None:
        """Park smoothly and release all device handles."""
        try:
            if self.wave_watch is not None:
                await self.wave_watch.aclose()
            if self.wave_task is not None:
                self.wave_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.wave_task
        finally:
            if self.hardware:
                await asyncio.to_thread(self.palmimo.disconnect)


def build_robot(settings: Settings) -> Robot:
    """Configure the ReSpeaker audio path and SDK peripheral ownership."""
    from palmimo_sdk import EchoCanceller, MicStream

    processors = (
        [EchoCanceller(near_channel=settings.near_channel, reference_channel=settings.reference_channel)]
        if settings.echo_cancel
        else []
    )
    mic = MicStream(device_name_hint=settings.mic_device, processors=processors)
    if settings.audio_only:
        return Robot(Palmimo(mic=mic), hardware=True, mic=mic, motion_enabled=False)
    from palmimo_companion_agent.core.tools import make_look_at_face_tool
    from palmimo_companion_agent.core.vision import FaceLocator, VisionWatch, WaveDetector
    from palmimo_sdk import DynamixelDriver, FaceDisplay, HeadCamera

    robot = Robot(
        Palmimo(driver=DynamixelDriver(port=settings.port), display=FaceDisplay(), camera=HeadCamera(), mic=mic),
        hardware=True,
        mic=mic,
    )
    robot.tools.register(make_look_at_face_tool(FaceLocator()))
    robot.face_tracking = True
    robot.wave_watch = VisionWatch(robot.palmimo.camera, [WaveDetector()])
    return robot
