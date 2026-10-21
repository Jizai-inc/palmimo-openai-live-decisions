"""Cancellable continuous walking through the public Palmimo facade."""

import contextlib
import threading
from typing import ClassVar, Literal

from palmimo_sdk.agent import PalmimoLike
from palmimo_sdk.agent.tools import Tool, ToolResult
from palmimo_sdk.robot import MotionCancelled


def make_walking_tool(stop: threading.Event, started: threading.Event) -> type[Tool]:
    """Keep the gait continuous while closing the gap between SDK cancellation scopes."""

    class ContinuousWalk(Tool):
        """Walk until explicitly stopped; all gait computation stays in the SDK."""

        name: ClassVar[str] = "continuous_walk"
        description: ClassVar[str] = "Walk continuously until a stop request or controller failure."
        long_running: ClassVar[bool] = True
        direction: Literal["forward", "backward", "creep", "strafe_left", "strafe_right"]

        def execute(self, robot: PalmimoLike) -> ToolResult:
            """Use short cancellable runs without resetting the motion between them."""
            if stop.is_set():
                return ToolResult(text="Walking was cancelled before starting.")
            try:
                if self.direction == "forward":
                    robot.forward()
                elif self.direction == "backward":
                    robot.backward()
                elif self.direction == "creep":
                    robot.creep()
                elif self.direction == "strafe_left":
                    robot.strafe_left()
                else:
                    robot.strafe_right()
                while not stop.is_set():
                    robot.run(seconds=0.25)
                    started.set()
            except MotionCancelled:
                robot.stop()
                with contextlib.suppress(MotionCancelled):
                    robot.run(seconds=0.5)
                raise
            finally:
                robot.stop()
            robot.run(seconds=0.5)
            return ToolResult(text="Walking stopped.")

    return ContinuousWalk
