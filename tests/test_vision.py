"""Verify fresh visual context, cancellable requests, and motion arbitration."""

import asyncio
import json
import time
from typing import Any

import httpx
import numpy as np
import pytest
from palmimo_sdk import Palmimo

from palmimo_openai_live_decisions import vision
from palmimo_openai_live_decisions.audio import NullPlayback
from palmimo_openai_live_decisions.config import Settings
from palmimo_openai_live_decisions.events import EventLog
from palmimo_openai_live_decisions.robot import Robot
from palmimo_openai_live_decisions.runtime import Session
from palmimo_openai_live_decisions.vision import CameraFrames, SceneVision, VisionUnavailable


class Camera:
    """A frame source with the SDK consumer interface."""

    def add_consumer(self, consumer: Any) -> None:
        self.consumer = consumer


def test_camera_rejects_frozen_frames_and_encodes_only_the_latest() -> None:
    cv2 = pytest.importorskip("cv2")
    camera = Camera()
    frames = CameraFrames(camera)
    with pytest.raises(VisionUnavailable):
        frames.jpeg()
    camera.consumer(np.zeros((600, 1200, 3), dtype=np.uint8), time.monotonic() - 3)
    with pytest.raises(VisionUnavailable):
        frames.jpeg()
    camera.consumer(np.full((600, 1200, 3), 255, dtype=np.uint8), time.monotonic())
    jpeg, _ = frames.jpeg()
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (384, 768, 3) and decoded.mean() > 250


async def test_image_request_is_stateless_and_returns_model_text(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "A white cup is visible."},
                        ],
                    }
                ],
            },
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(vision.httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(handle), **kw))
    reader = SceneVision(CameraFrames(Camera()), "test-key")
    assert await reader._request(b"jpeg", "What is this?") == "A white cup is visible."
    assert seen[0]["store"] is False and "previous_response_id" not in seen[0]
    assert seen[0]["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")
    assert "test-key" not in json.dumps(seen)


async def test_incomplete_vision_output_is_not_reported_as_an_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"status": "incomplete", "output": []}))
    monkeypatch.setattr(vision.httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    with pytest.raises(VisionUnavailable):
        await SceneVision(CameraFrames(Camera()), "test")._request(b"jpeg", "")


async def test_camera_failure_does_not_reuse_an_old_description() -> None:
    session = Session(Settings(), Robot(Palmimo()), EventLog(), vision=SceneVision(CameraFrames(Camera()), "test"))
    sent: list[dict[str, Any]] = []

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    session.send = send
    await session.observe_scene("look")
    assert "確認できません" in sent[-1]["content"] and "以前の観察" in sent[-1]["content"]
    assert sent[-1]["delegation_id"] == "look"


async def test_vision_wait_keeps_audio_and_stop_available_and_cancels_on_exit() -> None:
    started, cancelled = asyncio.Event(), asyncio.Event()

    class SlowVision:
        async def describe(self, question: str) -> tuple[str, float]:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return "unreachable", 0

    robot = Robot(Palmimo())
    session = Session(Settings(), robot, EventLog(), vision=SlowVision())  # type: ignore[arg-type]
    sent: list[dict[str, Any]] = []

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    session.send = send
    player = NullPlayback()
    await session.dispatch({"type": "session.input_transcript.delta", "delta": "What is this?"}, player)
    await session.delegated(
        {
            "delegation_id": "look",
            "event": {
                "type": "decision.completed",
                "decision": {
                    "answers": [
                        {"name": "action", "type": "choice", "choice": "observe", "confidence": 1},
                    ]
                },
            },
        }
    )
    await asyncio.wait_for(started.wait(), 1)
    await session.dispatch({"type": "session.output_audio.delta", "delta": "AAA="}, player)
    session.stop_motion("test_stop")
    assert session.motion is not None
    await asyncio.wait_for(session.motion, 1)
    assert player.bytes == 2 and "停止しました" in sent[-1]["content"]
    await session.settle()
    assert cancelled.is_set() and session.visual_task is None


async def test_wave_is_discarded_while_an_explicit_motion_is_active() -> None:
    robot = Robot(Palmimo())
    session = Session(Settings(), robot, EventLog())

    async def busy() -> None:
        await asyncio.Event().wait()

    session.schedule(busy)
    previous = session.motion
    robot.last_wave = time.monotonic()
    session.check_wave()
    assert session.motion is previous and not robot.take_wave()
    await session.settle()


async def test_wave_back_uses_safe_sdk_motion_without_notifying_voice() -> None:
    positions: list[dict[str, int]] = []
    robot = Robot(Palmimo(on_step=positions.append))
    session = Session(Settings(), robot, EventLog())
    sent: list[dict[str, Any]] = []

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    session.send = send
    robot.last_wave = time.monotonic()
    session.check_wave()
    assert session.motion is not None and not sent
    await asyncio.wait_for(session.motion, 6)
    assert positions and all(200 <= value <= 3900 for frame in positions for value in frame.values())
    assert not sent
    await session.settle()


async def test_wave_watcher_is_owned_until_robot_shutdown() -> None:
    ready, closed = asyncio.Event(), asyncio.Event()

    class Watch:
        async def watch(self) -> Any:
            ready.set()
            yield object()
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            closed.set()

    robot = Robot(Palmimo())
    robot.wave_watch = Watch()
    await robot.connect()
    await asyncio.wait_for(ready.wait(), 1)
    assert robot.take_wave() and not robot.take_wave()
    await robot.close()
    assert closed.is_set() and robot.wave_task is not None and robot.wave_task.done()
