"""Operator commands and API preflight, without import-time device access."""

import argparse
import asyncio
import json
import os
import signal
import sys
import urllib.request
from pathlib import Path

import websockets
from dotenv import load_dotenv

from .audio import NullPlayback
from .config import DECISIONS_URL, LIVE_URL, Settings, reaction_questions
from .events import EventLog
from .idle import IdleDecider
from .policy import Conversation
from .robot import build_robot
from .runtime import ProtocolError, Session, close_connection, run, start_connection


async def check_api(key: str, settings: Settings, log: EventLog) -> None:
    """Check the actual model and trigger configuration without opening devices."""
    from palmimo_sdk import Palmimo

    from .robot import Robot

    def decide() -> dict:
        request = urllib.request.Request(
            DECISIONS_URL,
            data=json.dumps(
                {"model": "gpt-6-luna", "input": "パルミーモ、どう思う？", "questions": reaction_questions()},
                ensure_ascii=False,
            ).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)

    decision = await asyncio.to_thread(decide)
    log.write("decisions_verified", answers=decision.get("answers"))
    if settings.idle_enabled:
        idle_action = await IdleDecider(key).choose("人が2人で話している。", "机の前に人がいる。", True)
        if idle_action is None:
            raise ProtocolError("Idle decision returned no supported choice")
        log.write("idle_verified", action=idle_action)
    session = Session(settings, Robot(Palmimo()), log)
    async with websockets.connect(
        LIVE_URL, additional_headers={"Authorization": f"Bearer {key}"}, open_timeout=20
    ) as ws:
        history = Conversation()
        history.append("user", "これから英語で話してね。")
        history.append("assistant", "Sure, let's speak English.")
        await start_connection(ws, settings, history.history())
        try:
            async with asyncio.timeout(1):
                while True:
                    event = json.loads(await ws.recv())
                    if event.get("type") == "error":
                        raise ProtocolError(str(event.get("error", {}).get("code")))
        except TimeoutError:
            pass
        log.write("live_verified", model="gpt-live-1", native_decisions=True, initial_history=True)
        await close_connection(ws, session, NullPlayback())


async def operate(settings: Settings, key: str, log: EventLog) -> None:
    """Keep SIGINT/SIGTERM attached until parking has finished."""
    stop = asyncio.Event()
    commands: asyncio.Queue[str] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    def read_command() -> None:
        text = sys.stdin.readline()
        if text:
            command = text.strip().lower()
            if command == "q":
                stop.set()
            else:
                commands.put_nowait(command)
        else:
            loop.remove_reader(sys.stdin.fileno())

    reader = False
    if sys.stdin.isatty():
        loop.add_reader(sys.stdin.fileno(), read_command)
        reader = True
    try:
        robot = build_robot(settings)
        log.write(
            "operator",
            commands="q=exit, x=stop motion, i=resume idle, m=mute input, l=listen, s=status",
            seconds=settings.seconds,
            echo_cancel=settings.echo_cancel,
        )
        await run(settings, key, robot, log, stop, commands)
    finally:
        if reader:
            loop.remove_reader(sys.stdin.fileno())
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


def parser() -> argparse.ArgumentParser:
    """Expose independent input/output device selection and no default time limit."""
    result = argparse.ArgumentParser(description="Palmimo GPT-Live conversational companion")
    result.add_argument("command", choices=["run", "check-api"])
    result.add_argument("--env-file", type=Path)
    result.add_argument("--log-path", type=Path)
    result.add_argument("--voice", default="marin")
    result.add_argument("--mic-device", default="ReSpeaker")
    result.add_argument("--speaker-device", default="ReSpeaker")
    result.add_argument("--near-channel", type=int, default=0)
    result.add_argument("--reference-channel", type=int, default=5)
    result.add_argument("--no-echo-cancel", action="store_true")
    result.add_argument("--audio-only", action="store_true")
    result.add_argument("--no-idle", action="store_true", help="Disable autonomous gestures for audio comparison")
    result.add_argument("--port")
    result.add_argument("--seconds", type=float, default=0, help="0 runs until manually stopped (default)")
    return result


def main() -> None:
    """Start the requested mode and return a nonzero status on any failure."""
    args = parser().parse_args()
    if args.seconds < 0:
        raise SystemExit("--seconds must be nonnegative")
    if args.env_file:
        load_dotenv(args.env_file, override=False)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is required")
    settings = Settings(
        voice=args.voice,
        mic_device=args.mic_device,
        speaker_device=args.speaker_device,
        near_channel=args.near_channel,
        reference_channel=args.reference_channel,
        echo_cancel=not args.no_echo_cancel,
        audio_only=args.audio_only,
        idle_enabled=not args.no_idle,
        port=args.port,
        seconds=args.seconds,
    )
    log = EventLog(args.log_path)
    try:
        asyncio.run(check_api(key, settings, log) if args.command == "check-api" else operate(settings, key, log))
    except Exception as exc:
        log.write("failed", error=type(exc).__name__, message=str(exc).replace(key, "[REDACTED]")[:800])
        raise SystemExit(1) from None
    finally:
        log.close()
