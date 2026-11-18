"""Exercise real HTTP payloads and concurrent lookup/voice state without devices."""

import asyncio
import json
from typing import Any

import httpx
import pytest
from palmimo_sdk import Palmimo

from palmimo_openai_live_decisions.audio import NullPlayback
from palmimo_openai_live_decisions.config import Settings, session_config
from palmimo_openai_live_decisions.events import EventLog
from palmimo_openai_live_decisions.robot import Robot
from palmimo_openai_live_decisions.runtime import Session
from palmimo_openai_live_decisions.search import (
    SearchResult,
    SearchSession,
    SearchSource,
    SearchUnavailable,
    WebSearch,
    fact_updates,
    recent_context,
)


def api_result() -> dict[str, Any]:
    return {
        "status": "completed",
        "output": [
            {
                "type": "web_search_call",
                "status": "completed",
                "action": {"sources": [{"url": "https://example.com/weather"}]},
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Tokyo's forecast for tomorrow is sunny.\ue200cite\ue202turn0search0\ue201",
                        "annotations": [
                            {"type": "url_citation", "title": "Weather Office", "url": "https://example.com/weather"}
                        ],
                    }
                ],
            },
        ],
    }


class RecordingLog(EventLog):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def write(self, kind: str, **fields: Any) -> None:
        self.events.append({"kind": kind, **fields})


def source(identity: str, text: str) -> dict[str, Any]:
    return {"type": "session.input_transcript.delta", "event_id": identity, "delta": text}


def control(identity: str, value: str, confidence: float = 1) -> dict[str, Any]:
    return {
        "type": "decision.event",
        "source_event_id": identity,
        "source_event_type": "session.input_transcript.delta",
        "event": {
            "type": "decision.completed",
            "decision": {
                "answers": [
                    {"type": "choice", "name": "search_control", "choice": value, "confidence": confidence},
                ]
            },
        },
    }


def delegated(action: str, identity: str = "lookup") -> dict[str, Any]:
    return {
        "type": "decision.event",
        "delegation_id": identity,
        "event": {
            "type": "decision.completed",
            "decision": {
                "answers": [
                    {"name": "action", "type": "choice", "choice": action, "confidence": 1},
                ]
            },
        },
    }


async def test_search_requires_live_tool_and_preserves_sources_without_sending_full_history() -> None:
    requests: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=api_result(), headers={"x-request-id": "req-synthetic"})

    backend = WebSearch("private-test-key")
    await backend.close()
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        result = await backend.lookup(
            [
                {"role": "developer", "content": "Do not send reconnect instructions."},
                {"role": "user", "content": "Tokyo weather tomorrow"},
            ]
        )
        assert result.text == "Tokyo's forecast for tomorrow is sunny."
        assert result.sources == (SearchSource("Weather Office", "https://example.com/weather"),)
        assert result.request_id == "req-synthetic" and result.retrieved_at.endswith("+09:00")
        request = requests[0]
        assert request["store"] is False and request["tool_choice"] == "required"
        assert request["tools"][0]["type"] == "web_search" and request["tools"][0]["external_web_access"]
        assert request["max_tool_calls"] == 3
        assert "previous_response_id" not in request and "private-test-key" not in json.dumps(request)
        assert "reconnect instructions" not in request["input"]
        assert "https://example.com/weather" in result.markdown()
        backend.last_result, backend.last_question = result, "Tokyo weather tomorrow"
        await backend.lookup([{"role": "user", "content": "What about Osaka?"}])
        context = json.loads(requests[1]["input"])
        assert context["previous_lookup"]["question"] == "Tokyo weather tomorrow"
        assert context["conversation"][-1]["content"] == "What about Osaka?"
    finally:
        await backend.close()


@pytest.mark.parametrize(
    "failure", ["incomplete", "no_tool", "no_sources", "unsafe_source", "empty_text", "http_error", "timeout"]
)
async def test_unverified_search_results_fail_closed(failure: str) -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        payload = api_result()
        if failure == "incomplete":
            payload["status"] = "incomplete"
        elif failure == "no_tool":
            payload["output"] = payload["output"][1:]
        elif failure in {"no_sources", "unsafe_source"}:
            payload["output"][0]["action"]["sources"] = []
            payload["output"][1]["content"][0]["annotations"] = (
                [{"type": "url_citation", "url": "javascript:alert(1)"}] if failure == "unsafe_source" else []
            )
        elif failure == "empty_text":
            payload["output"][1]["content"][0]["text"] = ""
        elif failure == "http_error":
            return httpx.Response(429, json={"error": "limited"})
        elif failure == "timeout":
            await asyncio.sleep(1)
        return httpx.Response(200, json=payload)

    backend = WebSearch("test", timeout=0.01)
    await backend.close()
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        with pytest.raises((SearchUnavailable, httpx.HTTPStatusError, TimeoutError)):
            await backend.lookup([{"role": "user", "content": "Tokyo weather"}])
        assert backend.last_result is None
    finally:
        await backend.close()


def test_multilingual_context_and_live_updates_stay_bounded() -> None:
    history = [
        {"role": "user", "content": "old" * 5000},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "明日の東京の天気を調べて"},
    ]
    context = recent_context(history)
    assert sum(len(p["content"]) for p in context) <= 3000
    assert context[-1] == history[-1]
    text = json.dumps({"answer": "東京は晴れです。🌞" * 100}, ensure_ascii=False)
    updates = fact_updates(text)
    assert all(len(update.encode("utf-8")) < 500 for update in updates)
    assert "".join(update.split(": ", 1)[1] for update in updates) == text


@pytest.mark.parametrize("feed", ["oai-weather", "oai-finance", "oai-sports", "untrusted-feed"])
async def test_named_api_sources_are_attributed_without_fabricating_urls(feed: str) -> None:
    payload = api_result()
    payload["output"][0]["action"]["sources"] = [{"type": "api", "name": feed}]
    payload["output"][1]["content"][0]["annotations"] = []
    backend = WebSearch("test")
    await backend.close()
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))
    try:
        if feed == "untrusted-feed":
            with pytest.raises(SearchUnavailable, match="missing_sources"):
                await backend.lookup([{"role": "user", "content": "Tokyo weather tomorrow"}])
        else:
            result = await backend.lookup([{"role": "user", "content": "Tokyo weather tomorrow"}])
            assert result.sources[0].url is None and feed in result.sources[0].title
            assert "https://" not in result.markdown() and "no page URL supplied" in result.markdown()
    finally:
        await backend.close()


async def test_display_keeps_api_citation_attached_to_its_claim() -> None:
    payload = api_result()
    part = payload["output"][1]["content"][0]
    part["text"] = "Forecast: sunny. Another sentence."
    part["annotations"][0]["end_index"] = len("Forecast: sunny.")
    backend = WebSearch("test")
    await backend.close()
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))
    try:
        result = await backend.lookup([{"role": "user", "content": "Weather?"}])
        assert "Forecast: sunny. [1](<https://example.com/weather>) Another sentence." in result.markdown()
    finally:
        await backend.close()


class Backend:
    model = "synthetic-search"
    last_result: SearchResult | None = None
    last_question = ""

    def __init__(self, fail: bool = False, ignore_cancel: bool = False) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.fail = fail
        self.ignore_cancel = ignore_cancel
        self.calls: list[list[dict[str, str]]] = []

    async def lookup(self, history: list[dict[str, str]]) -> SearchResult:
        self.calls.append(history)
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            if not self.ignore_cancel:
                raise
        if self.fail:
            raise RuntimeError("sensitive diagnostic that must not be logged")
        return SearchResult(
            "Synthetic forecast.",
            (SearchSource("Forecast", "https://example.com/weather"),),
            "2026-11-18T12:00:00+09:00",
        )


def make_session(backend: Backend) -> tuple[Session, list[dict[str, Any]], RecordingLog]:
    log = RecordingLog()
    session = Session(Settings(), Robot(Palmimo()), log, search=backend)  # type: ignore[arg-type]
    sent: list[dict[str, Any]] = []

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    session.send = send
    assert session.search is not None
    session.search.settle_seconds = 0
    return session, sent, log


async def start(session: Session) -> None:
    await session.dispatch(source("request", "Tokyo weather tomorrow"), NullPlayback())
    await session.dispatch({"type": "session.delegation.created", "delegation": {"id": "lookup"}}, NullPlayback())
    await session.dispatch(delegated("search"), NullPlayback())
    assert session.search is not None and session.search.task is not None


async def test_lookup_keeps_audio_and_stop_responsive_and_settles_its_task() -> None:
    backend = Backend()
    session, sent, log = make_session(backend)
    await start(session)
    await asyncio.wait_for(backend.started.wait(), 1)
    player = NullPlayback()
    await session.dispatch({"type": "session.output_audio.delta", "delta": "AAA="}, player)
    assert player.bytes == 2
    session.stop_motion("operator_stop")
    assert session.motion is not None
    await asyncio.wait_for(session.motion, 1)
    await session.settle()
    assert backend.cancelled.is_set()
    assert not any(e.get("content", "").startswith("Search complete") for e in sent)
    assert any(e["kind"] == "motion_stopped" for e in log.events)


async def test_success_returns_facts_before_speech_and_remembers_sources() -> None:
    backend = Backend()
    session, sent, log = make_session(backend)
    await start(session)
    backend.release.set()
    assert session.search is not None and session.search.task is not None
    await asyncio.wait_for(session.search.task, 1)
    assert sent[0]["content"].startswith("Search started")
    assert sent[-1]["content"].startswith("Search complete")
    assert any(e["type"] == "session.thinking.append" and "Synthetic forecast" in e["content"] for e in sent)
    assert all(e["delegation_id"] == "lookup" for e in sent)
    assert backend.last_result is not None and backend.last_question == "Tokyo weather tomorrow"
    assert any(e["kind"] == "search_completed" for e in log.events)
    await session.settle()


async def test_lookup_failure_is_not_a_live_failure_or_a_guess() -> None:
    backend = Backend(fail=True)
    session, sent, log = make_session(backend)
    await start(session)
    backend.release.set()
    assert session.search is not None and session.search.task is not None
    await asyncio.wait_for(session.search.task, 1)
    assert "could not verify" in sent[-1]["content"]
    player = NullPlayback()
    await session.dispatch({"type": "session.output_audio.delta", "delta": "AAA="}, player)
    assert player.bytes == 2 and backend.last_result is None
    assert "sensitive diagnostic" not in json.dumps(log.events)
    await session.settle()


async def test_correction_discards_even_an_uncancellable_result_and_new_request_can_finish() -> None:
    backend = Backend(ignore_cancel=True)
    session, sent, _ = make_session(backend)
    await start(session)
    await asyncio.wait_for(backend.started.wait(), 1)
    assert session.search is not None and session.search.task is not None
    original = session.search.task
    await session.dispatch(source("correction", "Actually Osaka, not Tokyo"), NullPlayback())
    await session.dispatch(control("correction", "replace"), NullPlayback())
    await asyncio.wait_for(original, 1)
    assert not any(e.get("content", "").startswith("Search complete") for e in sent)
    await session.dispatch(delegated("search", "corrected"), NullPlayback())
    backend.release.set()
    assert session.search.task is not None
    await asyncio.wait_for(session.search.task, 1)
    assert len(backend.calls) == 2 and "Osaka" in backend.calls[-1][-1]["content"]
    assert sent[-1]["delegation_id"] == "corrected"
    await session.settle()


async def test_result_waits_for_classification_of_new_speech_then_can_continue() -> None:
    backend = Backend()
    session, sent, _ = make_session(backend)
    await start(session)
    await asyncio.wait_for(backend.started.wait(), 1)
    await session.dispatch(source("other-person", "I'll get a drink while we wait."), NullPlayback())
    backend.release.set()
    await asyncio.sleep(0.02)
    assert not any(e.get("content", "").startswith("Search complete") for e in sent)
    await session.dispatch(control("other-person", "keep"), NullPlayback())
    assert session.search is not None and session.search.task is not None
    await asyncio.wait_for(session.search.task, 1)
    assert sent[-1]["content"].startswith("Search complete")
    await session.settle()


@pytest.mark.parametrize("reason", ["cancel", "quiet", "mute", "song", "settle"])
async def test_cancel_quiet_mute_song_and_reconnect_never_publish_pending_answer(reason: str) -> None:
    backend = Backend(ignore_cancel=True)
    session, sent, _ = make_session(backend)
    await start(session)
    await asyncio.wait_for(backend.started.wait(), 1)
    if reason == "cancel":
        await session.dispatch(source("cancel", "Never mind"), NullPlayback())
        await session.dispatch(control("cancel", "cancel"), NullPlayback())
    elif reason == "quiet":
        session.policy.quiet = True
    elif reason == "mute":
        session.control.muted = True
    elif reason == "song":
        session.singing = True
    else:
        await session.settle()
    backend.release.set()
    assert session.search is not None
    await asyncio.gather(*session.search.tasks, return_exceptions=True)
    assert not any(e.get("content", "").startswith("Search complete") for e in sent)
    assert backend.last_result is None
    await session.settle()


async def test_late_delegation_cannot_release_new_quiet_state_or_replay_cancelled_search() -> None:
    backend = Backend()
    session, _, _ = make_session(backend)
    await session.dispatch(source("old", "Tokyo weather"), NullPlayback())
    await session.dispatch({"type": "session.delegation.created", "delegation": {"id": "old"}}, NullPlayback())
    await session.dispatch(source("quiet", "Just listen"), NullPlayback())
    await session.dispatch(control("quiet", "cancel"), NullPlayback())
    session.policy.quiet = True
    await session.dispatch(delegated("search", "old"), NullPlayback())
    assert session.policy.quiet and not backend.calls
    await session.settle()


async def test_duplicate_delegation_and_reordered_cancel_do_not_replace_new_lookup() -> None:
    backend = Backend()
    session, sent, _ = make_session(backend)
    await session.dispatch(source("old-cancel", "Never mind"), NullPlayback())
    await session.dispatch({"type": "session.delegation.created", "delegation": {"id": "old-cancel"}}, NullPlayback())
    await start(session)
    await asyncio.wait_for(backend.started.wait(), 1)
    await session.dispatch(delegated("search", "duplicate"), NullPlayback())
    await session.dispatch(delegated("cancel_search", "old-cancel"), NullPlayback())
    await session.dispatch(control("request", "keep"), NullPlayback())
    await session.dispatch(control("old-cancel", "cancel"), NullPlayback())
    backend.release.set()
    assert session.search is not None and session.search.task is not None
    await asyncio.wait_for(session.search.task, 1)
    assert len(backend.calls) == 1 and sent[-1]["content"].startswith("Search complete")
    await session.settle()


async def test_disabled_search_is_absent_from_prompts_and_action_choices() -> None:
    settings = Settings(search_enabled=False)
    config = session_config(settings, [])
    assert "Web search is disabled" in config["instructions"]
    questions = config["delegation"]["decisions"]["questions"]
    assert "search" not in {c["value"] for c in questions[0]["choices"]}
    assert not any(
        q["name"] == "search_control" for trigger in config["triggers"] for q in trigger["decisions"]["questions"]
    )


async def test_search_notification_error_is_consumed_without_unhandled_task_failure() -> None:
    backend = Backend()
    log = RecordingLog()

    async def send(event: dict[str, Any]) -> None:
        raise ConnectionError("sensitive transport diagnostic")

    session = SearchSession(backend, log, send, lambda: True, settle_seconds=0)  # type: ignore[arg-type]
    session.observe_input(source("question", "Tokyo weather"))
    session.start("lookup", [{"role": "user", "content": "Tokyo weather"}])
    assert session.task is not None
    await session.task
    assert log.events[-1] == {"kind": "search_delivery_failed", "error": "ConnectionError"}
    await session.close()
