# Web search

Palmimo can look up current information while continuing to listen. GPT-Live
delegates an explicit lookup to Decisions; the application runs a separate
Responses request with the hosted `web_search` tool and returns a short answer
to GPT-Live. Speech uses the selected Live voice and the conversation language.
The existing camera, songbook, motion and audio recovery paths remain available.

## Use it

Search is enabled by default with the existing `OPENAI_API_KEY`. The project
needs Responses access and access to the search model (`gpt-6-luna` by default).
Search calls incur additional model/tool usage. There is no separate search
provider credential. `--no-web-search` disables the feature; Portal exposes
the equivalent `no_web_search` parameter.

Examples addressed to Palmimo:

- "東京の明日の天気を調べて" / "Look up tomorrow's weather in Tokyo."
- "OpenAIはどんな会社？ 調べて" / "Look up what OpenAI does."
- "やっぱり大阪で" / "Actually Osaka, not Tokyo."
- "その会社の製品は？" / "What are their products?"
- "調べなくていいよ" / "Cancel the search."

The app acknowledges once, keeps listening, and speaks a short sourced answer
at a conversational opening. Background city/company mentions are not lookup
requests. Ambiguous places or company names should prompt clarification.
Weather answers must match the requested place/date and distinguish a forecast
from an observation. Company lookups prefer official websites and investor
relations. Retrieval time is retained separately from a source's publication or
forecast issue time. Missing evidence is not replaced with a guess.

Relative dates use `--search-timezone Asia/Tokyo` by default. Set another IANA
timezone for another venue. Timezone is not geolocation: a request for "here"
needs an explicit place. Initial weather lookup uses the same hosted web tool;
it is not a dedicated weather-data integration.

## Inspect without devices

```bash
uv run --frozen palmimo-openai-live-decisions check-search --env-file .env \
  --query "Look up tomorrow's weather in Tokyo."
```

This is a real, potentially billable request. It opens no microphone, speaker,
camera, display, or servo bus. The output includes the answer, clickable Markdown
source citations and retrieval time. `check-api` separately validates the Live
session and Decisions configuration; it does not establish search access.

During an interactive run, `w` + Enter shows the latest delivered lookup and its
sources. This record is kept in memory only and survives a Live reconnection,
but not application exit. Portal does not provide an interactive terminal or
a new search-results page. If a UI displays these results, preserve the prominent
inline clickable citations as required by the
[OpenAI search guide](https://developers.openai.com/api/docs/guides/tools-web-search#output-and-citations).
Voice mentions source names and relevant dates rather than reciting URLs.
The API can also return named data feeds (`oai-weather`, `oai-finance`, or
`oai-sports`) instead of web pages. Their source names are retained explicitly;
no URL is invented when the API supplies none. These are accepted only from
the completed tool call's source metadata, never from model prose.

## Lifecycle and limits

- Search runs outside the audio/event receiver, with a reused HTTP connection
  pool, a 30-second request deadline, at most three tool calls, and bounded
  context/output. There is one current lookup; a replacement cancels the older
  client request. Cancellation cannot guarantee upstream computation/billing stops.
- A new transcript pauses answer delivery until its fresh Decisions result has
  classified it as keeping, replacing or cancelling the lookup. Out-of-order
  decisions cannot undo a newer classification. Unclassified speech suppresses
  delivery after an eight-second wait instead of risking a stale answer.
- Quiet listening, mute, stop, a new physical/scene/song request, audio recovery,
  and connection teardown invalidate pending lookup delivery. Reconnection never
  automatically replays a lookup. A direct fresh search can release quiet mode.
- Network errors, rate limits, unsupported models, incomplete answers and missing
  source metadata fail only that lookup. A current request gets a brief failure
  reply; the Live conversation remains available. Retrying requires a new request.
- Only completed search responses with API-provided HTTP(S) sources or one of
  the named data feeds above are accepted.
  A result includes its sources and retrieval time. Full web pages are not passed
  to Live; short reference chunks stay below the 500-token append limit. Page
  content is reference data and must not initiate robot actions or change policy.
- Native Decisions and speech recognition are model-driven. Fragment timing,
  background conversation and corrections still need supervised acoustic testing.
  An append acknowledgment or `search_completed` means context was submitted,
  not that a person heard the intended answer. Already generated speech cannot
  be retracted; Live's interruption policy handles subsequent spoken corrections.

Search receives at most six recent conversation parts (3,000 characters total)
and the previous delivered lookup for referents. It receives no camera frame or
raw audio. Requests use `store: false`; this does not override the API account's
retention policy. Search diagnostics log status, elapsed time, source count,
retrieval time and request ID, not raw search responses or exception bodies.
Existing conversation transcript logging is unchanged and remains private.

See the [supervised checklist](validation.md) for verification on a robot.
