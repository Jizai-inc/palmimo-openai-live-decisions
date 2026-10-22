# Palmimo × OpenAI — GPT-Live + Decisions API

A multilingual conversational reference app for [Palmimo DevKit](https://github.com/Jizai-inc/palmimo-devkit).
GPT-Live listens and speaks concurrently, while the Decisions API selects
expressions, gestures, and occasional contributions to a conversation.

This is an experimental sample maintained by Jizai Inc., not an OpenAI product.
It requires your own OpenAI API access and a configured Palmimo host. It is not
a general-purpose desktop voice assistant or an unattended robot controller.

## What it does

- Listen while speaking, acknowledge a person mid-sentence, and participate in
  a shared conversation without treating every human utterance as a stop command.
- Coordinate speech with SDK motions: announce requested actions before moving,
  dance for four seconds, and keep walking until explicitly stopped.
- React quietly while listening, mainly through the face display, with occasional
  glances, nods, tilts, and stretches. Autonomous neck movements share a 20-second
  minimum interval; requested motions remain immediately eligible.
- Silently wave back at a detected hand wave, answer questions about the current
  camera view, and occasionally clap when the conversation calls for it.
- Prioritize stop requests, reject stale decisions, and settle and park on exit.

## Conversation language

Palmimo follows the language of the person addressing it, including backchannels,
action announcements, stop confirmations, and answers about the camera view. An
explicit language preference takes priority until changed. In a group, it answers
the person addressing it or uses the main conversation language when contributing.
Names, quoted phrases, and backend notices should not trigger language changes.
There is no separate language selector or language-detection API request.

For example, speak Japanese or English, or ask to switch during a session:

```text
パルミーモ、こんにちは
パルミーモはどう思う？
今、目の前に何が見える？
ダンスして
止まって
自由に動いて

Hi Palmimo, what do you think?
What can you see in front of you?
Please dance.
Stop.
You can move freely again.
Let's speak English now.
日本語に戻してね
```

Language adaptation is model-driven, not a guarantee of equal performance in
every language, accent, or mixed-language group. Japanese was used for the prior
supervised hardware sessions. Validate speech, action selection, and stop behavior
in the languages and acoustic conditions you plan to use; see the
[manual validation checklist](docs/validation.md).

## Requirements

- Python **3.12** and [uv](https://docs.astral.sh/uv/).
- A configured Palmimo DevKit Linux host with working servos, face display,
  camera, and internal ReSpeaker microphone/speaker. Install the device drivers
  and services using the [DevKit installation guide](https://github.com/Jizai-inc/palmimo-devkit/blob/v0.1.3/doc/guides/installation.md)
  first. This app does not configure the hardware, Wi-Fi, or system services.
- ALSA tools (`arecord` and `aplay`) and permission to access the audio, camera,
  and servo devices. The live audio backend is Linux-specific; software tests
  can run without peripherals on macOS or Linux.
- An OpenAI project API key with access to `gpt-live-1`, `gpt-6-luna`, Live,
  Decisions (`api.decisions.write`), and Responses (`api.responses.write`).
- Outbound HTTPS/WSS access to OpenAI, and HTTPS access to GitHub, PyPI, and
  model-download hosts during installation and first use.

The SDK is pinned to `0.1.3`. Face tracking and wave detection come from a fixed
commit of the **public** DevKit companion example. No private GitHub dependency
or Jizai credential is required. Initial model downloads are handled by those
dependencies; see [third-party notices](THIRD_PARTY_NOTICES.md).

## Quickstart

Clone and install on the configured robot host:

```bash
git clone https://github.com/Jizai-inc/palmimo-openai-live-decisions.git
cd palmimo-openai-live-decisions
uv sync --frozen --extra robot
cp .env.example .env
chmod 600 .env
```

Edit `.env` locally and set `OPENAI_API_KEY`. Keep the key on the trusted host.
Never commit it, include it in an issue, or place it in a browser client.

Check API access before opening any devices:

```bash
uv run --frozen --extra robot palmimo-openai-live-decisions check-api --env-file .env
```

This performs real, potentially billable API requests. It does not open the
microphone, speaker, camera, display, or servo bus.

Stop every other Palmimo app. First test the microphone and speaker:

```bash
uv run --frozen --extra robot palmimo-openai-live-decisions run --env-file .env \
  --audio-only --seconds 60
```

Before enabling motion, verify the hardware using the
[SDK diagnostics](https://github.com/Jizai-inc/palmimo-devkit/blob/v0.1.3/scripts/README.md):
all 21 servos must respond, report no hardware errors, and have torque off.
Support the robot with its feet clear for initial tests and supervise it.
Then run in an interactive terminal:

```bash
uv run --frozen --extra robot palmimo-openai-live-decisions run --env-file .env
```

If needed, select the serial device with `--port /dev/ttyACM0`. When controlling
the app over SSH, allocate a terminal (`ssh -tt`) so local operator commands work.

**There is no default time limit.** Type `q` and press Enter to end the app.
Ctrl+C and SIGTERM also request graceful shutdown. Wait for `stopped` before
disconnecting power or launching another app. Never use a forced process kill
as the normal stop procedure. After a hardware session, verify device release
and all 21 servos' torque-off state using the SDK diagnostics.

## Stop and operator controls

| Input | Effect |
| --- | --- |
| Say `止まって` or `Stop` | Return legs and neck to their startup stance and hold autonomous motion; conversation remains active |
| Say `自由に動いて` / `You can move freely again`, or request a new motion | Resume movement |
| `s` + Enter | Show connection, audio diagnostics, and motion state |
| `x` + Enter | Return to stance and hold motion while keeping the conversation open |
| `i` + Enter | Release the motion hold and resume idle gestures |
| `m` + Enter | Hold motion and mute input; already generated speech may continue |
| `l` + Enter | Resume input |
| `q` + Enter | Close Live, settle motion, park, release devices, and exit |

The app has a direct transcript stop path for Japanese stop phrases and English
`stop`; other languages depend on model-based action delegation. These paths do
not promise identical recognition or latency across languages.

A spoken stop is cloud-dependent and is not a physical emergency stop.
Keep a local operator stop available. Connection loss holds motion; reconnecting
does not replay an old action. Microphone and server-event watchdogs stop
continuous walking when their streams stall.

## Configuration

Use `uv run palmimo-openai-live-decisions --help` for the complete CLI.

| Option | Default / purpose |
| --- | --- |
| `--env-file` | Optional environment file; a process-level `OPENAI_API_KEY` is also accepted |
| `--port` | SDK serial-port discovery; specify a device when required |
| `--voice` | `marin` |
| `--seconds` | `0`, meaning until manually stopped; a positive value bounds a test |
| `--audio-only` | Disable servos, display, and camera |
| `--no-idle` | Disable autonomous gestures for comparison |
| `--mic-device` / `--speaker-device` | `ReSpeaker`; input and output selection are independent |
| `--near-channel` / `--reference-channel` | `0` / `5` for the internal ReSpeaker path |
| `--no-echo-cancel` | Disable software echo cancellation when using a separately validated audio path |
| `--log-path` | Optional JSONL file containing conversation text and diagnostics |

External audio needs its own acoustic validation. DTLN echo cancellation requires
the actual loudspeaker signal on a loopback capture channel; choosing an unrelated
USB speaker does not provide that signal. See [architecture](docs/architecture.md)
for playback buffering and the motion scheduler.

## Data, API access, and cost

Microphone audio is sent to OpenAI during a session. Conversation text is used by
Decisions for reactions and idle selection. With the camera enabled, current JPEG
frames are sent to the Responses API approximately every ten seconds and on a
visual request. Each image request uses `store=false`; that setting is not a
promise of zero retention across all OpenAI services.

This app does not save raw audio or camera images. **Conversation text is printed
to the terminal**, so terminal capture or a service manager may retain it. Optional
`--log-path` files also contain conversation text and are created with owner-only
permissions. Those files are ignored by Git. Inform participants about the cloud
audio/image processing and avoid attaching real conversation logs to issues.

Live sessions are billed by duration; Decisions and vision requests incur
additional usage. Billing continues while an open session is quiet. Check
[OpenAI pricing](https://developers.openai.com/api/docs/pricing) and use explicit
test durations and project usage controls. This app has no built-in spending cap.
The Decisions API is currently in public beta, so API compatibility may change.

## Development and validation

The software suite uses simulated robot I/O and mocked network transports. It
does not need an API key or robot:

```bash
uv sync --frozen --group dev
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen mypy
uv run --frozen pytest -q
uv build
```

CI also checks hardware-dependency installation and importability on Linux.
Always run compute-only tests before testing changes on a supported physical
robot. See [contributing](CONTRIBUTING.md) and the
[manual validation checklist](docs/validation.md).

## Limitations

- Multiple people share one transcript. There is no stable speaker identification,
  acoustic localization, or separation of simultaneous voices. Face tracking
  follows the largest face, not the current speaker.
- GPT-Live receives text descriptions from a separate vision model, not camera
  frames directly. Visual descriptions can be wrong or stale.
- Backchannels, spontaneous participation, and reactions are model-dependent.
  Automated tests do not establish social appropriateness or acoustic quality.
- Playback buffering reduces jitter but adds latency. Occasional audio underruns
  remain possible. Noisy venues, external audio, and endurance need independent
  validation on the target setup.
- SDK safety guards and graceful shutdown do not constitute a whole-robot thermal
  or physical safety guarantee. Supervise the hardware, especially during walking.

## License and references

Application code is licensed under [Apache-2.0](LICENSE). Dependencies and
downloaded models retain their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
OpenAI services and model access are separate from this source-code license.

- [GPT-Live](https://developers.openai.com/api/docs/guides/live)
- [Decisions voice integration](https://developers.openai.com/api/docs/guides/decisions-voice)
- [Prompting and backchannels](https://developers.openai.com/api/docs/guides/live-prompting)
- [WebSocket audio](https://developers.openai.com/api/docs/guides/voice-websockets?api=live)
