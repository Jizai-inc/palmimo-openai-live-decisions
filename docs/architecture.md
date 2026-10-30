# Architecture

The voice receiver, vision requests, idle selection, and motor execution run as
separate asynchronous tasks. The SDK owns servo I/O, gait calculation, smoothing,
device reservations, and peripheral lifetime. Models select a fixed action name;
they never supply servo positions or executable code.

## Conversation and decisions

`gpt-live-1` controls speech, backchannels, and interruption timing. Native
Decisions transcript triggers evaluate both human and assistant transcript
fragments with `gpt-6-luna`. The application rejects stale, duplicate,
out-of-order, and low-confidence results, and limits gesture and contribution
frequency independently. Automatic nods, head shakes, and idle gaze share a
20-second minimum interval. A suppressed neck gesture can still change the face
or steer conversation; the interval never blocks an explicitly requested motion.
Explicit neck requests restart that interval for subsequent autonomous movement.

GPT-Live also chooses the conversation language from spoken context and explicit
preferences. Action notices describe intent rather than supply a fixed spoken
phrase. The voice model renders announcements, stop results, and visual facts in
the conversation language; internal notices do not select a language. This adds
no language-detection round trip or language-specific TTS service. Reconnect
history retains the recent multilingual turns and any preference still in that
bounded context, rather than resetting speech to a fixed language.

Explicit motion requests use native Decisions delegation. A separate Decisions
loop chooses silent idle gestures from recent conversation, a recent scene
description, and movement history. It does not wait for a silence threshold.
After each idle action settles, a random two-to-four-second pause gives the robot
time to observe. Speech, reactions, and explicit requests bypass this idle-only
pause. At least three display-only idle actions precede each physical idle action,
including the first one. Expressions can persist across consecutive turns when
the mood is unchanged; physical movement is optional, never forced by a quota.
Curiosity, thought, warmth, surprise, and the neutral face convey reactions without
moving the neck. Idle display changes also update reaction deduplication so a
later conversation reaction is not suppressed because of an obsolete face state.
Idle actions never select continuous walking or generate speech notifications.

A fresh human-input `yield` decision enters quiet listening and appends a
listen-only instruction to Live. Every exit appends an explicit release:
`answer` for a direct address or request to speak, `resume_idle`, or operator `i`.
Names are interpreted in context by Decisions, not matched locally to release
quiet mode. Participation questions classify only the latest human speech,
not a standing listening state or a request already handled earlier.

A compact state notice is appended to native Decisions on listening transitions
and quiet reconnection. Source-event ordering rejects results that predate a
transition; new quiet requests have no timed lockout. Reconnection discards old
source events but preserves quiet state and motion cooldowns. Neither assistant
transcript events nor stale results can change quiet state.

A fresh direct-address release sends one `session.commentary.append` to request
the conversational answer, in addition to updating instructions. An instruction
acknowledgment alone can leave the voice model silent after a short question.
The reply request does not repeat on adjacent `answer` decisions, announce
reconnection, override a newer quiet request, or confirm an unexecuted action.
Restoring quiet state on reconnection sends no speech request.

`quiet_entered`, `quiet_released`, and `quiet_restored` log state changes.
`steering_sent` and `steering_accepted` correlate instruction delivery by event ID.
`quiet_reply_requested` and `quiet_reply_accepted` track the separate speech request.
Acceptance does not prove speech: `quiet_reply_started` marks the first subsequent
assistant transcript fragment and its latency, while audio diagnostics and supervised listening
are still needed to establish audible, meaningful recovery.

## Motion and stopping

Explicit requests preempt spontaneous gestures through SDK cancellation. A
requested action waits for a spoken acknowledgment transcript plus a 500 ms
playback margin. Missing voice onset skips the action after five seconds. This
approximates speech alignment; it is not measured acoustic synchronization.
Stop requests bypass that gate, cancel pending actions, and stream the SDK's
smooth return to the startup stance. The hold survives reconnection.

Walking continues until stopped. Dance runs for four seconds, applause for
three seconds, most gestures for 1.5 seconds, and turns for one second. Missing
capture or server events stop walking after a two-second watchdog interval.
Exit settles tasks before parking and releasing resources.

## Audio and vision

The internal audio path uses ReSpeaker capture and SDK DTLN echo cancellation.
ALSA playback runs off the event loop with a 500 ms buffer and 400 ms startup
threshold. Bounded queueing prevents stale audio accumulating indefinitely;
packet-gap, backlog, and underrun counters expose failures without saving audio.

The public companion example provides local face tracking and hand-wave
detection. A fresh camera frame is interpreted by a stateless Responses request
using `gpt-6-luna`. GPT-Live receives concise text from that request. Frames are
limited to a 768-pixel maximum width, never saved by this app, and rejected when
frozen or stale. Local wave detection does not need a cloud image request.
Automatic wave-back stays silent: completion is recorded in the local event log
without sending a completion notification to the voice model.

## Recovery and boundaries

Recovery keeps at most the latest 12,000 transcript characters in memory and
preserves the mute and motion-hold state. The reaction policy also survives, and a
new connection in quiet listening receives the listen-only instruction again,
because appended instructions do not carry over. It does not replay actions or recreate
the exact previous audio state.

Startup history uses one typed text part per message (`input_text` for user and
developer turns, `output_text` for assistant turns). It keeps the continuity
notice and the most recent turns, capped at 128 messages and 6,000 UTF-8 text
bytes. The byte budget conservatively leaves room for message overhead within
Live's default 8,192-token initial-history limit, including multilingual text.

A single `decision_failed` drops that evaluation while audio and conversation
continue in the same session. Failed trigger/delegation IDs are remembered so a
late result cannot execute the failed action. The app sends an empty
`session.update` as a health probe and requires its matching `session.updated`
within ten seconds. This changes no settings and does not ask the model to speak:
quiet listening is not mistaken for a broken connection. Three failed evaluations
in thirty seconds, a missing probe acknowledgement, or a lost/expired session
require a replacement session. Transport keepalives also detect a lost socket
independently of speech. This cannot prove every upstream subsystem is healthy;
it confirms a responsive session and subsequent successful decisions.

Before reconnecting, producers and motions settle, the robot returns to its
neutral stance, and playback is discarded. Movement stays held until a fresh
request; conversation resumes with recent history and language. Six quick retries
use capped exponential delays, then further attempts wait thirty seconds each
until connectivity returns or the operator stops the app. Configuration,
authentication, protocol/policy, safety closure, and device failures are not
retried. Explicit remote hangup and local stop end the app. `q` and SIGTERM work
during a reconnect delay.

Diagnostics include session IDs, decision IDs and source associations when
available, API error category and parameter, a question-configuration hash,
session closure reason, health-probe latency, retry delays, and recovery latency.
Periodic session-health records report event counts and stream ages, without
audio bytes or complete protocol payloads. Existing transcript logging remains
private; never commit it or share it without review. An error can refer to a
trigger that never produced a source event, so its source association may be
unknown rather than guessed.

This app does not authenticate nearby people or associate a face with a speaker.
It has no web server, remote command listener, persistent personal memory, or
autonomous navigation. A supervised local operator remains responsible for
physical operation.
