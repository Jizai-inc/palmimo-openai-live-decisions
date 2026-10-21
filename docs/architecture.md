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
frequency independently.

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
pause. Idle actions alternate families, with a body gesture every four to six
idle actions and smaller gaze, neck, or expression changes in between. They never
select continuous walking or generate speech notifications.

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
preserves the mute and motion-hold state. It does not replay actions or recreate
the exact previous audio state. Repeated connection failures park and stop;
configuration and device failures are fatal.

This app does not authenticate nearby people or associate a face with a speaker.
It has no web server, remote command listener, persistent personal memory, or
autonomous navigation. A supervised local operator remains responsible for
physical operation.
