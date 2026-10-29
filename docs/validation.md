# Manual validation

Run the software checks in the README first. Use a configured robot with a local
operator, feet supported clear of the surface for initial motion tests, no other
Palmimo apps running, and clean SDK servo diagnostics. Stop immediately if the
hardware behaves unexpectedly. Do not infer physical behavior from a successful
API call or a simulated test.

1. Run API preflight, then an audio-only session. Confirm that the selected
   microphone and speaker work and that the robot does not connect its servos.
2. Start a supervised full session. Speak for 20–30 seconds and listen for
   backchannels. Interrupt an answer and check that the correction is used.
3. Have two people discuss a topic, then ask Palmimo for its opinion. Assess
   response timing, interruptions, and whether it speaks too often.
4. Request a nod, wave, and dance. Check that acknowledgment precedes or
   accompanies movement and that dance lasts approximately four seconds.
5. Request walking briefly, then say the stop command. Verify that legs and neck
   return to the startup stance and remain held. Also verify local `x`.
6. Say the idle-resume phrase or make a new motion request. Check that autonomous
   motion resumes and does not audibly disrupt conversation. Observe for at least
   a minute during both silence and conversation: expressions should carry most
   idle reactions, with no repeated neck movement on adjacent transcript fragments.
   Request a nod during the automatic neck cooldown and check that the explicit
   request still executes. Confirm the face visibly changes with context.
7. Wave silently and confirm that Palmimo waves back without announcing the
   gesture. Ask about a visible object, replace the object and ask again, then
   request applause. Assess actual behavior rather than event counts alone.
8. Under supervision with the local stop available, test connection loss. Check
   that motion holds and an old command is not replayed on reconnection.
   Inject one synthetic `decision_failed` at the client event boundary: verify
   audio continues, the health probe is acknowledged, and the session ID stays
   unchanged. Inject three errors within thirty seconds: verify a new session,
   preserved conversational language and quiet state, and no replayed motion.
   Test missing probe acknowledgement, prolonged outage, and stopping during the
   retry delay with the local WebSocket tests before hardware. Synthetic events
   verify client recovery, not the root cause or recoverability of a real API
   failure. Record naturally occurring failures separately.
9. Repeat conversation, action announcements, stop, resume, and scene questions
   in each target language. Switch languages mid-session, explicitly request a
   language, and check that a quoted foreign phrase does not cause a switch.
   Use two speakers with different languages where practical. Check that a
   reconnect preserves the recent language preference without replaying motion.
   Japanese stop phrases and English `stop` use a direct transcript path; other
   languages rely on model delegation, so assess their latency separately.
10. In both Japanese and English, ask Palmimo to just listen, converse with another
   person for 20–30 seconds, then ask "can you hear me?" / "聞こえてる？" without
   its name. Confirm a meaningful audible answer, then immediately ask it to be
   quiet again. Repeat with "please talk" and "what do you think?". While quiet,
   mention Palmimo to someone else and confirm it stays quiet; then address it
   directly by name. Test `i` and the spoken idle-resume command as recovery paths.
   Reconnect while quiet and confirm it remains quiet until a new direct request.
   Check quiet transitions, steering acceptance, and subsequent assistant text
   against what was heard. An acknowledgment event alone is not a successful reply.
11. Install the app through Portal using the PR branch or a release containing
   this manifest. Set the API key locally, start with the selected voice, then
   stop through Portal and verify graceful parking and resource release.
12. End with `q`, wait for `stopped`, and independently verify device release,
   21 responding servos, no hardware errors, and torque off.

Record the app commit, SDK and helper versions, hardware/audio setup, API models,
network type, test duration, audible issues, and shutdown result. Keep real
participant transcripts, images, credentials, hostnames, and network addresses
outside the repository. Use synthetic examples in bug reports.

Prior supervised prototypes exercised conversation, requested movement,
speech-before-motion, autonomous gestures, and neutral-pose stopping. They do
not establish compatibility with every robot/audio setup or an endurance result
for this packaged release. Fresh-environment CI and unit tests establish software
behavior only; publish hardware claims only after the corresponding physical
checks have been completed.
