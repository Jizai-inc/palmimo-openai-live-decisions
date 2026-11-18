# Prepared songs

The optional songbook plays local recordings as soon as a song is selected and
introduced. Music is prepared beforehand; there is no music-generation request
in the conversation loop. GPT-Live still handles conversation and Decisions
selects an allowlisted song. The conversation voice and the recorded singing
voice are independent. A Japanese recording remains Japanese when requested in
English; the introduction and subsequent conversation follow the person.

## Prepare a songbook

Create or obtain short songs you can use. For example, [Eleven Music](https://elevenlabs.io/docs/overview/capabilities/music)
can generate vocal songs from an original prompt and lyrics. Audition the vocals,
pronunciation, ending, and speaker volume before installing a song. Provider
credentials are needed only for that separate preparation step. They are not
required by this app or sent to the robot for playback.

Keep song files outside Git. This repository does not ship recordings. Convert
each selected file to mono 16 kHz, 16-bit PCM WAV using FFmpeg on your workstation:

```bash
ffmpeg -i original.mp3 -af loudnorm=I=-19:TP=-2:LRA=7 \
  -ar 16000 -ac 1 -c:a pcm_s16le hello.wav
```

Place a `songbook.json` beside the WAV files:

```json
{
  "version": 1,
  "songs": [
    {
      "id": "hello",
      "title": "Hello, Palmimo",
      "description": "A warm, short self-introduction song",
      "language": "Japanese",
      "file": "hello.wav"
    }
  ]
}
```

The first entry is the default for an unspecified song request. Install one to
eight songs, each at most sixty seconds. IDs must be unique lowercase identifiers.
Paths must remain inside the songbook directory, including symlink targets. Files
are validated and preloaded before any hardware opens, so missing or incompatible
assets cannot interrupt a running song. No FFmpeg process is needed on the robot.

Validate without an API key or any hardware access:

```bash
uv run palmimo-openai-live-decisions check-songs --songbook /path/to/songbook.json
```

After the normal hardware checks in the [README](../README.md#quickstart), enable
the songbook:

```bash
uv run palmimo-openai-live-decisions run --env-file .env \
  --songbook /path/to/songbook.json
```

Alternatively set `PALMIMO_SONGBOOK` to that absolute path, including in Portal's
app environment. The files must be readable by the app's OS user. Keep the path
local to your deployment. Omit it to retain ordinary conversation without songs.

## Try it

Ask “歌って” or “Please sing your song.” A named request selects an installed
song. Palmimo gives a brief introduction before playback. During the song it
shows a happy expression and keeps the body neutral; normal conversational
gestures and voice output do not compete with playback. The microphone and echo
canceller stay running to receive stop requests.

Live sends continuous PCM, including silence, and has no output-audio completion
event. The handoff therefore waits for actual voice samples followed by at least
600 ms of near-digital silence in fresh PCM, with a recent introduction transcript
as corroboration. A network gap alone cannot start the song. The app drains the
queued introduction through the speaker before giving the song exclusive output.
A short introduction is requested to limit pauses within that phrase; completion
detection remains a client-side heuristic that must be checked on the speaker.

Say “止めて” or “Stop singing,” or use local `x` in the interactive terminal.
Pending audio, including the ALSA buffer, is discarded. A spoken stop also holds
autonomous movement; conversation stays available. Ask a normal question to
verify an audible response. After natural completion, ordinary conversation and
autonomous behavior resume. Music never resumes automatically after reconnection.

Spoken stopping depends on cloud recognition and the acoustic environment. The
same local controls and hardware supervision requirements as the main app apply.
Test the actual robot speaker: music can leave more echo than speech. Do not infer
successful recognition or audible recovery from a unit test alone.

Diagnostics include `song_introduction_started`, `song_introduction_ready`,
`song_output_drained`, `song_started`, and `song_ended` (completion, cancellation,
introduction timeout, or playback failure). `song_introduction_timeout` reports
what is missing: text, audible onset, fresh audio, or the end of speech. Intro
diagnostics include received/audible/quiet durations and packet age, without
recording the PCM or transcript. The app does not record song PCM or microphone
audio in these events. A playback failure
attempts to reset the speaker and restore conversation; an unrecoverable device
failure still follows the normal safe-shutdown path.

Dynamic composition from conversation is not implemented. Future generated songs
can use this same local format and playback path after preparation succeeds.
