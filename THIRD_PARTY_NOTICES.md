# Third-party notices

Palmimo OpenAI Live Decisions is Copyright 2026 Jizai Inc. and licensed under Apache-2.0.
This repository does not vendor dependency source, model weights, audio, images,
or native libraries. Python dependencies are installed separately; some models
are downloaded on first use. Their licenses remain in force independently of
this application's license.

The following identifies direct runtime dependencies and the audio/vision models
used by this app. It is not a license inventory for a redistributed operating
system image. An image or bundled binary distribution needs its own complete
dependency review, including native libraries and model assets.

| Component | License | Source |
| --- | --- | --- |
| Palmimo SDK 0.1.3 | Apache-2.0 | [Jizai-inc/palmimo-devkit](https://github.com/Jizai-inc/palmimo-devkit/tree/v0.1.3/packages/palmimo_sdk) |
| Palmimo companion example | Apache-2.0 | [Public pinned source](https://github.com/Jizai-inc/palmimo-devkit/tree/5b545d1a8d5551822460d2aa9edd37c4d02c147f/examples/agents/companion) |
| HTTPX | BSD-3-Clause | [encode/httpx](https://github.com/encode/httpx) |
| NumPy | BSD-3-Clause | [numpy/numpy](https://github.com/numpy/numpy) |
| python-dotenv | BSD-3-Clause | [theskumar/python-dotenv](https://github.com/theskumar/python-dotenv) |
| websockets | BSD-3-Clause | [python-websockets/websockets](https://github.com/python-websockets/websockets) |

The companion's `core.tools` and `core.vision` supply face tracking and hand-wave
detection. The public revision is pinned in `pyproject.toml` and `uv.lock`.
Its full package also installs dependencies for features this app does not use.
Preserve the installed packages' license files. Consult the pinned upstream
[companion notices](https://github.com/Jizai-inc/palmimo-devkit/blob/5b545d1a8d5551822460d2aa9edd37c4d02c147f/examples/agents/companion/THIRD_PARTY_NOTICES.md)
and [SDK notices](https://github.com/Jizai-inc/palmimo-devkit/blob/5b545d1a8d5551822460d2aa9edd37c4d02c147f/packages/palmimo_sdk/THIRD_PARTY_NOTICES.md)
for additional dependencies and binary-distribution obligations.

## Models and audio/vision runtimes

- **DTLN-AEC** echo-cancellation weights: MIT, Nils L. Westhausen,
  [breizhn/DTLN-aec](https://github.com/breizhn/DTLN-aec). The SDK downloads and
  checksums the selected TFLite model pair. Inference uses LiteRT (Apache-2.0).
- **MediaPipe HandLandmarker and FaceDetector** weights and runtime: Apache-2.0,
  Google, [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide).
  The companion downloads the hand and face models on first use.
- **OpenCV** Python package: Apache-2.0,
  [opencv/opencv-python](https://github.com/opencv/opencv-python). Wheels may
  bundle separately licensed libraries, including FFmpeg under LGPL; preserve
  their `LICENSE-3RD-PARTY.txt` when redistributing a configured image.
- **sounddevice**: MIT, [spatialaudio/python-sounddevice](https://github.com/spatialaudio/python-sounddevice);
  its PortAudio backend is BSD-3-Clause.
- **ALSA utilities** (`arecord`/`aplay`) are external host programs, not shipped
  by this repository. Consult the operating-system package's license notices.

OpenAI APIs are hosted services governed by their own service terms. No OpenAI
model weights are included or licensed by this repository. Model availability,
permissions, and usage charges are separate from the source-code license.
