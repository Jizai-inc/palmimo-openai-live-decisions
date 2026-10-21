# Contributing

This repository is a small experimental reference application. Issues and pull
requests are welcome, but there is no guaranteed support response time or API
stability commitment.

Use Python 3.12 and uv. Follow `AGENTS.md` and the development checks in the
README. Keep source comments and documentation in English; Japanese runtime
conversation is intentional. Keep gait and servo algorithms in the Palmimo SDK.

Describe the observed behavior, expected behavior, software versions, and
reproduction steps. Prefer synthetic conversation examples. Never attach API
keys, `.env` files, private host details, raw audio/images, or participant logs.
For security reports, follow `SECURITY.md` instead of opening a public issue.

Include tests for changes to cancellation, concurrency, motion priority, or
resource lifetime. Run compute-only tests before physical tests. State exactly
which physical checks were performed; passing mocks does not demonstrate
acoustic quality or robot safety. Do not enable automatic hardware access in CI.

Contributions are provided under the repository's Apache-2.0 license. Preserve
third-party notices and identify the source and license of any reused code.
