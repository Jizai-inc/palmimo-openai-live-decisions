# Palmimo OpenAI Live Decisions

Use Python 3.12 and uv. Source comments, docstrings, and documentation are English;
runtime conversation follows the interlocutor's language. Keep robot algorithms in the pinned Palmimo SDK.
Never commit credentials, device configuration, conversation logs, or audio recordings.
Run `uv run ruff check .`, `uv run mypy`, and `uv run pytest` before deployment.
Dry-run before live testing. Hardware startup requires idle devices, a clean servo
diagnostic, and a supported robot with its feet in the air. Always settle running
SDK tools before parking. Do not force-kill the robot process.
