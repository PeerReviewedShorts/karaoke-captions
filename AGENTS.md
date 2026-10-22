# Instructions for AI coding agents (Claude Code, Codex, …)

**Using the tool** (captioning clips): follow [karaoke_agent/guide.md](karaoke_agent/guide.md). It ships with the
package; `karaoke-agent guide` prints it and `karaoke-agent init` copies it into a clip folder.

## Developing this repo

- Set up with `pip install -e ".[dev]"` in `.venv`; then `ruff check .`, `pyright` and `python -m pytest` must pass.
  The tests need no speech model, only ffmpeg.
- Keep tool 1 free of Whisper and of any LLM dependency.
