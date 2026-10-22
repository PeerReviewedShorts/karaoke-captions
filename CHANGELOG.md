# Changelog

## 0.1.1 (2026-10-22)

- **Fix: long audio lost whole sentences.** Parakeet-TDT decoded 2-minute windows, and on a 6-minute narration it silently
  dropped a 15-word sentence. Windows are now at most 60 s (14-, 30- and 60-s windows all kept it).
- `karaoke check` warns about "sound but no words" (2+ s of speech-level audio without a timed word), so lost words can
  not slip through unnoticed again. It warns rather than fails because music without speech looks the same.
- `karaoke-agent sysinfo` marks Whisper model files under 60 MiB as incomplete.

## 0.1.0 (2026-10-22)

First public version.

- `karaoke`: Parakeet-TDT transcription with word timing in one pass, start snapping after pauses, timing checks,
  review/fix workflow, transparent overlay and burn-in.
- `karaoke-agent`: Whisper second opinion (whisper.cpp, mlx-whisper, faster-whisper, openai-whisper), glossary check,
  suggestions with accept/reject log, `sysinfo`/`setup` so an AI agent can pick the Whisper setup for the machine.
- Styles: three presets, any installed or bundled font, variable-font axes, colors as hex/rgb/names, automatic
  fallback font for alphabets the chosen font lacks, long words shrink to fit, `karaoke preview` for single frames.
- Video handling: any frame size, phone rotation flags, non-square pixels, odd sizes, mp4/mov/mkv/webm, audio codecs
  that need converting for mp4, HDR kept as HEVC 10-bit when libx265 is available.
- Agents: `karaoke-agent guide` prints the instructions, `karaoke-agent init` puts them into a clip folder
  (AGENTS.md + Claude Code skill); `sysinfo` lists Whisper models already on disk.
- Hardware: `--device auto` (CUDA, Apple MPS, CPU) with fallback to the CPU when the GPU fails.
