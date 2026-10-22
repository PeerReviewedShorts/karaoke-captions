# karaoke-captions: instructions for AI coding agents (Claude Code, Codex, …)

`karaoke-agent guide` prints this file for the installed version. `karaoke-agent init` copies it into the current
folder as `AGENTS.md` (plus `CLAUDE.md` and a Claude Code skill), so the next agent session finds it on its own.

This tool makes word-by-word karaoke captions for short videos. You drive it from the shell. Everything is plain
files next to the clip, every command takes `--json`, and exit codes tell you what happened. Progress, suggestions and
hints go to **stderr**; stdout stays empty unless you pass `--json` (then it holds the result as JSON).

There are two tools:

- **`karaoke`** (tool 1): NVIDIA Parakeet transcribes the clip and times every word in one pass, offline. It then checks,
  renders and burns the captions.
- **`karaoke-agent`** (tool 2): adds a second opinion. Whisper transcribes the same clip, and every word where the two
  models disagree becomes a suggestion. A glossary catches names both models got wrong. You (or the user) accept or
  reject each suggestion.

## Never

- **Never edit word times by hand.** Text fixes go through `karaoke fix` or `karaoke-agent decide`. Both keep the time
  span of the replaced words.
- Never overwrite or delete the original clip.
- Never guess a name that is not in the glossary. Leave the suggestion open and ask the user.
- **Never download a model or install a package without asking first.**
  - Models are 0.15–3 GB. Tell the user the name, the source and the size, and wait for a yes.
  - The exception is when the user has already said yes for this session.
- Never upload or publish anything. This tool only writes files.

## 1. Set up Whisper once per machine (you decide which one)

```
karaoke-agent sysinfo --json
```

It reports the OS, chip, RAM, GPUs (NVIDIA VRAM, Apple Metal), the torch devices, free disk space, which Whisper
backends are installed, which Whisper models are already on disk (`whisper_models`: no download needed for those; skip entries marked
`incomplete`),
and how each backend could be installed here. Pick the setup that fits; the table below is guidance,
not a rule. Then record your choice and **why**:

```
karaoke-agent setup --backend whisper.cpp --model large-v3-turbo --language en \
    --reason "Apple M1, 16 GB unified memory, no CUDA: whisper.cpp with Metal"
```

| Machine | Usually fits | Model |
|---|---|---|
| macOS, Apple Silicon | `whisper.cpp` (Homebrew, Metal) or `mlx-whisper` | `large-v3-turbo`; on 8 GB RAM `small` or `medium` |
| Linux/Windows with an NVIDIA GPU | `faster-whisper` (CUDA, `--compute-type float16`) | `large-v3` with plenty of VRAM, else `large-v3-turbo` or `int8_float16` |
| CPU only (Intel Mac, older laptop, no GPU) | `faster-whisper` with `--compute-type int8`, or `whisper.cpp` | `small` or `medium`; `large-v3-turbo` if time does not matter |
| Nothing can be installed | skip the second opinion | use only the glossary check |

- **Check the numbers before relying on them.** VRAM needs and speeds change between versions, so verify them on the
  project pages.
- **whisper.cpp model files:**
  - Pass `--model` either as a name, looked up in `KARAOKE_WHISPER_MODELS`, `~/.cache/whisper.cpp`, …, or as a path to a
    `ggml-*.bin` file.
  - `setup` stores the file it found in `karaoke.toml`.

`setup` fails with exit code 1 if the backend or the model is missing, and prints the install command. Install only
after asking the user.

## 2. Caption a clip

```
karaoke-agent run clip.mp4 --glossary glossary.txt      # Parakeet + Whisper + glossary + timing check
karaoke-agent list clip.mp4                             # open suggestions
karaoke-agent decide clip.mp4 --accept 1,3 --reject 2 --edit "3=Apptronik's"
karaoke run clip.mp4 [--preset amber|bold|clean]        # render + burn -> clip_captioned.mp4
```

| Exit code | Meaning |
|---|---|
| 0 | done |
| 1 | error (message on stderr; `{"error": …}` with `--json`) |
| 2 | a timing check failed (`clip.check.json` lists the offending words) |
| 3 | suggestions are waiting for a decision |

### Style requests ("bigger", "yellow", "use my font")

Use the style options instead of editing presets: `--font` (path or installed font name), `--size`, `--color`,
`--accent`, `--outline`, `--outline-width`, `--plate` (`#000000B0` or `none`), `--uppercase/--no-uppercase`, `--y`
(the **bottom edge** of the caption block, as a fraction of the height or in pixels), `--max-words` (`1`-`2` for a
one-line look). Check a style before rendering the whole clip:

```
karaoke preview clip.mp4 --at 4.2 --font "Arial Black" --accent "#00FF88"     # -> clip.preview.png, look at it
```

**Always look at one or two preview frames before rendering**, at moments where the video shows its own text
(titles, lower thirds, labels). If the captions sit on it, move them with `--y` and tell the user why.

If the output warns that the font has no glyph for some characters, a fallback font was used; tell the user and offer
a font that covers the language. For a style the user wants to keep, write a preset JSON
(`karaoke presets --show bold > style.json`, edit, `--preset style.json`).

### Videos and hardware

Phone rotation, non-square pixels, odd sizes, webm/mkv/mov and HDR are handled by `karaoke burn`; do not pre-convert
the clip. Parakeet runs on `--device auto` (CUDA, then Apple MPS, then CPU) and falls back to the CPU on a GPU error.
On an Intel Mac Parakeet cannot be installed (no PyTorch 2.5+ builds); say so instead of trying workarounds.

### The glossary

The user knows the names. Ask them for a few lines if there is no glossary:

```
# names, brands, topic
Apptronik (robot maker)
Gemini 4 Argon
```

### Deciding suggestions

Read `clip.review.txt`, `clip.disagree.json` and `clip.suggestions.json`.

- **Glossary hits:** accept them when the context fits the note. Adjust the form with `--edit`: possessive, plural,
  case.
- **Whisper disagreements:**
  - Accept Whisper's word when Parakeet's version is ungrammatical or implausible in context, e.g. "watched …, *map* the
    scene" → "mapped".
  - Reject when Parakeet's version is fine and Whisper's is only a different spelling or style.
- **Numbers** (`kind: number`, e.g. "4" vs "four") and **spacing** ("corn hole" vs "cornhole") are not suggestions. Pick
  the form the user prefers, otherwise leave them alone.
- **If you are not sure:**
  1. Leave the suggestion open.
  2. Report it to the user with its time (`s` in the suggestion or disagreement) so they can listen to that second.
  3. Do not render until it is settled, or render anyway only if the user says so.
- **Your own findings:** file them with `karaoke-agent suggest clip.mp4 "old|new" --why "…"` and decide them like the
  others. Never edit `clip.words.json` directly.

### Several clips

Run the steps per clip and collect the open questions. Ask the user once at the end, with clip name, time and both
versions, instead of stopping after every clip.

### When a check fails (exit 2)

Look at `clip.check.json`:

| Problem | What to do |
|---|---|
| `past_end` (words timed after the clip ends) | Usually a wrong clip or a corrupt audio track. Re-run with `--retranscribe` and report if it stays. |
| `very_long` (a word lasting more than 3 s) | Usually music or a long held sound. Report it. |
| `early_start` (words that light up before the voice) | Snapping already ran. Report the words and their times. |

`karaoke run --force` renders despite the failed check (it does not transcribe again; that is `--retranscribe`). Use
it only if the user agrees.

## Without an agent

`karaoke-agent run … --llm` asks an LLM proofreader for suggestions. It needs `pip install anthropic` and
`ANTHROPIC_API_KEY` or `ant auth login`.

Inside Claude Code or Codex you do not need it: you are the proofreader.

## Files per clip

| File | What |
|---|---|
| `clip.words.json` | `{"words": [{"w", "s", "e"}], "provenance", "fixes"}`, the source of truth |
| `clip.review.txt` | Plain text for reading. `karaoke fix clip.mp4 --from-review` carries edits made here back onto the timed words. |
| `clip.srt` | Captions for platforms that take a file |
| `clip.check.json` | Timing check result |
| `clip.whisper.json`, `clip.disagree.json` | Second opinion and where it differs |
| `clip.suggestions.json` | Every suggestion with source and status (`open`, `accepted`, `rejected`, `stale`) |
| `clip.captions.mov` | Transparent overlay (qtrle with alpha) |
| `clip_captioned.mp4` | The result |
| `karaoke.toml` | The chosen Whisper setup and why (working folder) |
