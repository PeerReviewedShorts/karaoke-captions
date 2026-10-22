# Contributing

Thanks for helping. Bug reports with a short clip (or the `clip.words.json` and the ffprobe output of the video) are
the fastest way to get something fixed.

## Set up

```
git clone https://github.com/PeerReviewedShorts/karaoke-captions && cd karaoke-captions
python3 -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

You need ffmpeg on your PATH for the render tests. The speech model is not needed for the tests.

## Before a pull request

```
ruff check .
pyright
python -m pytest
```

All three run in CI on Linux, macOS and Windows. Please add a test for any bug you fix; `tests/test_media.py` shows
how to generate a tiny video with ffmpeg for it.

## Ground rules

- Tool 1 (`karaoke`) stays free of Whisper and of any LLM or API dependency.
- Word times come from Parakeet. Text fixes keep the time span of the words they replace.
- Nothing may overwrite the user's original clip.
- New presets and fonts need a license that allows redistribution (OFL, Apache, MIT); add the license text next to
  the font.
