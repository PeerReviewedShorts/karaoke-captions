# Security

`karaoke` runs locally: it reads media files, calls ffmpeg and ffprobe, and downloads the Parakeet model from Hugging
Face on first use. `karaoke-agent --llm` sends the transcript (not the audio or video) to the Anthropic API, and only
when you pass `--llm`.

If you find a security problem, for example a crafted file name or preset that makes the tool run commands or write
outside the clip's folder, please report it privately through GitHub's "Report a vulnerability" button (Security tab)
instead of a public issue. We aim to answer within a week.
