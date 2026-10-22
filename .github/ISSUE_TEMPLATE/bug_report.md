---
name: Bug report
about: Something went wrong with a clip
---

**What happened** (the command and the last lines it printed):

```
karaoke ...
```

**The video** (output of `ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate:format=duration your_clip.mp4`):

```
```

**Machine:** OS, chip/GPU, Python version, `karaoke --version` (or paste `karaoke-agent sysinfo`).

If the words or their timing are wrong, attach `clip.words.json` and say which word and at what second.
