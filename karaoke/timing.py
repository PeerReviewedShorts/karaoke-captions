"""Speech/silence from the audio envelope, and the one timing correction we trust without a second model: a word that
starts in silence is moved to where the voice actually starts.

Parakeet's boundaries sit on an 80 ms grid with a speaker-dependent offset, so after a pause a word can light up 0.1-0.2 s
before it is spoken. The re-entry after silence is unambiguous in the envelope, so snapping to it is safe. Word ends are
left alone: a word that stays lit through the following pause looks natural in karaoke captions.
"""
import numpy as np

from . import media

LEAD = 0.02          # light up 20 ms before the voice, not exactly on it
MAX_SHIFT = 0.40     # never move a start by more than this


def speech_mask(samples):
    """Per 10 ms frame: True where someone is speaking (adaptive threshold from noise floor and peak)."""
    rms = media.rms_envelope(samples)
    if len(rms) == 0:
        return rms.astype(bool), rms
    srt = np.sort(rms)
    floor = np.median(srt[:max(len(srt) // 10, 1)])
    peak = srt[int(0.99 * (len(srt) - 1))]
    return rms > max(floor * 3.0, peak * 0.05), rms


def snap_starts(words, samples, min_silence=0.12):
    """Returns (words, n_moved). A start is moved when the first `min_silence` seconds after it are silent and speech
    begins within MAX_SHIFT and before the word's own end."""
    speech, _ = speech_mask(samples)
    n, out, moved = len(speech), [], 0
    win = int(min_silence * 100)
    for w in words:
        w = dict(w)
        f0 = int(w["s"] * 100)
        if f0 + win <= n and not speech[f0:f0 + win].any():
            hits = np.nonzero(speech[f0:min(n, f0 + int(MAX_SHIFT * 100))])[0]
            if len(hits):
                s_new = round((f0 + int(hits[0])) / 100 - LEAD, 3)
                if s_new < w["e"] - 0.05:
                    w["s"] = s_new
                    moved += 1
        out.append(w)
    for a, b in zip(out, out[1:]):                       # keep order: a word never ends after the next one starts
        if a["e"] > b["s"]:
            a["e"] = b["s"]
    return out, moved
