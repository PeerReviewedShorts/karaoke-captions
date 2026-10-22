"""Timing checks before anything gets burned in. `karaoke check` exits with 2 when a check fails; `karaoke run` refuses to
render then (unless --force).

Structure (from the words alone): words after the end of the clip, zero-length words, words that last for seconds, and the
share of word pairs that touch exactly (real speech has pauses; a transcript where every word touches the next is a
sentence span spread over its words, not measured timing).
Energy (against the clip audio): words that light up before the voice (no speech in the first 120 ms after their start),
the start error after clear pauses
(>= 300 ms of quiet, then speech: the re-entry is unambiguous in the envelope, so it serves as ground truth). The envelope
has roughly +-40 ms error of its own; it separates 200 ms from 30 ms, it does not certify 50 ms.
"""
import numpy as np

from . import media, timing

# Lingering is fine (a word stays lit through the pause after it, which is how Parakeet's ends behave on fluent speech);
# lighting up BEFORE the voice is the error people see. So the gates are early starts and the start error after pauses;
# highlight time on silence and touching pairs are reported, not gated.
FAIL = {"past_end": 0, "very_long": 0, "early_start": 0.10, "onset_p50": 0.25}     # more than this fails
WARN = {"long": 0, "early_start": 0.03, "onset_p50": 0.12}
EARLY = 0.12                                                           # s of silence after a word start = "too early"
# (karaoke transcribe already snaps such starts to the voice; early starts left over mean the snap could not decide)
LONG, VERY_LONG = 1.5, 3.0                                             # seconds per word


def structure(words, dur):
    n = len(words)
    lens = [w["e"] - w["s"] for w in words]
    abut = sum(1 for i in range(n - 1) if abs(words[i + 1]["s"] - words[i]["e"]) < 0.011)
    return {"n_words": n, "zero_len": sum(1 for x in lens if x <= 0), "long": sum(1 for x in lens if x > LONG),
            "very_long": sum(1 for x in lens if x > VERY_LONG), "longest_s": round(max(lens, default=0.0), 3),
            "past_end": sum(1 for w in words if dur and w["e"] > dur + 0.05),
            "abut_share": round(abut / max(n - 1, 1), 3)}


def energy(words, samples):
    speech, rms = timing.speech_mask(samples)
    if len(rms) < 50:
        return {}
    srt = np.sort(rms)
    floor = np.median(srt[:max(len(srt) // 10, 1)])
    peak = srt[int(0.99 * (len(srt) - 1))]
    quiet, loud = max(floor * 2.0, peak * 0.03), max(floor * 4.0, peak * 0.10)
    inword = np.zeros(len(rms), dtype=bool)
    for w in words:
        inword[int(w["s"] * 100):min(int(w["e"] * 100) + 1, len(rms))] = True
    sil = float((inword & ~speech).sum()) / max(int(inword.sum()), 1)
    starts = np.array(sorted(w["s"] for w in words))
    errs, run = [], 0
    for f in range(len(rms)):
        if rms[f] < quiet:
            run += 1
            continue
        if run >= 30 and rms[f] > loud and len(starts):
            cand = starts[starts >= f / 100.0 - 0.40]
            if len(cand):
                errs.append(float(cand[0] - f / 100.0))
        run = 0
    win = int(EARLY * 100)
    early = [i for i, w in enumerate(words) if int(w["s"] * 100) + win <= len(speech)
             and not speech[int(w["s"] * 100):int(w["s"] * 100) + win].any()]
    out = {"highlight_on_silence": round(sil, 3), "early_start": round(len(early) / max(len(words), 1), 3),
           "early_words": early[:20], "n_pauses": len(errs)}
    if len(errs) >= 3:
        out["onset_p50"] = round(float(np.percentile(np.abs(errs), 50)), 3)
        out["onset_p90"] = round(float(np.percentile(np.abs(errs), 90)), 3)
    return out


def run(words, clip):
    """Returns {"ok": bool, "failed": [...], "warnings": [...], "metrics": {...}, "offenders": [...]}."""
    info = media.probe(clip)
    m = structure(words, info["duration"])
    if info["has_audio"]:
        m.update(energy(words, media.read_audio(clip)))
    failed = [f"{k} = {m[k]} (limit {v})" for k, v in FAIL.items() if k in m and m[k] > v]
    warnings = [f"{k} = {m[k]} (limit {v})" for k, v in WARN.items()
                if k in m and m[k] > v and not any(x.startswith(k + " ") for x in failed)]
    early = set(m.get("early_words", []))
    def why(i, w):
        if w["e"] > info["duration"] + 0.05:
            return "after the end of the clip"
        if i in early:
            return f"lights up {EARLY:.2f}+ s before the voice"
        return f"lasts {w['e'] - w['s']:.1f} s"
    offenders = [{"i": i, "w": w["w"], "s": w["s"], "e": w["e"], "why": why(i, w)} for i, w in enumerate(words)
                 if w["e"] > info["duration"] + 0.05 or w["e"] - w["s"] > LONG or i in early][:20]
    return {"ok": not failed, "failed": failed, "warnings": warnings, "metrics": m, "offenders": offenders,
            "duration": info["duration"]}
