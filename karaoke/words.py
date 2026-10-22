"""The word file and everything derived from it: review text, SRT, chunks, text fixes.

clip.words.json = {"words": [{"w": "Google's", "s": 0.0, "e": 0.48}, ...], "provenance": {...}, "fixes": [...]}
Times are seconds from the start of the clip. Text fixes never move the outer edges of a span: the first new word starts
where the first old word started, the last new word ends where the last old word ended.
"""
import difflib
import json
import os
import re
import time


def paths(clip, out_dir=None):
    """Every file the tools write for one clip, next to it (or in out_dir)."""
    base = os.path.splitext(os.path.basename(clip))[0]
    d = out_dir or os.path.dirname(os.path.abspath(clip))
    j = lambda suffix: os.path.join(d, base + suffix)
    return {"words": j(".words.json"), "review": j(".review.txt"), "srt": j(".srt"), "mov": j(".captions.mov"),
            "out": j("_captioned.mp4"), "check": j(".check.json"), "whisper": j(".whisper.json"),
            "disagree": j(".disagree.json"), "suggestions": j(".suggestions.json"), "dir": d, "base": base}


def load(path):
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict) or "words" not in doc:
        raise ValueError(f"{path} is not a karaoke words file")
    return doc


def save(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def norm(w):
    """Comparison form of a word: lower case, letters and digits only."""
    return re.sub(r"[^\w]", "", w.lower())


def find(words, needle, start=0):
    n = [norm(x) for x in needle.split() if norm(x)]
    ws = [norm(w["w"]) for w in words]
    for i in range(start, len(ws) - len(n) + 1):
        if n and ws[i:i + len(n)] == n:
            return i
    return -1


def spread(new_tokens, t0, t1):
    """Distribute [t0, t1] over new words by character length."""
    total = sum(len(p) for p in new_tokens) or 1
    out, t = [], t0
    for j, p in enumerate(new_tokens):
        e = t1 if j == len(new_tokens) - 1 else round(t + (t1 - t0) * len(p) / total, 3)
        out.append({"w": p, "s": round(t, 3), "e": round(e, 3)})
        t = e
    return out


def replace(words, old, new, start=0):
    """Replace the word sequence `old` (matched without case/punctuation) by `new`, any word count on either side.
    Returns (words, info) or raises KeyError if `old` is not found."""
    i = find(words, old, start)
    if i < 0:
        raise KeyError(old)
    k = len([x for x in old.split() if norm(x)])
    span = words[i:i + k]
    t0, t1 = span[0]["s"], span[-1]["e"]
    out = spread(new.split(), t0, t1) if new.strip() else []
    info = {"old": " ".join(w["w"] for w in span), "new": new, "s": t0, "e": t1, "index": i}
    return words[:i] + out + words[i + k:], info


def replace_span(words, i0, i1, new):
    """Replace words[i0:i1] by the words of `new`, keeping the span's outer times. i1 == i0 inserts at a pause or into
    the previous word's time."""
    if i1 > i0:
        t0, t1 = words[i0]["s"], words[i1 - 1]["e"]
        out = spread(new.split(), t0, t1) if new.strip() else []
        info = {"old": " ".join(w["w"] for w in words[i0:i1]), "new": new, "s": t0, "e": t1, "index": i0}
        return words[:i0] + out + words[i1:], info
    edited = [w["w"] for w in words[:i0]] + new.split() + [w["w"] for w in words[i0:]]
    out, changes = apply_text(words, " ".join(edited))
    return out, dict(changes[0] if changes else {"old": "", "new": new}, index=i0)


def apply_text(words, text):
    """Take an edited review text and carry its changes onto the timed words. Unchanged words keep their exact
    times; changed spans get the time of the words they replace; inserted words take a pause or split the previous
    word. Returns (words, list of changes)."""
    new = text.split()
    a, b = [norm(w["w"]) for w in words], [norm(x) for x in new]
    out, changes = [], []
    for op, i0, i1, j0, j1 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            for i, j in zip(range(i0, i1), range(j0, j1)):
                w = dict(words[i])
                if w["w"] != new[j]:                        # punctuation or case only
                    changes.append({"old": w["w"], "new": new[j], "s": w["s"], "e": w["e"]})
                    w["w"] = new[j]
                out.append(w)
        elif op in ("replace", "delete"):
            span = words[i0:i1]
            repl = spread(new[j0:j1], span[0]["s"], span[-1]["e"]) if j1 > j0 else []
            changes.append({"old": " ".join(w["w"] for w in span), "new": " ".join(new[j0:j1]),
                            "s": span[0]["s"], "e": span[-1]["e"]})
            out.extend(repl)
        elif op == "insert":
            prev_e = out[-1]["e"] if out else 0.0
            next_s = words[i0]["s"] if i0 < len(words) else prev_e + 0.3 * (j1 - j0)
            if next_s - prev_e >= 0.1:
                t0, t1 = prev_e, next_s
            elif out:                                       # no pause: share the previous word's time
                t0, t1 = out[-1]["s"], out[-1]["e"]
                prev = out.pop()
                new_part = [prev["w"]] + new[j0:j1]
                out.extend(spread(new_part, t0, t1))
                changes.append({"old": prev["w"], "new": " ".join(new_part), "s": t0, "e": t1})
                continue
            else:
                t0, t1 = 0.0, max(next_s, 0.05)
            out.extend(spread(new[j0:j1], t0, t1))
            changes.append({"old": "", "new": " ".join(new[j0:j1]), "s": t0, "e": t1})
    return out, changes


def log_fix(doc, changes, source):
    doc.setdefault("fixes", []).extend(dict(c, source=source, at=time.strftime("%Y-%m-%dT%H:%M:%S")) for c in changes)


# ------------------------------------------------------------------ derived files
# A group should not end on a word that leans on the next one ("Seven in a | row!" reads worse than "Seven | in a row!").
LEANING = {"a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "from", "with", "and", "or", "but", "my", "your",
           "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "und", "oder", "mit", "von", "zu",
           "im", "am", "auf", "für", "le", "la", "les", "un", "une", "de", "et", "el", "los", "las", "y", "il", "e"}


def chunk(words, max_words=3, max_chars=22, max_gap=0.6):
    """Caption groups: up to max_words, break on sentence ends, on commas after two words and on pauses; a group that
    is cut only for length never ends on an article/preposition/conjunction — those move on to the next group."""
    chunks, cur = [], []
    for i, wd in enumerate(words):
        cur.append(wd)
        text = " ".join(x["w"] for x in cur)
        gap = (words[i + 1]["s"] - wd["e"]) if i + 1 < len(words) else 99
        end_punct = re.search(r"[.!?]$", wd["w"])
        soft_punct = re.search(r"[,:;–]$", wd["w"])
        natural = end_punct or (soft_punct and len(cur) >= 2) or gap > max_gap
        if natural or len(cur) >= max_words or len(text) >= max_chars:
            carry = []
            if not natural:
                while len(cur) > 1 and norm(cur[-1]["w"]) in LEANING:
                    carry.insert(0, cur.pop())
            chunks.append(cur)
            cur = carry
    if cur:
        chunks.append(cur)
    return chunks


def srt_ts(t):
    h, r = divmod(max(t, 0.0), 3600)
    m, s = divmod(r, 60)
    ms = int(round((s % 1) * 1000))
    if ms == 1000:
        s, ms = s + 1, 0
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"


def write_srt(words, path, per_cue=2, max_words=3):
    """SRT for platforms that take a caption file (YouTube): two caption groups per cue."""
    groups, buf = [], []
    for ch in chunk(words, max_words):
        buf.append(ch)
        if len(buf) >= per_cue:
            groups.append(buf)
            buf = []
    if buf:
        groups.append(buf)
    with open(path, "w", encoding="utf-8") as fh:
        for n, grp in enumerate(groups, 1):
            ws = [w for c in grp for w in c]
            fh.write(f"{n}\n{srt_ts(ws[0]['s'])} --> {srt_ts(ws[-1]['e'])}\n{' '.join(w['w'] for w in ws)}\n\n")
    return len(groups)


def review_text(words, width=80):
    """Plain text, wrapped, one paragraph: what a human reads in ten seconds."""
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w["w"]) > width:
            lines.append(cur)
            cur = w["w"]
        else:
            cur = f"{cur} {w['w']}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines) + "\n"


def write_derived(words, p):
    """Review text and SRT from the current words (after every change)."""
    with open(p["review"], "w", encoding="utf-8") as fh:
        fh.write(review_text(words))
    return write_srt(words, p["srt"])
