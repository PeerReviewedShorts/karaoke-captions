"""Where two transcripts of the same clip disagree, and which words look like a glossary term spelled wrong.

Disagreement kinds: "words" (a real difference, worth a look), "spacing" ("corn hole" vs "cornhole"), "number"
("4" vs "four"). Comparison ignores case and punctuation.
"""
import difflib
import re

from karaoke.words import norm

NUMBER_WORDS = set("""zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion first
second third null eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf zwanzig dreißig hundert tausend million
dollar dollars percent prozent euro euros""".split())


def _kind(a, b):
    if "".join(a) == "".join(b):
        return "spacing"
    num = lambda xs: xs and all(re.fullmatch(r"\d[\d.,]*(st|nd|rd|th|s)?", x) or x in NUMBER_WORDS for x in xs)
    if num(a) and num(b) and any(re.match(r"\d", x) for x in a + b):
        return "number"
    return "words"


def disagreements(words, other_text, context=4):
    """words: Parakeet words (list of {"w", "s", "e"}); other_text: the second transcript as plain text."""
    a = [norm(w["w"]) for w in words]
    other = other_text.split()
    b = [norm(x) for x in other]
    out = []
    for op, i0, i1, j0, j1 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        pa, pb = [x for x in a[i0:i1] if x], [x for x in b[j0:j1] if x]
        if not pa and not pb:
            continue
        left = " ".join(w["w"] for w in words[max(0, i0 - context):i0])
        right = " ".join(w["w"] for w in words[i1:i1 + context])
        mine = " ".join(w["w"] for w in words[i0:i1])
        out.append({"i0": i0, "i1": i1, "parakeet": mine, "whisper": " ".join(other[j0:j1]), "kind": _kind(pa, pb),
                    "s": words[i0]["s"] if i0 < len(words) else None,
                    "context": f"{left} [{mine}] {right}".strip()})
    return out


def load_glossary(path):
    """One term per line; '# …' comments; an optional note in parentheses: 'Apptronik (robot maker)'."""
    terms = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            m = re.match(r"^(.*?)\s*\((.*)\)\s*$", line)
            term, note = (m.group(1), m.group(2)) if m else (line, "")
            if term:
                terms.append({"term": term.strip(), "note": note.strip()})
    return terms


def glossary_hits(words, terms, threshold=0.75):
    """Spans of 1-3 words that look like a glossary term but are not spelled like it ('Aptronics' ~ 'Apptronik')."""
    toks = [norm(w["w"]) for w in words]
    hits = []
    for t in terms:
        tn = norm(t["term"].replace(" ", ""))
        n = len(t["term"].split())
        if len(tn) < 4:
            continue
        exact = set()                                    # word positions where the term is already spelled right
        for size in range(1, n + 2):
            for i in range(0, len(toks) - size + 1):
                if "".join(toks[i:i + size]) == tn:
                    exact.update(range(i, i + size))
        best = {}
        for size in {max(1, n - 1), n, n + 1}:
            for i in range(0, len(toks) - size + 1):
                cand = "".join(toks[i:i + size])
                if len(cand) < 4 or cand == tn or exact.intersection(range(i, i + size)):
                    continue
                stem = cand[:-1] if cand.endswith("s") and not tn.endswith("s") else cand      # possessive / plural
                if stem == tn:
                    continue
                r = max(difflib.SequenceMatcher(None, cand, tn).ratio(), difflib.SequenceMatcher(None, stem, tn).ratio())
                if r >= threshold and r > best.get(i, (0,))[0]:
                    best[i] = (r, size)
        for i, (r, size) in best.items():
            if any(h["i0"] <= i < h["i1"] for h in hits):
                continue
            span = " ".join(w["w"] for w in words[i:i + size])
            hits.append({"i0": i, "i1": i + size, "old": span, "term": t["term"], "note": t["note"], "score": round(r, 2)})
    return sorted(hits, key=lambda h: h["i0"])
