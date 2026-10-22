"""clip.suggestions.json: every proposed text fix with its source and its fate.

    {"suggestions": [{"id": 1, "i0": 41, "i1": 42, "old": "Aptronics", "new": "Apptronik's", "why": "...",
                      "source": "glossary" | "whisper" | "llm" | "agent", "status": "open" | "accepted" | "rejected" | "stale"}]}

Fixes only ever change text. The time span of the replaced words stays where it is (karaoke.words.replace_span).
"""
import os
import time

from karaoke import words as wd


def load(p):
    if os.path.exists(p["suggestions"]):
        return _read(p["suggestions"])
    return {"suggestions": []}


def _read(path):
    import json
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save(p, doc):
    import json
    tmp = p["suggestions"] + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, p["suggestions"])


def add(doc, items, source):
    """Add suggestions, skipping ones already present (same span, same new text). Returns the added ones."""
    have = {(s["i0"], s["i1"], wd.norm(s["new"])) for s in doc["suggestions"]}
    nid = max((s["id"] for s in doc["suggestions"]), default=0)
    added = []
    for it in items:
        key = (it["i0"], it["i1"], wd.norm(it["new"]))
        if key in have or wd.norm(it["old"]) == wd.norm(it["new"]):
            continue
        nid += 1
        s = {"id": nid, "i0": it["i0"], "i1": it["i1"], "old": it["old"], "new": it["new"], "why": it.get("why", ""),
             "source": source, "status": "open", "added": time.strftime("%Y-%m-%dT%H:%M:%S")}
        doc["suggestions"].append(s)
        have.add(key)
        added.append(s)
    return added


def open_items(doc):
    return [s for s in doc["suggestions"] if s["status"] == "open"]


def decide(words, doc, accept=(), reject=(), edits=None):
    """Apply accepted suggestions (back to front so indices stay valid), mark rejected ones. `edits` {id: new text}
    lets the decider adjust a suggestion before accepting it ("Apptronik" -> "Apptronik's"). A suggestion whose old
    words are no longer at its position (an earlier fix moved them) is looked up by text, else marked stale."""
    edits = edits or {}
    byid = {s["id"]: s for s in doc["suggestions"]}
    for i in reject:
        if i in byid and byid[i]["status"] == "open":
            byid[i]["status"] = "rejected"
    applied = []
    for s in sorted((byid[i] for i in accept if i in byid and byid[i]["status"] == "open"), key=lambda s: -s["i0"]):
        new = edits.get(s["id"], s["new"])
        here = " ".join(w["w"] for w in words[s["i0"]:s["i1"]])
        if wd.norm(here) == wd.norm(s["old"]):
            words, info = wd.replace_span(words, s["i0"], s["i1"], new)
        else:
            try:
                words, info = wd.replace(words, s["old"], new)
            except KeyError:
                s["status"] = "stale"
                continue
        s.update(status="accepted", new=new, decided=time.strftime("%Y-%m-%dT%H:%M:%S"))
        applied.append(dict(info, id=s["id"], source=s["source"]))
    return words, applied
