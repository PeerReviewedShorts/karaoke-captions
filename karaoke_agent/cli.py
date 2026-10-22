"""karaoke-agent: the same captions, double-checked — built to be driven by an AI coding agent (Claude Code, Codex).

AI agents: run `karaoke-agent guide` first and follow it.

    karaoke-agent guide                           the instructions for AI agents (rules, steps, how to decide)
    karaoke-agent init                            put them into this folder (AGENTS.md, CLAUDE.md, a Claude Code skill)

    karaoke-agent sysinfo                         facts about this machine (OS, chip, RAM, GPU, installed Whisper backends)
    karaoke-agent setup --backend whisper.cpp --model large-v3-turbo --reason "..."    record the chosen Whisper in karaoke.toml
    karaoke-agent run clip.mp4 --glossary g.txt   Parakeet + Whisper second opinion + glossary check -> open suggestions
    karaoke-agent list clip.mp4                   open suggestions
    karaoke-agent suggest clip.mp4 "old|new" --why "..."     file your own suggestion
    karaoke-agent decide clip.mp4 --accept 1,2 --reject 3 [--edit 1="Apptronik's"]   apply / reject
    karaoke run clip.mp4                          then render + burn (tool 1, reuses the corrected words)

Single steps: second-opinion (Whisper only), proofread (glossary; --llm adds an API proofreader for runs without an agent).
Exit codes: 0 ok, 1 error, 2 timing check failed, 3 open suggestions wait for a decision. Every command takes --json.
"""
import argparse
import json
import os
import platform
import sys
import time

from karaoke import __version__, media
from karaoke import check as ck
from karaoke import words as wd
from karaoke.cli import clip_paths, do_transcribe, find_clip, log

from . import compare, config, sysinfo, whisper
from . import suggestions as sg

GUIDE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guide.md")
SKILL = """---
name: karaoke-captions
description: Burn word-by-word karaoke captions into videos (Shorts, Reels, TikToks) with the karaoke and karaoke-agent
  command line tools, including a Whisper second opinion and glossary check. Use when the user wants captions or
  subtitles on a clip.
---

Run `karaoke-agent guide` and follow it exactly. It is the instruction set for the installed version.
"""
COMMANDS = ("guide", "init", "sysinfo", "setup", "run", "second-opinion", "proofread", "list", "suggest", "decide")


def _clip(a):
    clip, p = clip_paths(a.clip, getattr(a, "out_dir", None))
    return clip or find_clip(p), p


def _glossary(path):
    return compare.load_glossary(path) if path else []


def _whisper_cfg(a):
    cfg = config.load(a.config).get("whisper", {})
    backend = getattr(a, "backend", None) or cfg.get("backend")
    model = getattr(a, "model", None) or (cfg.get("model_file") if backend == "whisper.cpp" else None) or cfg.get("model")
    return backend, model, cfg


# ------------------------------------------------------------------ steps
def do_second_opinion(clip, p, a, glossary):
    backend, model, cfg = _whisper_cfg(a)
    if not backend or not model:
        log("whisper   not set up (karaoke-agent setup --backend … --model …); second opinion skipped")
        return None
    prompt = ", ".join(g["term"] for g in glossary) or None
    log(f"whisper   {backend} · {cfg.get('model') or model} ...")
    opts = {k: cfg[k] for k in ("compute_type", "device") if backend == "faster-whisper" and k in cfg}
    res = whisper.transcribe(clip, backend, model, cfg.get("language"), prompt, **opts)
    with open(p["whisper"], "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    words = wd.load(p["words"])["words"]
    dis = compare.disagreements(words, res["text"])
    with open(p["disagree"], "w", encoding="utf-8") as fh:
        json.dump({"backend": backend, "model": model, "disagreements": dis}, fh, ensure_ascii=False, indent=1)
    real = [d for d in dis if d["kind"] == "words"]
    log(f"whisper   {res['seconds']} s · {len(real)} words disagree with Parakeet"
        + (f" (+{len(dis) - len(real)} spacing/number only)" if len(dis) > len(real) else ""))
    doc = sg.load(p)
    added = sg.add(doc, [{"i0": d["i0"], "i1": d["i1"], "old": d["parakeet"], "new": d["whisper"],
                          "why": f"Whisper ({cfg.get('model') or model}) heard {d['whisper']!r}: {d['context']}"}
                         for d in real if d["parakeet"]],
                   "whisper")
    sg.save(p, doc)
    return {"disagreements": len(real), "suggestions_added": len(added), "whisper_seconds": res["seconds"]}


def do_proofread(clip, p, a, glossary):
    words = wd.load(p["words"])["words"]
    doc = sg.load(p)
    hits = compare.glossary_hits(words, glossary)
    added = sg.add(doc, [{"i0": h["i0"], "i1": h["i1"], "old": h["old"], "new": h["term"],
                          "why": f"glossary: {h['term']}" + (f" ({h['note']})" if h["note"] else "")
                          + f", similarity {h['score']}"} for h in hits], "glossary")
    log(f"glossary  {len(glossary)} terms · {len(hits)} near-misses in the transcript")
    llm = 0
    if getattr(a, "llm", False):
        from . import proofread
        dis = []
        if os.path.exists(p["disagree"]):
            with open(p["disagree"], encoding="utf-8") as fh:
                dis = json.load(fh)["disagreements"]
        items = proofread.run(words, dis, glossary, getattr(a, "topic", None), model=a.llm_model)
        llm = len(sg.add(doc, items, "llm"))
        log(f"llm       {a.llm_model}: {len(items)} suggestions ({llm} new)")
    sg.save(p, doc)
    return {"glossary_hits": len(hits), "suggestions_added": len(added) + llm}


def show(doc):
    for s in sg.open_items(doc):
        log(f"  #{s['id']:<3} {s['source']:8} [{s['i0']}:{s['i1']}] {s['old']!r} -> {s['new']!r}   {s['why'][:90]}")


# ------------------------------------------------------------------ commands
def cmd_guide(a):
    with open(GUIDE, encoding="utf-8") as fh:
        text = fh.read()
    if not a.json:
        print(text)
    return {"guide": text}, 0


def cmd_init(a):
    """AGENTS.md (read by Codex and others), CLAUDE.md importing it and .claude/skills/karaoke-captions/SKILL.md
    (Claude Code) in this folder.
    Existing files are left alone unless --force."""
    folder = os.path.abspath(a.folder or os.getcwd())
    with open(GUIDE, encoding="utf-8") as fh:
        guide = fh.read()
    targets = {os.path.join(folder, "AGENTS.md"): guide,
               os.path.join(folder, "CLAUDE.md"): "@AGENTS.md\n",          # Claude Code reads CLAUDE.md, which imports it
               os.path.join(folder, ".claude", "skills", "karaoke-captions", "SKILL.md"): SKILL}
    written, kept = [], []
    for path, text in targets.items():
        if os.path.exists(path) and not a.force:
            kept.append(path)
            log(f"init      {path} exists, left alone (--force to replace; or add: see `karaoke-agent guide`)")
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        written.append(path)
        log(f"init      wrote {path}")
    return {"written": written, "kept": kept}, 0


def cmd_sysinfo(a):
    info = sysinfo.collect()
    if not a.json:
        log(f"{info['os']} {info['os_version']} · {info['arch']} · {info['cpu']} · {info['cores']} cores · "
            f"{info['ram_gb']} GB RAM · disk free {info['disk_free_gb']} GB")
        for g in info["gpus"] or [{"vendor": "-", "name": "no GPU found"}]:
            log(f"gpu       {g['vendor']} {g['name']}" + (f" · {g['vram_gb']} GB" if g.get("vram_gb") else ""))
        if info["torch"]:
            log(f"torch     {info['torch']}")
        for k, v in info["whisper_installed"].items():
            log(f"whisper   {k:15} {'installed' if v['installed'] else '-'}")
        for m in info["whisper_models"]:
            log(f"model     {m['backend']:15} {m['model']:22} {m['size_gb']} GB  {m['path']}")
        cur = config.load(a.config).get("whisper")
        log(f"chosen    {cur['backend']} · {cur['model']} ({cur.get('reason', '')})" if cur else "chosen    nothing yet")
    return info, 0


def cmd_setup(a):
    ok, msg = whisper.available(a.backend, a.model)
    if not ok:
        hint = sysinfo.install_hints().get(a.backend, {})
        log(f"setup     {a.backend} not ready: {msg}")
        if hint:
            log(f"install   {hint.get('install')}")
            log(f"models    {hint.get('models')}")
        return {"ok": False, "reason": msg, "hint": hint}, 1
    cfg = config.load(a.config)
    model_file = whisper.ggml_path(a.model) if a.backend == "whisper.cpp" else None     # remember where the file was found
    cfg["whisper"] = {"backend": a.backend, "model": a.model, "model_file": model_file, "language": a.language,
                      "compute_type": a.compute_type,
                      "device": a.device, "reason": a.reason or "", "decided": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "machine": f"{platform.system()} {platform.machine()}"}
    path = config.save(cfg, a.config)
    log(f"setup     {a.backend} · {a.model} -> {path}")
    return {"ok": True, "config": path, "whisper": cfg["whisper"]}, 0


def cmd_second_opinion(a):
    clip, p = _clip(a)
    if not os.path.exists(p["words"]):
        do_transcribe(clip, p)
    res = do_second_opinion(clip, p, a, _glossary(a.glossary))
    return (res or {"skipped": "whisper not set up"}), (0 if res else 1)


def cmd_proofread(a):
    clip, p = _clip(a)
    if not os.path.exists(p["words"]):
        do_transcribe(clip, p)
    res = do_proofread(clip, p, a, _glossary(a.glossary))
    doc = sg.load(p)
    show(doc)
    return dict(res, open=len(sg.open_items(doc))), (3 if sg.open_items(doc) else 0)


def cmd_run(a):
    clip, p = _clip(a)
    glossary = _glossary(a.glossary)
    doc, _ = do_transcribe(clip, p, a.device, a.retranscribe)
    out = {"words": p["words"], "review": p["review"]}
    try:
        out["second_opinion"] = do_second_opinion(clip, p, a, glossary)
    except whisper.BackendError as e:
        log(f"whisper   failed: {e}")
        out["second_opinion"] = {"error": str(e)}
    out["proofread"] = do_proofread(clip, p, a, glossary)
    res = ck.run(wd.load(p["words"])["words"], clip)
    with open(p["check"], "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    out["check"] = {"ok": res["ok"], "failed": res["failed"], "warnings": res["warnings"]}
    log(f"check     {'passed' if res['ok'] else 'FAILED'}" + "".join(f" · {x}" for x in res["failed"] + res["warnings"]))
    sdoc = sg.load(p)
    out["open"] = len(sg.open_items(sdoc))
    show(sdoc)
    if out["open"]:
        log(f"next      decide: karaoke-agent decide {a.clip} --accept … --reject …   then: karaoke run {a.clip}")
        return out, 3
    if not res["ok"]:
        return out, 2
    log(f"next      karaoke run {a.clip}")
    return out, 0


def cmd_list(a):
    _, p = clip_paths(a.clip, a.out_dir)
    doc = sg.load(p)
    show(doc)
    return {"open": sg.open_items(doc), "all": len(doc["suggestions"])}, 0


def cmd_suggest(a):
    _, p = clip_paths(a.clip, a.out_dir)
    words = wd.load(p["words"])["words"]
    old, new = a.change.split("|", 1)
    i0 = a.at if a.at is not None else wd.find(words, old)
    if i0 < 0:
        raise ValueError(f"{old!r} not found in the transcript")
    i1 = i0 + len([x for x in old.split() if wd.norm(x)])
    doc = sg.load(p)
    added = sg.add(doc, [{"i0": i0, "i1": i1, "old": " ".join(w["w"] for w in words[i0:i1]), "new": new,
                          "why": a.why or ""}], "agent")
    sg.save(p, doc)
    show(doc)
    return {"added": added}, 0


def _ids(s):
    return [int(x) for x in s.split(",") if x.strip()] if s else []


def cmd_decide(a):
    _, p = clip_paths(a.clip, a.out_dir)
    wdoc = wd.load(p["words"])
    doc = sg.load(p)
    accept, reject = _ids(a.accept), _ids(a.reject)
    if a.accept == "all":
        accept = [s["id"] for s in sg.open_items(doc)]
    edits = {}
    for e in a.edit or []:
        k, _, v = e.partition("=")
        edits[int(k)] = v
    if a.interactive:
        for s in sg.open_items(doc):
            ans = input(f"#{s['id']} {s['old']!r} -> {s['new']!r} ({s['why'][:80]})  [y]es/[n]o/[e]dit/[s]kip: ").strip().lower()
            if ans.startswith("y"):
                accept.append(s["id"])
            elif ans.startswith("n"):
                reject.append(s["id"])
            elif ans.startswith("e"):
                edits[s["id"]] = input("    new text: ").strip()
                accept.append(s["id"])
    words, applied = sg.decide(wdoc["words"], doc, accept, reject, edits)
    for c in applied:
        log(f"applied   #{c['id']} {c['old']!r} -> {c['new']!r}  [{c['s']:.2f}-{c['e']:.2f}]")
    if applied:
        wdoc["words"] = words
        wd.log_fix(wdoc, applied, "suggestions")
        wd.save(p["words"], wdoc)
        wd.write_derived(words, p)
    sg.save(p, doc)
    left = sg.open_items(doc)
    log(f"decide    {len(applied)} applied · {len(reject)} rejected · {len(left)} still open")
    return {"applied": applied, "open": left}, (3 if left else 0)


def parser():
    ap = argparse.ArgumentParser(prog="karaoke-agent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"karaoke-agent {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, clip=True):
        s = sub.add_parser(name)
        if clip:
            s.add_argument("clip")
            s.add_argument("--out-dir")
        s.add_argument("--json", action="store_true")
        s.add_argument("--config", help=f"path to karaoke.toml (default ./{config.NAME})")
        s.set_defaults(fn=fn)
        return s

    def whisper_opts(s):
        s.add_argument("--glossary", help="names, brands, topic words; one per line, '# comments', 'Term (note)'")
        s.add_argument("--backend", choices=whisper.BACKENDS, help="override karaoke.toml")
        s.add_argument("--model", help="override karaoke.toml")

    add("guide", cmd_guide, clip=False)
    s = add("init", cmd_init, clip=False)
    s.add_argument("folder", nargs="?", help="default: the current folder")
    s.add_argument("--force", action="store_true", help="replace existing AGENTS.md / SKILL.md")
    add("sysinfo", cmd_sysinfo, clip=False)
    s = add("setup", cmd_setup, clip=False)
    s.add_argument("--backend", required=True, choices=whisper.BACKENDS)
    s.add_argument("--model", required=True, help="large-v3-turbo, medium, small …; for whisper.cpp also a ggml .bin path")
    s.add_argument("--language", default="auto", help="ISO code (en, de …) or auto")
    s.add_argument("--compute-type", help="faster-whisper: float16, int8_float16, int8 …")
    s.add_argument("--device", help="faster-whisper: cuda, cpu, auto")
    s.add_argument("--reason", help="why this choice fits this machine (stored for the next run)")
    s = add("run", cmd_run)
    whisper_opts(s)
    s.add_argument("--llm", action="store_true", help="also ask the API proofreader (needs `pip install anthropic`)")
    s.add_argument("--llm-model", default="claude-opus-5-5")
    s.add_argument("--topic", help="one line about the video, for the LLM proofreader")
    s.add_argument("--device", default="auto", help="torch device for Parakeet: auto, cpu, mps, cuda")
    s.add_argument("--retranscribe", action="store_true")
    s = add("second-opinion", cmd_second_opinion)
    whisper_opts(s)
    s = add("proofread", cmd_proofread)
    s.add_argument("--glossary")
    s.add_argument("--llm", action="store_true")
    s.add_argument("--llm-model", default="claude-opus-5-5")
    s.add_argument("--topic")
    add("list", cmd_list)
    s = add("suggest", cmd_suggest)
    s.add_argument("change", help='"old words|new words"')
    s.add_argument("--why", help="reason, shown to whoever decides")
    s.add_argument("--at", type=int, help="word index of the first old word (when the words occur more than once)")
    s = add("decide", cmd_decide)
    s.add_argument("--accept", help="ids, comma separated, or 'all'")
    s.add_argument("--reject", help="ids, comma separated")
    s.add_argument("--edit", action="append", help='ID=new text, e.g. 1="Apptronik\'s" (accepts it with that text)')
    s.add_argument("--interactive", action="store_true", help="ask y/n/edit for every open suggestion")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        result, code = a.fn(a)
    except (FileNotFoundError, ValueError, KeyError, media.MediaError, RuntimeError, whisper.BackendError) as e:
        log(f"error     {e}")
        if getattr(a, "json", False):
            print(json.dumps({"error": str(e)}))
        return 1
    if a.json:
        print(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main())
