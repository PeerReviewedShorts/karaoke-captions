"""karaoke: word-by-word karaoke captions with one speech model, offline, no API keys.

    karaoke clip.mp4                       transcribe + check + render + burn  ->  clip_captioned.mp4
    karaoke transcribe clip.mp4            clip.words.json, clip.review.txt, clip.srt
    karaoke fix clip.mp4 "Aptronics|Apptronik's"     replace words (any count), timing of the span is kept
    karaoke fix clip.mp4 --from-review     carry your edits in clip.review.txt onto the timed words
    karaoke check clip.mp4                 timing checks, exit code 2 if one fails
    karaoke render clip.mp4 --preset bold  transparent overlay clip.captions.mov
    karaoke burn clip.mp4                  overlay onto the clip -> clip_captioned.mp4
    karaoke preview clip.mp4 --at 3.2      one frame with captions as PNG (try fonts and colors fast)
    karaoke presets [--show amber]         list the built-in styles (or print one as JSON to copy)

Style options for run/render/preview: --preset, --font, --size, --color, --accent, --outline, --outline-width, --plate,
--uppercase/--no-uppercase, --y, --max-words.

Every command takes --json (machine-readable result on stdout, logs on stderr) and --out-dir.
Exit codes: 0 ok, 1 error, 2 a timing check failed.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

from . import __version__, media
from . import check as ck
from . import render as rd
from . import words as wd

MEDIA_EXT = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".wav", ".mp3", ".m4a", ".flac", ".ogg")
COMMANDS = ("run", "transcribe", "fix", "check", "render", "burn", "preview", "presets")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def clip_paths(arg, out_dir=None):
    if arg.endswith(".words.json"):                     # a words file stands for its clip
        base = arg[:-len(".words.json")]
        return None, wd.paths(base + ".clip", out_dir or os.path.dirname(os.path.abspath(arg)))
    if not os.path.isfile(arg):
        raise FileNotFoundError(f"file not found: {arg}")
    return arg, wd.paths(arg, out_dir)


def find_clip(p):
    """The clip next to a words file (any extension), for commands that need the audio/video."""
    for f in sorted(os.listdir(p["dir"])):
        stem, ext = os.path.splitext(f)
        if not f.startswith("._") and stem == p["base"] and ext.lower() in MEDIA_EXT:
            return os.path.join(p["dir"], f)
    raise FileNotFoundError(f"no clip named {p['base']}.* next to the words file")


# ------------------------------------------------------------------ steps
def do_transcribe(clip, p, device="auto", force=False, snap=True):
    if os.path.exists(p["words"]) and not force:
        log(f"words     {p['words']} exists, reused (transcribe again: `karaoke run --retranscribe` "
            f"or `karaoke transcribe --force`)")
        return wd.load(p["words"]), False
    from . import parakeet  # torch/transformers only load when needed
    log(f"parakeet  {parakeet.MODEL} on {parakeet.pick_device(device)} ...")
    words, prov = parakeet.transcribe(clip, device=device, log=log)
    if snap:
        from . import timing
        words, moved = timing.snap_starts(words, media.read_audio(clip))
        prov["snapped_starts"] = moved
        log(f"timing    {moved} word starts moved onto the voice (they began in silence)")
    doc = {"words": words, "provenance": prov}
    os.makedirs(p["dir"], exist_ok=True)
    wd.save(p["words"], doc)
    cues = wd.write_derived(words, p)
    log(f"parakeet  {len(words)} words, text + timing, {prov['seconds']} s")
    log(f"wrote     {os.path.basename(p['words'])}, {os.path.basename(p['review'])}, "
        f"{os.path.basename(p['srt'])} ({cues} cues)")
    return doc, True


def do_check(clip, p, words):
    res = ck.run(words, clip)
    with open(p["check"], "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    m = res["metrics"]
    log(f"check     {'passed' if res['ok'] else 'FAILED'} · {m['n_words']} words · longest {m['longest_s']} s"
        + (f" · early starts {m['early_start']:.0%}" if "early_start" in m else "")
        + (f" · start error after pauses {m['onset_p50'] * 1000:.0f} ms" if "onset_p50" in m else ""))
    for x in res["failed"]:
        log(f"  fail    {x}")
    for x in res["warnings"]:
        log(f"  warn    {x}")
    return res


def _video(clip):
    info = media.probe(clip)
    if not info["has_video"]:
        raise media.MediaError(f"{clip} has no video stream to put captions on (audio only? `karaoke transcribe` still "
                               f"writes the words and an .srt)")
    return info


def _style(a):
    """CLI style options -> preset overrides."""
    o = {"font": getattr(a, "font", None), "size": getattr(a, "size", None), "text": getattr(a, "color", None),
         "accent": getattr(a, "accent", None), "stroke": getattr(a, "outline", None),
         "stroke_w": getattr(a, "outline_width", None), "uppercase": getattr(a, "uppercase", None)}
    plate = getattr(a, "plate", None)
    if plate is not None:
        o["plate"] = False if plate.lower() in ("none", "off", "no") else plate
    return o


def _warn_style(pr, missing, fallback=None):
    for w in pr.get("_warnings", []):
        log(f"  warn    {w}")
    if missing:
        name = os.path.basename(rd.resolve_font(pr["font"], pr.get("_dir")))
        log(f"  warn    font {name} has no glyph for {''.join(missing)!r}: "
            + (f"using {fallback} instead" if fallback else "they will show as boxes")
            + "; choose a font that covers your language with --font")


def do_render(clip, p, words, preset="amber", y=None, max_words=None, overrides=None, full_frame=False):
    info = _video(clip)
    pr = rd.load_preset(preset, overrides)
    t0 = time.time()
    dur = info["duration"] or (max((w["e"] for w in words), default=0) + 0.5)
    r = rd.render(words, p["mov"], info["width"], info["height"], info["fps"], dur, pr, y, max_words, full_frame)
    log(f"render    {os.path.basename(p['mov'])} · {info['width']}x{info['height']} · {r['frames']} frames · "
        f"preset {pr['name']} · {time.time() - t0:.1f} s")
    _warn_style(pr, r["missing_glyphs"], r["fallback_font"])
    return dict(r, mov=p["mov"], preset=pr["name"], width=info["width"], height=info["height"])


def overlay_info(mov):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height:format_tags=comment",
                          "-of", "json", mov], capture_output=True, text=True)
    d = json.loads(out.stdout or "{}")
    st = (d.get("streams") or [{}])[0]
    m = re.search(r"karaoke_y=(\d+)", (d.get("format", {}).get("tags") or {}).get("comment", ""))
    return st.get("width"), st.get("height"), int(m.group(1)) if m else 0


def burn_cmd(clip, mov, out, info, ov):
    """The ffmpeg command that puts the overlay on the clip. Rotation is applied by ffmpeg's autorotate, non-square
    pixels are scaled to square ones, odd sizes padded to even (H.264 needs that); HDR stays HDR (HEVC 10-bit) when
    libx265 is there."""
    ow, oh, oy = ov
    W, H = info["width"], info["height"]
    if ow != W or oy + oh > H:
        raise ValueError(f"the overlay was rendered for another frame size ({ow} px wide, video is {W}x{H}): "
                         f"run `karaoke render` again")
    pre = f"scale={W}:{H}," if info["sar"] != 1.0 else ""
    hdr = info["hdr"] and media.has_encoder("libx265")
    fmt = "yuv420p10le" if hdr else "yuv420p"
    fc = (f"[0:v]{pre}setsar=1[b];[b][1:v]overlay=0:{oy}:eof_action=pass:format=auto,"
          f"pad={W + W % 2}:{H + H % 2}:0:0,format={fmt}[v]")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", clip, "-i", mov, "-filter_complex", fc, "-map", "[v]"]
    if hdr:
        cmd += ["-c:v", "libx265", "-preset", "fast", "-crf", "20", "-tag:v", "hvc1", "-x265-params", "log-level=error"]
        cmd += ["-color_primaries", "bt2020", "-color_trc", info["transfer"], "-colorspace", "bt2020nc"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
    if info["has_audio"]:
        copy = info["audio_codec"] in media.MP4_AUDIO
        cmd += ["-map", "0:a:0"] + (["-c:a", "copy"] if copy else ["-c:a", "aac", "-b:a", "192k"])
    return cmd + ["-movflags", "+faststart", out]


def do_burn(clip, p, out=None):
    out = out or p["out"]
    if not os.path.exists(p["mov"]):
        raise FileNotFoundError(f"{p['mov']} missing: run `karaoke render` first")
    if os.path.abspath(out) == os.path.abspath(clip):
        raise ValueError("refusing to overwrite the original clip")
    info = _video(clip)
    if info["hdr"] and not media.has_encoder("libx265"):
        log(f"  warn    HDR video ({info['transfer']}) and no libx265 in this ffmpeg: the result is SDR H.264 "
            f"without tone mapping and may look washed out")
    t0 = time.time()
    subprocess.run(burn_cmd(clip, p["mov"], out, info, overlay_info(p["mov"])), check=True)
    log(f"burn      {out} · {time.time() - t0:.1f} s" + (" · HDR kept (HEVC 10-bit)" if info["hdr"] and
                                                         media.has_encoder("libx265") else ""))
    return out


def do_preview(clip, p, words, t, preset="amber", y=None, max_words=None, overrides=None, out=None):
    """One frame of the clip at t with the captions on top, as PNG."""
    from PIL import Image
    info = _video(clip)
    pr = rd.load_preset(preset, overrides)
    if t is None:                                        # the middle of the first group with a highlighted word
        ch = wd.chunk(words, max_words or pr["max_words"])
        t = (ch[0][min(1, len(ch[0]) - 1)]["s"] + 0.02) if ch else 0.0
    W, H = info["width"], info["height"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", clip, "-frames:v", "1",
                          "-vf", f"scale={W}:{H},setsar=1", "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
                         capture_output=True).stdout
    frame = Image.frombytes("RGBA", (W, H), raw) if len(raw) == W * H * 4 else Image.new("RGBA", (W, H), (40, 40, 40, 255))
    cap, missing, fallback = rd.still(words, t, W, H, pr, y, max_words)
    frame.alpha_composite(cap)
    out = out or os.path.join(p["dir"], p["base"] + ".preview.png")
    frame.convert("RGB").save(out)
    log(f"preview   {out} · t={t:.2f} s · preset {pr['name']}")
    _warn_style(pr, missing, fallback)
    return {"png": out, "t": round(t, 3), "missing_glyphs": missing, "fallback_font": fallback}


# ------------------------------------------------------------------ commands
def cmd_run(a):
    clip, p = clip_paths(a.clip, a.out_dir)
    doc, _ = do_transcribe(clip, p, a.device, a.retranscribe, not a.no_snap)
    res = do_check(clip, p, doc["words"])
    out = {"clip": clip, "words": p["words"], "review": p["review"], "srt": p["srt"], "check": res}
    if not res["ok"] and not a.force:
        log("stopped   a timing check failed; fix the words (karaoke fix) or pass --force")
        return out, 2
    out["render"] = do_render(clip, p, doc["words"], a.preset, a.y, a.max_words, _style(a), a.full_frame)
    if not a.no_burn:
        out["output"] = do_burn(clip, p, a.output)
    return out, 0


def cmd_transcribe(a):
    clip, p = clip_paths(a.clip, a.out_dir)
    doc, new = do_transcribe(clip, p, a.device, a.force, not a.no_snap)
    return {"words": p["words"], "review": p["review"], "srt": p["srt"], "n_words": len(doc["words"]), "new": new}, 0


def cmd_fix(a):
    _, p = clip_paths(a.clip, a.out_dir)
    doc = wd.load(p["words"])
    words, changes = doc["words"], []
    if a.from_review:
        with open(p["review"], encoding="utf-8") as fh:
            words, changes = wd.apply_text(words, fh.read())
    for spec in a.replace or []:
        if "|" not in spec:
            raise ValueError(f'expected "old words|new words", got {spec!r}')
        old, new = spec.split("|", 1)
        try:
            words, info = wd.replace(words, old, new)
        except KeyError:
            log(f"not found {old!r}")
            continue
        changes.append(info)
    for c in changes:
        log(f"fix       {c['old']!r} -> {c['new']!r}  [{c['s']:.2f}-{c['e']:.2f}]")
    if not changes:
        log("fix       nothing changed")
        return {"changes": []}, (1 if a.replace else 0)
    if not a.dry_run:
        doc["words"] = words
        wd.log_fix(doc, changes, "review" if a.from_review else "replace")
        wd.save(p["words"], doc)
        wd.write_derived(words, p)
        log(f"wrote     {os.path.basename(p['words'])} (+ review, srt)")
    return {"changes": changes, "dry_run": a.dry_run}, 0


def cmd_check(a):
    clip, p = clip_paths(a.clip, a.out_dir)
    clip = clip or find_clip(p)
    res = do_check(clip, p, wd.load(p["words"])["words"])
    return res, (0 if res["ok"] else 2)


def cmd_render(a):
    clip, p = clip_paths(a.clip, a.out_dir)
    clip = clip or find_clip(p)
    return do_render(clip, p, wd.load(p["words"])["words"], a.preset, a.y, a.max_words, _style(a), a.full_frame), 0


def cmd_burn(a):
    clip, p = clip_paths(a.clip, a.out_dir)
    clip = clip or find_clip(p)
    return {"output": do_burn(clip, p, a.output)}, 0


def cmd_preview(a):
    clip, p = clip_paths(a.clip, a.out_dir)
    clip = clip or find_clip(p)
    return do_preview(clip, p, wd.load(p["words"])["words"], a.at, a.preset, a.y, a.max_words, _style(a), a.output), 0


def cmd_presets(a):
    if a.show:
        pr = rd.load_preset(a.show)
        out = {k: v for k, v in pr.items() if not k.startswith("_")}
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return out, 0
    out = {n: rd.load_preset(n).get("note", "") for n in rd.presets()}
    for n, note in out.items():
        log(f"{n:8} {note}")
    return out, 0


def parser():
    ap = argparse.ArgumentParser(prog="karaoke", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"karaoke {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, clip=True):
        s = sub.add_parser(name)
        if clip:
            s.add_argument("clip", help="video/audio file (or its .words.json)")
            s.add_argument("--out-dir", help="where the files go (default: next to the clip)")
        s.add_argument("--json", action="store_true", help="print the result as JSON on stdout")
        s.set_defaults(fn=fn)
        return s

    def style(s):
        s.add_argument("--preset", default="amber", help="style: built-in name or a .json path (default amber)")
        s.add_argument("--font", help="font file (.ttf/.otf/.ttc) or the name of a bundled or installed font")
        s.add_argument("--size", type=float, help="font size in px for a 1080 px short side (scales with the video)")
        s.add_argument("--color", help="text color: #RRGGBB, a name (white) or r,g,b")
        s.add_argument("--accent", help="color of the word being spoken")
        s.add_argument("--outline", help="outline color")
        s.add_argument("--outline-width", type=float, help="outline width in px (0 = none)")
        s.add_argument("--plate", help="box behind the text: a color with alpha (#000000B0) or none")
        s.add_argument("--uppercase", action=argparse.BooleanOptionalAction, default=None, help="all capitals")
        s.add_argument("--y", type=float, help="bottom edge of the captions: fraction of the height (0.72) or pixels")
        s.add_argument("--max-words", type=int, help="words per caption group (preset default 3)")
        s.add_argument("--full-frame", action="store_true", help="frame-sized overlay .mov (for video editors)")

    s = add("run", cmd_run)
    style(s)
    s.add_argument("-o", "--output", help="output video (default <clip>_captioned.mp4)")
    s.add_argument("--device", default="auto", help="torch device for Parakeet: auto (default: cuda > mps > cpu), cpu, mps, cuda")
    s.add_argument("--retranscribe", action="store_true", help="transcribe again even if <clip>.words.json exists")
    s.add_argument("--force", action="store_true", help="render even if a timing check fails")
    s.add_argument("--no-burn", action="store_true", help="stop after the transparent overlay")
    s.add_argument("--no-snap", action="store_true", help="keep Parakeet's raw word starts (no snap onto the voice)")
    s = add("transcribe", cmd_transcribe)
    s.add_argument("--device", default="auto")
    s.add_argument("--no-snap", action="store_true")
    s.add_argument("--force", action="store_true", help="overwrite an existing words file")
    s = add("fix", cmd_fix)
    s.add_argument("replace", nargs="*", help='"old words|new words" (matched without case and punctuation)')
    s.add_argument("--from-review", action="store_true", help="apply your edits in <clip>.review.txt")
    s.add_argument("--dry-run", action="store_true")
    add("check", cmd_check)
    s = add("render", cmd_render)
    style(s)
    s = add("burn", cmd_burn)
    s.add_argument("-o", "--output")
    s = add("preview", cmd_preview)
    style(s)
    s.add_argument("--at", type=float, help="time in seconds (default: the first caption)")
    s.add_argument("-o", "--output", help="PNG path (default <clip>.preview.png)")
    s = add("presets", cmd_presets, clip=False)
    s.add_argument("--show", help="print one preset as JSON (a starting point for your own)")
    return ap


def _utf8_console():
    for st in (sys.stdout, sys.stderr):                 # Windows consoles: never die on a character they cannot show
        reconfigure = getattr(st, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(errors="replace")
            except ValueError:
                pass


def main(argv=None):
    _utf8_console()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "run")                           # `karaoke clip.mp4` = `karaoke run clip.mp4`
    a = parser().parse_args(argv)
    try:
        result, code = a.fn(a)
    except (FileNotFoundError, ValueError, KeyError, media.MediaError, RuntimeError, subprocess.CalledProcessError) as e:
        log(f"error     {e}")
        if getattr(a, "json", False):
            print(json.dumps({"error": str(e)}))
        return 1
    if a.json:
        print(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main())
