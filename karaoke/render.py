"""Transparent karaoke overlay: Pillow draws every distinct caption state once, ffmpeg stores it as QuickTime Animation
(qtrle, lossless, with alpha, tiny for mostly transparent frames). Any frame size: 9:16, 16:9, 1:1, 4K.

By default the overlay is a full-width BAND only as tall as the captions; its vertical position is stored in the .mov
(metadata `karaoke_y`) and `karaoke burn` places it there. `--full-frame` writes a frame-sized overlay instead, for
dropping into a video editor.

A preset (karaoke/presets/<name>.json or a path) sets font, colors, outline, plate, position and the pop-in. Missing keys
fall back to DEFAULTS. Sizes are in pixels for a 1080 px short side and scale with the video.
"""
import functools
import glob
import json
import os
import platform
import re
import shutil
import subprocess

from PIL import Image, ImageColor, ImageDraw, ImageFont

from . import words as wd

HERE = os.path.dirname(os.path.abspath(__file__))
PRESETS = os.path.join(HERE, "presets")
FONTS = os.path.join(HERE, "fonts")
OVERLAY_FPS = 30.0                  # caption changes need no more; the burn step keeps the video's own frame rate

DEFAULTS = {
    "font": "Archivo-Variable.ttf", "font_index": 0, "axes": None, "style": None, "size": 74, "uppercase": False,
    "text": "#FFFFFF", "accent": "#FFC400", "stroke": "#000000", "stroke_w": 6, "plate": None, "plate_radius": 18,
    "line_gap": 16, "max_width": 0.89, "y": {"vertical": 0.72, "landscape": 0.9}, "max_words": 3, "pop": 0.0,
}
KEYS = set(DEFAULTS) | {"name", "note"}
AXIS_TAGS = {"wght": "weight", "wdth": "width", "slnt": "slant", "ital": "italic", "opsz": "optical size"}
FONT_EXT = (".ttf", ".otf", ".ttc", ".otc")
# Wide-coverage fonts tried, in order, when the chosen font lacks characters of the transcript (Cyrillic, Greek, …).
FALLBACK_FONTS = ("NotoSans-Black", "NotoSans-Bold", "DejaVuSans-Bold", "Arial Black", "Arial Bold", "arialbd",
                  "Segoe UI Bold", "segoeuib", "LiberationSans-Bold", "Arial Unicode", "DejaVuSans", "Arial")


class StyleError(ValueError):
    pass


# ------------------------------------------------------------------ presets
def presets():
    # skip ._ files (macOS writes them on exFAT drives)
    return sorted(f[:-5] for f in os.listdir(PRESETS) if f.endswith(".json") and not f.startswith("."))


def color(v, key="color"):
    """[r, g, b], [r, g, b, a], "r,g,b", "#RRGGBB", "#RRGGBBAA", "#RGB", a CSS name ("white") or "rgb(…)"
    -> RGBA tuple."""
    try:
        if isinstance(v, str) and re.fullmatch(r"\s*\d+\s*(,\s*\d+\s*){2,3}", v):
            c = tuple(int(x) for x in v.split(","))
        elif isinstance(v, str):
            c = ImageColor.getrgb(v.strip())
        elif isinstance(v, (list, tuple)) and len(v) in (3, 4) and all(isinstance(x, (int, float)) for x in v):
            c = tuple(int(x) for x in v)
        else:
            raise ValueError
    except ValueError:
        raise StyleError(f"{key}: {v!r} is not a color (use \"#FFC400\", [255, 196, 0] or a name like \"white\")") from None
    if any(not 0 <= x <= 255 for x in c):
        raise StyleError(f"{key}: {v!r} has a channel outside 0-255")
    return c + (255,) if len(c) == 3 else c


def validate(p):
    """Fill in defaults, check types. Returns (preset, warnings)."""
    warnings = [f"unknown preset key {k!r} (ignored)" for k in p if k not in KEYS]
    out = dict(DEFAULTS, **{k: v for k, v in p.items() if k in KEYS})
    for k in ("text", "accent", "stroke"):
        out[k] = color(out[k], k)
    out["plate"] = None if out["plate"] in (None, False, "none", "") else color(out["plate"], "plate")
    for k, lo in (("size", 4), ("stroke_w", 0), ("plate_radius", 0), ("line_gap", -200), ("max_words", 1), ("pop", 0)):
        if not isinstance(out[k], (int, float)) or isinstance(out[k], bool) or out[k] < lo:
            raise StyleError(f"{k}: expected a number >= {lo}, got {out[k]!r}")
    if not isinstance(out["max_width"], (int, float)) or not 0.2 <= out["max_width"] <= 1.0:
        raise StyleError(f"max_width: expected a fraction of the width between 0.2 and 1.0, got {out['max_width']!r}")
    y = out["y"]
    if isinstance(y, (int, float)) and not isinstance(y, bool):
        out["y"] = {"vertical": y, "landscape": y}
    elif not (isinstance(y, dict) and all(isinstance(y.get(k), (int, float)) for k in ("vertical", "landscape"))):
        raise StyleError(f"y: expected a number or {{\"vertical\": 0.72, \"landscape\": 0.9}}, got {y!r}")
    if out["axes"] is not None and not isinstance(out["axes"], (list, dict)):
        raise StyleError(f"axes: expected a list [900, 85] or a dict {{\"wght\": 900}}, got {out['axes']!r}")
    out["uppercase"] = bool(out["uppercase"])
    out["max_words"] = int(out["max_words"])
    return out, warnings


def load_preset(name, overrides=None):
    """A built-in name or a .json path, plus CLI overrides. Returns the validated preset; preset["_warnings"] lists
    anything worth telling the user."""
    path = name if os.path.isfile(name) else os.path.join(PRESETS, f"{name}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"preset {name!r} not found (built in: {', '.join(presets())}; or pass a .json path)")
    try:
        with open(path, encoding="utf-8") as fh:
            p = json.load(fh)
    except json.JSONDecodeError as e:
        raise StyleError(f"{path}: not valid JSON ({e})") from None
    if not isinstance(p, dict):
        raise StyleError(f"{path}: a preset is a JSON object")
    p.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    p.update({k: v for k, v in (overrides or {}).items() if v is not None})
    out, warnings = validate(p)
    out["_dir"] = os.path.dirname(os.path.abspath(path))
    out["_warnings"] = warnings
    return out


# ------------------------------------------------------------------ fonts
def _system_font_dirs():
    home = os.path.expanduser("~")
    s = platform.system()
    if s == "Darwin":
        return ["/System/Library/Fonts", "/System/Library/Fonts/Supplemental", "/Library/Fonts", f"{home}/Library/Fonts"]
    if s == "Windows":
        win = os.environ.get("WINDIR", r"C:\Windows")
        return [os.path.join(win, "Fonts"), os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts")]
    return ["/usr/share/fonts", "/usr/local/share/fonts", f"{home}/.local/share/fonts", f"{home}/.fonts"]


@functools.lru_cache(maxsize=1)
def _system_fonts():
    found = {}
    for d in _system_font_dirs():
        for f in glob.glob(os.path.join(d, "**", "*"), recursive=True):
            if f.lower().endswith(FONT_EXT) and not os.path.basename(f).startswith("._"):
                found.setdefault(_key(os.path.splitext(os.path.basename(f))[0]), f)
    return found


def _key(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_font(spec, preset_dir=None):
    """A font file path (absolute, relative to the working folder or to the preset) or the name of a bundled or installed
    font ("Anton", "Arial Black", "DejaVuSans-Bold")."""
    spec = os.path.expanduser(str(spec))
    for base in ([""] if os.path.isabs(spec) else [os.getcwd(), preset_dir or "", FONTS]):
        cand = os.path.join(base, spec)
        if os.path.isfile(cand):
            return cand
    k = _key(os.path.splitext(os.path.basename(spec))[0] if spec.lower().endswith(FONT_EXT) else spec)
    for f in os.listdir(FONTS):
        if f.lower().endswith(FONT_EXT) and _key(os.path.splitext(f)[0]).startswith(k):
            return os.path.join(FONTS, f)
    sysf = _system_fonts()
    if k in sysf:
        return sysf[k]
    if shutil.which("fc-match"):                             # Linux (and anything with fontconfig): ask it
        try:
            r = subprocess.run(["fc-match", "-f", "%{file}", spec], capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            r = ""
        if r and os.path.isfile(r) and k[:5] in _key(os.path.basename(r)):
            return r
    raise StyleError(f"font {spec!r} not found: give a .ttf/.otf path, a bundled font "
                     f"({', '.join(sorted(os.path.splitext(f)[0] for f in os.listdir(FONTS) if f.lower().endswith(FONT_EXT)))}) "
                     f"or the file name of an installed font")


def _axes_list(font, axes):
    """Preset axes -> the positional list Pillow wants. A list is taken as is; a dict maps tags/names to values
    ({"wght": 900, "wdth": 85}); values are clamped to the font's range."""
    info = font.get_variation_axes()
    vals = [a["default"] for a in info]
    if isinstance(axes, list):
        given = list(axes)[:len(info)]
        vals[:len(given)] = given
    else:
        names = [(a["name"].decode() if isinstance(a["name"], bytes) else str(a["name"])).lower() for a in info]
        for k, v in axes.items():
            n = AXIS_TAGS.get(k.lower(), k.lower())
            if n not in names:
                raise StyleError(f"axes: font has no {k!r} axis (it has: {', '.join(names)})")
            vals[names.index(n)] = v
    return [min(max(v, a["minimum"]), a["maximum"]) for v, a in zip(vals, info)]


@functools.lru_cache(maxsize=64)
def _font(path, size, index, axes_json, style):
    f = ImageFont.truetype(path, size, index=index)
    axes = json.loads(axes_json) if axes_json else None
    if axes or style:
        try:
            if style:
                f.set_variation_by_name(style)
            if axes:
                f.set_variation_by_axes(_axes_list(f, axes))
        except OSError:                                      # a static font: axes/style do not apply
            pass
        except StyleError:
            raise
        except ValueError as e:
            raise StyleError(f"style {style!r}: {e}") from None
    return f


def font_for(preset, size):
    path = resolve_font(preset["font"], preset.get("_dir"))
    return _font(path, max(4, int(size)), int(preset.get("font_index") or 0),
                 json.dumps(preset["axes"]) if preset.get("axes") else "", preset.get("style") or "")


def is_variable(preset):
    try:
        font_for(preset, 40).get_variation_axes()
        return True
    except OSError:
        return False


def missing_glyphs(font, text):
    """Characters the font has no glyph for (they would render as boxes or nothing)."""
    ref = font.getmask("\U0010fffd")
    ref_key = (ref.size, bytes(ref))
    miss = []
    for ch in sorted(set(text)):
        if ch.isspace() or ch in miss:
            continue
        m = font.getmask(ch)
        if (m.size, bytes(m)) == ref_key or (m.size[0] == 0 and ch.isprintable()):
            miss.append(ch)
    return miss


# ------------------------------------------------------------------ drawing
def position_y(preset, width, height, y=None):
    """Bottom edge of the caption block in px. y: None (preset), a fraction of the height (<= 1.5) or pixels."""
    if y is None:
        y = preset["y"]["vertical" if height >= width else "landscape"]
    return int(y * height) if y <= 1.5 else int(y)


class Painter:
    def __init__(self, preset, width, height, y=None):
        self.p, self.W, self.H = preset, width, height
        self.k = k = min(width, height) / 1080
        self.size = max(6, int(preset["size"] * k))
        self.font = font_for(preset, self.size)
        self.stroke = max(0, int(round(preset["stroke_w"] * k)))
        self.max_w = preset["max_width"] * width
        self.safe_w = width * 0.96                          # nothing may be wider than this, a long word shrinks instead
        self.line_gap = preset["line_gap"] * k
        self.bottom = min(height, max(1, position_y(preset, width, height, y)))
        self.text, self.accent = preset["text"], preset["accent"]
        self.outline, self.plate = preset["stroke"], preset["plate"]
        self.pad_x, self.pad_y = int(30 * k), int(8 * k)

    def label(self, w):
        return w.upper() if self.p["uppercase"] else w

    def _width(self, text, font):
        return ImageDraw.Draw(Image.new("RGBA", (8, 8))).textlength(text, font=font) + 2 * self.stroke

    def lines(self, chunk):
        out, line = [], []
        for w in chunk:
            trial = " ".join(self.label(x["w"]) for x in line + [w])
            if line and self._width(trial, self.font) > self.max_w:
                out.append(line)
                line = [w]
            else:
                line.append(w)
        if line:
            out.append(line)
        return out

    def fit(self, lines):
        """The font for this group: the preset size, or smaller when a single word is wider than the frame."""
        widest = max(self._width(" ".join(self.label(w["w"]) for w in ln), self.font) for ln in lines)
        extra = 2 * self.pad_x if self.plate else 0
        if widest + extra <= self.safe_w:
            return self.font
        size = max(6, int(self.size * (self.safe_w - extra) / (widest + extra)))
        return font_for(self.p, size)

    def block(self, lines, active):
        """The caption block as its own RGBA image (full width, block height) with the active word highlighted."""
        font = self.fit(lines)
        lh = int(font.size + self.line_gap)
        pad_x, pad_y = self.pad_x, self.pad_y
        h = lh * len(lines) + 2 * pad_y + 2 * self.stroke + int(20 * self.k)
        im = Image.new("RGBA", (self.W, max(1, h)), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        y, wi = pad_y + self.stroke, 0
        for ln in lines:
            toks = [self.label(w["w"]) for w in ln]
            bl, bt, br, bb = d.textbbox((0, 0), " ".join(toks), font=font, stroke_width=self.stroke)
            x = (self.W - (br - bl)) / 2 - bl
            if self.plate:
                d.rounded_rectangle([x + bl - pad_x, y + bt - pad_y, x + br + pad_x, y + bb + pad_y],
                                    radius=int(self.p["plate_radius"] * self.k), fill=self.plate)
            for tok in toks:
                d.text((x, y), tok, font=font, fill=self.accent if wi == active else self.text,
                       stroke_width=self.stroke, stroke_fill=self.outline)
                x += d.textlength(tok + " ", font=font)
                wi += 1
            y += lh
        return im


def cover(preset, text):
    """The preset, or a copy with a fallback font when its font lacks characters of `text`.
    Returns (preset, missing characters, name of the fallback or None)."""
    miss = missing_glyphs(font_for(preset, 60), text)
    if not miss:
        return preset, [], None
    for name in FALLBACK_FONTS:
        try:
            alt = dict(preset, font=resolve_font(name), axes=None, style=None, font_index=0)
            if not missing_glyphs(font_for(alt, 60), text):
                return alt, miss, os.path.basename(alt["font"])
        except (StyleError, OSError):
            continue
    return preset, miss, None


class Timeline:
    """Which caption group and word is lit at time t, and the image for that state."""

    def __init__(self, words, preset, width, height, y=None, max_words=None):
        self.chunks = wd.chunk(words, max_words or preset["max_words"])
        label = (lambda w: w.upper()) if preset["uppercase"] else (lambda w: w)
        preset, self.missing, self.fallback = cover(preset, " ".join(label(w["w"]) for w in words))
        self.pt = Painter(preset, width, height, y)
        self.laid = [self.pt.lines(c) for c in self.chunks]
        self.starts = [c[0]["s"] for c in self.chunks]
        self.pop = float(preset["pop"])                   # seconds of scale-in when a group appears (0 = off)
        self._images = {}

    def state(self, t, ci=0):
        """(key, ci) for time t; key None = nothing on screen. ci is a search hint (frames come in order)."""
        if not self.chunks:
            return None, 0
        if ci >= len(self.chunks) or (ci and t < self.starts[ci] - 0.08):
            ci = 0
        while ci + 1 < len(self.chunks) and t >= self.starts[ci + 1] - 0.08:
            ci += 1
        ch = self.chunks[ci]
        if not ch[0]["s"] - 0.08 <= t <= ch[-1]["e"] + 0.35:
            return None, ci
        ai = max([k for k, w in enumerate(ch) if t >= w["s"]] or [0])
        step = min(4, int(max(0.0, t - ch[0]["s"] + 0.08) / self.pop * 4)) if self.pop else 4
        return (ci, ai, step), ci

    def image(self, key):
        if key in self._images:
            return self._images[key]
        if len(self._images) > 256:
            self._images.clear()
        ci, ai, step = key
        blk = self.pt.block(self.laid[ci], ai)
        if step < 4:                                     # pop: 88 % -> 100 % in four steps
            s = 0.88 + 0.03 * step
            small = blk.resize((max(1, int(blk.width * s)), max(1, int(blk.height * s))), Image.Resampling.BICUBIC)
            blk = Image.new("RGBA", blk.size, (0, 0, 0, 0))
            blk.alpha_composite(small, (int((blk.width - small.width) / 2), blk.height - small.height))
        self._images[key] = blk
        return blk

    def band_height(self):
        return max((self.pt.block(ln, 0).height for ln in self.laid), default=2)


def render(words, out, width, height, fps, dur, preset, y=None, max_words=None, full_frame=False):
    """Write the transparent overlay to `out` (.mov). Returns {"frames", "y", "height", "fps", "missing_glyphs"}."""
    tl = Timeline(words, preset, width, height, y, max_words)
    band_h = height if full_frame else min(height, tl.band_height() + 2)
    top = 0 if full_frame else max(0, tl.pt.bottom - band_h)
    fps = min(float(fps or OVERLAY_FPS), OVERLAY_FPS)
    total = max(1, int(round(dur * fps)))
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{band_h}",
                           "-r", f"{fps:.6g}", "-i", "-", "-c:v", "qtrle", "-pix_fmt", "argb",
                           "-metadata", f"comment=karaoke_y={top}", out], stdin=subprocess.PIPE)
    assert ff.stdin is not None
    blank = Image.new("RGBA", (width, band_h), (0, 0, 0, 0)).tobytes()
    last_key, last_bytes, ci = "-", blank, 0
    try:
        for f in range(total):
            key, ci = tl.state(f / fps, ci)
            if key != last_key:
                if key is None:
                    last_bytes = blank
                else:
                    blk = tl.image(key)
                    img = Image.new("RGBA", (width, band_h), (0, 0, 0, 0))
                    img.alpha_composite(blk, (0, max(0, (tl.pt.bottom - top) - blk.height)))
                    last_bytes = img.tobytes()
                last_key = key
            ff.stdin.write(last_bytes)
    except BrokenPipeError:
        pass
    finally:
        ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed while writing the caption overlay")
    return {"frames": total, "y": top, "height": band_h, "fps": round(fps, 3),
            "missing_glyphs": tl.missing, "fallback_font": tl.fallback}


def still(words, t, width, height, preset, y=None, max_words=None):
    """The caption layer alone at time t, full frame (for previews)."""
    tl = Timeline(words, preset, width, height, y, max_words)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    key, _ = tl.state(t)
    if key is not None:
        blk = tl.image(key)
        img.alpha_composite(blk, (0, max(0, tl.pt.bottom - blk.height)))
    return img, tl.missing, tl.fallback
