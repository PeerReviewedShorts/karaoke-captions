"""ffmpeg/ffprobe helpers. ffmpeg is not bundled; the user installs it (brew, apt, winget).

probe() reports the size the viewer SEES: phone videos stored landscape with a rotation flag count as portrait, and
non-square pixels (anamorphic) are converted to square ones. The burn step applies the same conversion, so the overlay
always matches.
"""
import functools
import json
import shutil
import subprocess

import numpy as np

SR = 16000
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}      # PQ (HDR10, Dolby Vision base) and HLG (iPhone, Android)
MP4_AUDIO = {"aac", "mp3", "alac", "ac3", "eac3"}  # stream-copied into .mp4; anything else is re-encoded to AAC


class MediaError(RuntimeError):
    pass


def require(tool="ffmpeg"):
    if not shutil.which(tool):
        raise MediaError(f"{tool} not found on PATH. Install ffmpeg (macOS: brew install ffmpeg, "
                         f"Ubuntu: sudo apt install ffmpeg, Windows: winget install ffmpeg).")


def _rate(s):
    num, _, den = (s or "0/1").partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _rotation(stream):
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            return int(round(float(sd["rotation"]))) % 360
    try:
        return int((stream.get("tags") or {}).get("rotate", 0)) % 360
    except ValueError:
        return 0


def probe(path):
    """Duration, display size, frame rate, audio codec, rotation, pixel aspect and HDR transfer of a media file."""
    require("ffprobe")
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,"
                          "sample_aspect_ratio,color_transfer,pix_fmt,duration:stream_tags=rotate:stream_side_data=rotation",
                          "-of", "json", path], capture_output=True, text=True)
    if out.returncode != 0:
        raise MediaError(f"ffprobe could not read {path}: {out.stderr.strip()[:200]}")
    d = json.loads(out.stdout)
    info = {"duration": float(d.get("format", {}).get("duration", 0) or 0), "width": None, "height": None,
            "fps": None, "has_audio": False, "has_video": False, "audio_codec": None, "rotation": 0, "sar": 1.0,
            "stored_width": None, "stored_height": None, "hdr": False, "transfer": None, "pix_fmt": None}
    for s in d.get("streams", []):
        if s.get("codec_type") == "video" and not info["has_video"] and s.get("codec_name") not in ("mjpeg", "png"):
            w, h = int(s.get("width") or 0), int(s.get("height") or 0)
            if not w or not h:
                continue
            sar = _rate((s.get("sample_aspect_ratio") or "1:1").replace(":", "/")) or 1.0
            rot = _rotation(s)
            dw = w if abs(sar - 1) < 0.01 else int(round(w * sar / 2)) * 2
            if rot in (90, 270):
                dw, h = h, dw
            fps = _rate(s.get("avg_frame_rate"))
            if not 1 <= fps <= 240:
                fps = _rate(s.get("r_frame_rate"))
            info.update(width=dw, height=h, stored_width=w, stored_height=int(s.get("height") or 0), has_video=True,
                        fps=round(fps, 3) if 1 <= fps <= 240 else 30.0, rotation=rot,
                        sar=1.0 if abs(sar - 1) < 0.01 else round(sar, 4), transfer=s.get("color_transfer"),
                        hdr=s.get("color_transfer") in HDR_TRANSFERS, pix_fmt=s.get("pix_fmt"))
            if not info["duration"]:
                info["duration"] = float(s.get("duration") or 0)
        elif s.get("codec_type") == "audio" and not info["has_audio"]:
            info.update(has_audio=True, audio_codec=s.get("codec_name"))
            if not info["duration"]:
                info["duration"] = float(s.get("duration") or 0)
    return info


@functools.lru_cache(maxsize=1)
def has_filter(name):
    if not shutil.which("ffmpeg"):
        return False
    p = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
    return any(line.split()[1:2] == [name] for line in p.stdout.splitlines() if line.strip())


@functools.lru_cache(maxsize=4)
def has_encoder(name):
    if not shutil.which("ffmpeg"):
        return False
    p = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    return any(line.split()[1:2] == [name] for line in p.stdout.splitlines() if line.strip())


def read_audio(path, sr=SR):
    """Mono float32 samples at `sr` Hz, decoded by ffmpeg from any audio or video file."""
    require("ffmpeg")
    p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ar", str(sr), "-ac", "1",
                        "-f", "f32le", "-"], capture_output=True)
    if p.returncode != 0:
        raise MediaError(f"ffmpeg could not decode audio from {path}: {p.stderr.decode(errors='replace')[:200]}")
    return np.frombuffer(p.stdout, dtype="<f4").astype(np.float32)


def rms_envelope(samples, sr=SR, hop_s=0.01):
    """RMS per 10 ms frame."""
    hop = int(sr * hop_s)
    n = len(samples) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return np.sqrt((samples[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
