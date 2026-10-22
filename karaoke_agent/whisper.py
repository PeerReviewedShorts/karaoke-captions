"""Whisper as a second opinion: one adapter per backend, all returning the same shape.

    transcribe(clip, backend, model, language=None, prompt=None) -> {"text", "segments": [{"s", "e", "text"}], "backend", "model"}

Only the TEXT is used (compared word by word against Parakeet). Whisper's timestamps are kept for reference but never
replace Parakeet's timing.
"""
import glob
import json
import os
import shutil
import subprocess
import tempfile
import time

from karaoke import media

BACKENDS = ("whisper.cpp", "mlx-whisper", "faster-whisper", "openai-whisper")
GGML_DIRS = [os.environ.get("KARAOKE_WHISPER_MODELS", ""), "~/.cache/whisper.cpp", "~/.cache/whisper", "~/models",
             "/opt/homebrew/share/whisper-cpp/models", "/usr/local/share/whisper-cpp/models"]


class BackendError(RuntimeError):
    pass


def _lang(language):
    return None if language in (None, "", "auto") else language


def ggml_path(model):
    """A ggml model for whisper.cpp: a path, or a name such as large-v3-turbo looked up in the usual folders."""
    if os.path.isfile(os.path.expanduser(model)):
        return os.path.expanduser(model)
    for d in filter(None, GGML_DIRS):
        hits = glob.glob(os.path.join(os.path.expanduser(d), f"ggml-{model}.bin"))
        if hits:
            return hits[0]
    raise BackendError(f"no ggml-{model}.bin found (looked in KARAOKE_WHISPER_MODELS, ~/.cache/whisper.cpp, ...). "
                       f"Download: https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model}.bin "
                       f"or pass the file path as --model")


def available(backend, model=None):
    """(ok, message) without running a transcription."""
    try:
        if backend == "whisper.cpp":
            exe = shutil.which("whisper-cli") or shutil.which("whisper-cpp")
            if not exe:
                return False, "whisper.cpp binary (whisper-cli) not on PATH"
            if model:
                ggml_path(model)
            return True, exe
        mod = {"mlx-whisper": "mlx_whisper", "faster-whisper": "faster_whisper", "openai-whisper": "whisper"}[backend]
        import importlib.util
        if not importlib.util.find_spec(mod):
            return False, f"python package for {backend} not installed"
        return True, mod
    except BackendError as e:
        return False, str(e)
    except KeyError:
        return False, f"unknown backend {backend!r} (known: {', '.join(BACKENDS)})"


def _whisper_cpp(clip, model, language, prompt):
    exe = shutil.which("whisper-cli") or shutil.which("whisper-cpp")
    if not exe:
        raise BackendError("whisper.cpp binary (whisper-cli) not on PATH")
    mp = ggml_path(model)
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", clip, "-vn", "-ar", "16000", "-ac", "1",
                        "-c:a", "pcm_s16le", wav], check=True)
        cmd = [exe, "-m", mp, "-l", _lang(language) or "auto", "-oj", "-of", os.path.join(tmp, "out"), "-f", wav]
        if prompt:
            cmd += ["--prompt", prompt]
        p = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        out = os.path.join(tmp, "out.json")
        if p.returncode != 0 or not os.path.exists(out):
            raise BackendError(f"whisper.cpp failed: {p.stderr.strip()[-300:]}")
        d = json.load(open(out, encoding="utf-8"))
    return [{"s": s["offsets"]["from"] / 1000, "e": s["offsets"]["to"] / 1000, "text": s["text"].strip()}
            for s in d.get("transcription", []) if s.get("text", "").strip()]


def _mlx(clip, model, language, prompt):
    import mlx_whisper  # pyright: ignore[reportMissingImports]
    repo = model if "/" in model else f"mlx-community/whisper-{model}"
    r = mlx_whisper.transcribe(clip, path_or_hf_repo=repo, language=_lang(language), initial_prompt=prompt or None)
    return [{"s": s["start"], "e": s["end"], "text": s["text"].strip()} for s in r.get("segments", [])]


def _faster(clip, model, language, prompt, compute_type="default", device="auto"):
    from faster_whisper import WhisperModel  # pyright: ignore[reportMissingImports]
    m = WhisperModel(model, device=device, compute_type=compute_type)
    segs, _info = m.transcribe(clip, language=_lang(language), initial_prompt=prompt or None, beam_size=5)
    return [{"s": s.start, "e": s.end, "text": s.text.strip()} for s in segs]


def _openai(clip, model, language, prompt):
    import whisper
    # `whisper` is the openai-whisper package here, not this module
    m = whisper.load_model(model)  # pyright: ignore[reportAttributeAccessIssue]
    r = m.transcribe(clip, language=_lang(language), initial_prompt=prompt or None)
    return [{"s": s["start"], "e": s["end"], "text": s["text"].strip()} for s in r.get("segments", [])]


def transcribe(clip, backend, model, language=None, prompt=None, **opts):
    media.require("ffmpeg")
    t0 = time.time()
    fn = {"whisper.cpp": _whisper_cpp, "mlx-whisper": _mlx, "faster-whisper": _faster, "openai-whisper": _openai}.get(backend)
    if not fn:
        raise BackendError(f"unknown backend {backend!r} (known: {', '.join(BACKENDS)})")
    try:
        segs = fn(clip, model, language, prompt, **opts) if backend == "faster-whisper" else fn(clip, model, language, prompt)
    except ImportError as e:
        raise BackendError(f"{backend} is not installed: {e}") from e
    return {"backend": backend, "model": model, "language": language or "auto", "prompt": prompt or "",
            "seconds": round(time.time() - t0, 1), "text": " ".join(s["text"] for s in segs), "segments": segs}
