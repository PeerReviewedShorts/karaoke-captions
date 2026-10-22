"""Speech to timed words in ONE pass with NVIDIA Parakeet-TDT (nvidia/parakeet-tdt-0.6b-v3, CC BY 4.0, 25 European languages).

Why one model is enough: a TDT (token-and-duration transducer) head predicts, for every token, how many encoder frames it
spans. The word timings are the model's own output, not a guess spread over a sentence afterwards (which is what Whisper's
word timestamps are). Resolution is one encoder frame = 80 ms; on test sentences with exactly known word boundaries the
median error was 34-60 ms. Known weakness: proper names ("Aptronics" for "Apptronik's") — that is what the review file,
`karaoke fix` and the agent tool's second opinion are for.

Runs with transformers (>= 5) + torch on CPU. Long audio is cut into windows at quiet points.
"""
import os
import time
import warnings

import numpy as np

from . import media

MODEL = "nvidia/parakeet-tdt-0.6b-v3"
LICENSE = "CC-BY-4.0"
FRAME = 0.08                    # TDT durations count 80 ms encoder frames
WINDOW = 120.0                  # seconds per decoding window for long audio
_CACHE = {}


def pick_device(device="auto"):
    """auto -> cuda if available, else mps (Apple Silicon), else cpu."""
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load(model, device):
    key = (model, device)
    if key not in _CACHE:
        try:
            from transformers import ParakeetForTDT, ParakeetProcessor
        except ImportError as e:                    # pragma: no cover
            raise RuntimeError("Parakeet needs transformers >= 5 (pip install -U transformers torch librosa)") from e
        m = ParakeetForTDT.from_pretrained(model).to(device).eval()
        _CACHE[key] = (m, ParakeetProcessor.from_pretrained(model))
    return _CACHE[key]


def _decode(wav, model, proc, device, t_off):
    import torch
    dur = len(wav) / media.SR
    inp = proc(wav, sampling_rate=media.SR, return_tensors="pt").to(device)
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*max_length.*")     # transformers' generic generate() hint, not ours
        o = model.generate(**inp)
    seq, durs = o.sequences[0].tolist(), o.durations[0].tolist()
    tok = proc.tokenizer
    # A word ends after its OWN last token (ce), not where the next word starts: the <blank> frames in between are the
    # pause. At least one frame, never past the next word's start.
    pos, out, cur, cs, ce = 0, [], None, 0, 0
    for piece, d in zip(tok.convert_ids_to_tokens(seq), durs):
        if piece and piece != "<blank>" and piece.strip():
            if piece.startswith("▁") or cur is None:        # sentencepiece marks a word start with U+2581
                if cur:
                    end = min(max(ce, cs + 1), pos)
                    out.append((cur, cs * FRAME, min(end * FRAME, dur)))
                cur, cs = piece.replace("▁", ""), pos
            else:
                cur += piece
            ce = pos + d
        pos += d
    if cur:
        out.append((cur, cs * FRAME, min(max(ce, cs + 1) * FRAME, dur)))
    return [{"w": w, "s": round(s + t_off, 3), "e": round(e + t_off, 3)} for w, s, e in out if w.strip()]


def _run(wav, model, device):
    m, proc = _load(model, device)
    words = []
    for a, b in _windows(wav):
        words += _decode(wav[a:b], m, proc, device, a / media.SR)
    return words


def _windows(wav, window=WINDOW):
    """Cut points for long audio: the quietest 100 ms in the last 15 s of each window."""
    n, sr = len(wav), media.SR
    cuts, start = [0], 0
    while n - start > window * sr:
        lo, hi = start + int((window - 15) * sr), start + int(window * sr)
        env = media.rms_envelope(wav[lo:hi])
        k = int(np.argmin(np.convolve(env, np.ones(10), "same"))) if len(env) else 0
        start = lo + k * int(sr * 0.01)
        cuts.append(start)
    cuts.append(n)
    return list(zip(cuts[:-1], cuts[1:]))


def transcribe(path, device="auto", model=MODEL, log=None):
    """Audio or video file -> (words, provenance). A GPU device that fails falls back to the CPU."""
    t0 = time.time()
    wav = media.read_audio(path)
    if len(wav) < media.SR * 0.3:
        raise media.MediaError(f"{path}: less than 0.3 s of audio")
    device = pick_device(device)
    try:
        words = _run(wav, model, device)
    except (RuntimeError, NotImplementedError) as e:
        if device == "cpu":
            raise
        if log:
            log(f"parakeet  {device} failed ({str(e).splitlines()[0][:120]}), falling back to cpu")
        device = "cpu"
        words = _run(wav, model, device)
    prov = {"model": model, "license": LICENSE, "tool": "karaoke (parakeet, text + timing in one pass)",
            "device": device, "frame_s": FRAME, "audio": os.path.basename(path),
            "seconds": round(time.time() - t0, 1), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return words, prov
