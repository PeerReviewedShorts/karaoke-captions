"""karaoke.toml in the working folder: the agent's choices, with the reason, so the next run (or the next agent) knows why.

    [whisper]
    backend = "whisper.cpp"
    model = "large-v3-turbo"
    language = "auto"
    reason = "Apple M1, 16 GB, no CUDA: whisper.cpp with Metal"
    decided = "2026-10-15T12:00:00"
"""
import os
import tomllib

NAME = "karaoke.toml"


def path(folder=None):
    return os.path.join(folder or os.getcwd(), NAME)


def load(p=None):
    p = p or path()
    if not os.path.exists(p):
        return {}
    with open(p, "rb") as fh:
        return tomllib.load(fh)


def _val(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def save(cfg, p=None):
    """Flat [section] tables of str/int/float/bool only — that is all karaoke.toml holds."""
    p = p or path()
    lines = []
    for sec, table in cfg.items():
        lines.append(f"[{sec}]")
        lines += [f"{k} = {_val(v)}" for k, v in table.items() if v is not None]
        lines.append("")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return p
