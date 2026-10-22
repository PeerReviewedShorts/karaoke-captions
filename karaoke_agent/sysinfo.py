"""Facts about this machine for choosing a Whisper setup. Facts only: the decision is the agent's (see AGENTS.md)."""
import glob
import importlib.util
import os
import platform
import shutil
import subprocess
import sys


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _ram_gb():
    system = platform.system()
    try:
        if system == "Darwin":
            return round(int(_run(["sysctl", "-n", "hw.memsize"])) / 2 ** 30, 1)
        if system == "Linux":
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) / 2 ** 20, 1)
        if system == "Windows":
            import ctypes

            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEMSTAT()
            m.dwLength = ctypes.sizeof(MEMSTAT)
            # windll only exists on Windows
            kernel32 = ctypes.windll.kernel32  # pyright: ignore[reportAttributeAccessIssue]
            kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.ullTotalPhys / 2 ** 30, 1)
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _cpu():
    system = platform.system()
    if system == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _gpus():
    out = []
    if shutil.which("nvidia-smi"):
        for line in _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader,nounits"]).splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                out.append({"vendor": "nvidia", "name": parts[0], "vram_gb": round(float(parts[1]) / 1024, 1),
                            "driver": parts[2] if len(parts) > 2 else None})
    if shutil.which("rocm-smi"):
        out.append({"vendor": "amd", "name": "ROCm device (see rocm-smi)", "vram_gb": None})
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        out.append({"vendor": "apple", "name": _cpu(), "vram_gb": None, "note": "unified memory, Metal"})
    return out


def _torch():
    if not importlib.util.find_spec("torch"):
        return None
    try:
        import torch
        return {"version": torch.__version__, "cuda": bool(torch.cuda.is_available()),
                "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())}
    except Exception as e:                               # a broken torch install should not break sysinfo
        return {"error": str(e)[:200]}


def _version(mod):
    try:
        from importlib.metadata import version
        return version(mod)
    except Exception:
        return "installed"


def whisper_backends():
    """What is installed right now."""
    b = {}
    exe = shutil.which("whisper-cli") or shutil.which("whisper-cpp")
    b["whisper.cpp"] = {"installed": bool(exe), "binary": exe}
    for name, mod, dist in (("faster-whisper", "faster_whisper", "faster-whisper"), ("mlx-whisper", "mlx_whisper", "mlx-whisper"),
                            ("openai-whisper", "whisper", "openai-whisper")):
        ok = bool(importlib.util.find_spec(mod))
        b[name] = {"installed": ok, "version": _version(dist) if ok else None}
    return b


def whisper_models():
    """Whisper models already on disk (nothing to download for these)."""
    from .whisper import GGML_DIRS
    found, seen = [], set()
    for d in filter(None, GGML_DIRS):
        for f in sorted(glob.glob(os.path.join(os.path.expanduser(d), "ggml-*.bin"))):
            real = os.path.realpath(f)
            if real not in seen:
                seen.add(real)
                found.append({"backend": "whisper.cpp", "model": os.path.basename(f)[5:-4], "path": f,
                              "size_gb": round(os.path.getsize(f) / 2 ** 30, 2)})
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    for pattern, backend in (("models--Systran--faster-whisper-*", "faster-whisper"),
                             ("models--mobiuslabsgmbh--faster-whisper-*", "faster-whisper"),
                             ("models--mlx-community--whisper-*", "mlx-whisper")):
        for d in sorted(glob.glob(os.path.join(hub, pattern))):
            size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)
            found.append({"backend": backend, "model": os.path.basename(d).split("--")[-1], "path": d,
                          "size_gb": round(size / 2 ** 30, 2)})
    whisper_dir = os.path.expanduser("~/.cache/whisper")
    for f in sorted(glob.glob(os.path.join(whisper_dir, "*.pt"))):
        found.append({"backend": "openai-whisper", "model": os.path.basename(f)[:-3], "path": f,
                      "size_gb": round(os.path.getsize(f) / 2 ** 30, 2)})
    return found


def install_hints(system=None, machine=None):
    """How each backend gets onto this platform (facts for the agent, not a recommendation)."""
    system, machine = system or platform.system(), machine or platform.machine()
    apple = system == "Darwin" and machine == "arm64"
    h = {
        "whisper.cpp": {"possible": True,
                        "install": "brew install whisper-cpp" if system == "Darwin" else
                        "build from https://github.com/ggml-org/whisper.cpp (cmake; CUDA/Vulkan/Metal options) "
                        "or use a release binary",
                        "models": "ggml-<model>.bin from https://huggingface.co/ggerganov/whisper.cpp (0.15-3 GB)",
                        "accelerates_on": "Metal (Apple), CUDA, Vulkan, CPU"},
        "mlx-whisper": {"possible": apple, "install": "pip install mlx-whisper" if apple else "Apple Silicon only",
                        "models": "mlx-community/whisper-<model> on Hugging Face (downloaded on first use)",
                        "accelerates_on": "Apple Silicon GPU"},
        "faster-whisper": {"possible": True, "install": "pip install faster-whisper",
                           "models": "downloaded on first use (CTranslate2)",
                           "accelerates_on": "NVIDIA CUDA (needs cuBLAS + cuDNN), otherwise CPU int8"},
        "openai-whisper": {"possible": True, "install": "pip install openai-whisper",
                           "models": "downloaded on first use", "accelerates_on": "CUDA via torch, otherwise CPU (slow)"},
    }
    return h


def collect():
    du = shutil.disk_usage(os.getcwd())
    return {
        "os": platform.system(), "os_version": platform.mac_ver()[0] or platform.release(), "arch": platform.machine(),
        "cpu": _cpu(), "cores": os.cpu_count(), "ram_gb": _ram_gb(), "gpus": _gpus(), "torch": _torch(),
        "python": sys.version.split()[0], "ffmpeg": shutil.which("ffmpeg"), "disk_free_gb": round(du.free / 2 ** 30, 1),
        "whisper_installed": whisper_backends(), "whisper_models": whisper_models(), "whisper_options": install_hints(),
    }
