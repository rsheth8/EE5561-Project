"""Run-directory bookkeeping (EXPERIMENT_PLAN.md section 0.7)."""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<failed: {exc}>"


def gpu_identity() -> dict:
    """Physical GPU identity. Under Slurm the job sees cuda:0; record what it maps to."""
    info = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
        "SLURM_JOB_PARTITION": os.environ.get("SLURM_JOB_PARTITION"),
        "SLURM_STEP_GPUS": os.environ.get("SLURM_STEP_GPUS") or os.environ.get("SLURM_JOB_GPUS"),
        "host": socket.gethostname(),
    }
    q = _sh(["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,memory.total", "--format=csv,noheader"])
    info["nvidia_smi_visible"] = q
    return info


def git_commit_text() -> str:
    h = _sh(["git", "rev-parse", "HEAD"]) or "<no commit>"
    st = _sh(["git", "status", "--porcelain"])
    txt = h + "\n"
    if st:
        txt += "DIRTY\n" + _sh(["git", "diff", "--stat"]) + "\nuntracked/modified:\n" + st + "\n"
    return txt


def env_text() -> str:
    lines = [f"python {sys.version}", f"platform {platform.platform()}", f"host {socket.gethostname()}"]
    try:
        import torch

        lines.append(f"torch {torch.__version__} cuda {torch.version.cuda} cudnn {torch.backends.cudnn.version()}")
        lines.append(f"cuda_available {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"device0 {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"torch <{exc}>")
    for mod in ("numpy", "cv2", "skimage", "matplotlib", "pandas"):
        try:
            m = __import__(mod)
            lines.append(f"{mod} {m.__version__}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"{mod} <{exc}>")
    lines.append("")
    lines.append(_sh([str(REPO / ".venv/bin/python"), "-m", "pip", "--version"]))
    return "\n".join(lines) + "\n"


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_dir(base: str | os.PathLike, config: dict, run_dir: str | os.PathLike | None = None) -> Path:
    """Create <base>/<timestamp>/ (or use run_dir) and write config/git/env files."""
    rd = (Path(run_dir) if run_dir else Path(base) / timestamp()).resolve()
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "figures").mkdir(exist_ok=True)
    cfg = dict(config)
    cfg["_gpu"] = gpu_identity()
    cfg["_argv"] = sys.argv
    cfg["_started"] = _dt.datetime.now().isoformat(timespec="seconds")
    (rd / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    (rd / "git_commit.txt").write_text(git_commit_text())
    (rd / "env.txt").write_text(env_text())
    return rd


def write_notes(rd: str | os.PathLike, lines: list[str]) -> None:
    Path(rd, "notes.md").write_text("\n".join(f"- {l}" for l in lines) + "\n")


def load_json(path) -> dict:
    return json.loads(Path(path).read_text())


STORE = Path(os.environ.get("HDR_STORE", "/projects/standard/ee5561/shared/zha00175_hdr"))


def big_dir(rd: Path, name: str) -> Path:
    """<rd>/<name> as a symlink into project space (home quota is tight); plain dir otherwise."""
    rd = Path(rd).resolve()
    d = rd / name
    if d.exists():
        return d
    if STORE.exists():
        try:
            rel = rd.relative_to(REPO / "runs")
        except ValueError:
            rel = Path(rd.name)
        real = STORE / name / rel
        real.mkdir(parents=True, exist_ok=True)
        d.symlink_to(real)
    else:
        d.mkdir(parents=True)
    return d
