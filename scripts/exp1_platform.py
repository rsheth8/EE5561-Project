"""Exp 1: platform check (synthetic tensors; an environment check, NOT an HDR result).

launch/run_on_gpu.sh gpu0 logs/exp1-gpu0.out 1 .venv/bin/python scripts/exp1_platform.py
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hdr import SEED  # noqa: E402
from hdr.metrics import mu_law_torch  # noqa: E402
from hdr.model import AHDRNet, count_params  # noqa: E402
from hdr.runlog import make_run_dir, write_notes  # noqa: E402
from hdr.torchutil import setup_torch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, shell=isinstance(cmd, str), timeout=60).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<{exc}>"


def main():
    dev = setup_torch(SEED)
    assert dev.type == "cuda", "no GPU visible"
    cache_ok = (REPO / "cache" / "001.npz").exists()
    H, W = 1000, 1500  # dataset resolution (verified in Exp 2) / plan default
    rd = make_run_dir(REPO / "runs" / "exp1", {"exp": "exp1", "seed": SEED, "train_shape": [8, 3, 6, 256, 256],
                                                "fullres_shape": [1, 3, 6, H, W], "iters": 50, "warmup": 5,
                                                "tf32": False, "role": os.environ.get("HDR_ROLE")})
    info = {
        "host": socket.gethostname(),
        "os": platform.platform(),
        "cpu": sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2").strip(),
        "cpu_cores_node": os.cpu_count(),
        "cpus_allocated": len(os.sched_getaffinity(0)),
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "opencv": cv2.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "gpu_uuid": sh("nvidia-smi --query-gpu=uuid --format=csv,noheader"),
        "gpu_physical_index": sh("nvidia-smi --query-gpu=index --format=csv,noheader"),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (rd / "nvidia_smi.txt").write_text(sh("nvidia-smi"))

    model = AHDRNet().to(dev)
    n_params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    g = torch.Generator(device=dev).manual_seed(SEED)
    x = torch.rand(8, 3, 6, 256, 256, device=dev, generator=g)
    y = torch.rand(8, 3, 256, 256, device=dev, generator=g)
    p0 = [p.detach().clone() for p in model.parameters()]
    torch.cuda.reset_peak_memory_stats()
    times, losses = [], []
    for it in range(50):
        torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        loss = (mu_law_torch(model(x)) - mu_law_torch(y)).abs().mean()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        losses.append(float(loss))
        assert torch.isfinite(loss), f"non-finite loss at iter {it}"
        if it == 0:
            changed = any(not torch.equal(a, b) for a, b in zip(p0, model.parameters()))
            assert changed, "parameters did not change after step 1"
    it_s = (50 - 5) / sum(times[5:])
    peak_train = torch.cuda.max_memory_allocated() / 1e9

    model.eval()
    xf = torch.rand(1, 3, 6, H, W, device=dev, generator=g)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(xf)  # warm-up
        torch.cuda.synchronize()
        t0 = time.time()
        out = model(xf)
        torch.cuda.synchronize()
        t_full = time.time() - t0
    peak_full = torch.cuda.max_memory_allocated() / 1e9
    assert torch.isfinite(out).all()

    m = dict(info, n_params=n_params, train_it_per_s=it_s, train_peak_mem_gb=peak_train,
             loss_first=losses[0], loss_last=losses[-1], asserts_passed=True,
             fullres_hw=[H, W], fullres_forward_s=t_full, fullres_peak_mem_gb=peak_full,
             dataset_cached=cache_ok, note="synthetic tensors: environment check, not an HDR result")
    (rd / "metrics.json").write_text(json.dumps(m, indent=2))
    write_notes(rd, [
        "Synthetic random tensors - environment check only, not an HDR result.",
        f"GPU {info['gpu_name']} ({info['gpu_mem_gb']} GB) uuid {info['gpu_uuid']} on {info['host']}, Slurm job {info['slurm_job']}",
        f"M0 params {n_params}; train batch [8,3,6,256,256] fp32 (TF32 off): {it_s:.3f} it/s, peak {peak_train:.2f} GB",
        f"asserts passed: loss finite all 50 iters, parameters changed after step 1",
        f"full-res no-grad forward [1,3,6,{H},{W}]: {t_full:.3f} s, peak {peak_full:.2f} GB",
    ])
    # aggregate platform table over all exp1 runs
    rows = []
    for mj in sorted((REPO / "runs" / "exp1").glob("*/metrics.json")):
        r = json.loads(mj.read_text())
        r["run_dir"] = str(mj.parent.relative_to(REPO))
        rows.append(r)
    cols = ["run_dir", "host", "partition", "slurm_job", "gpu_name", "gpu_uuid", "gpu_physical_index", "gpu_mem_gb",
            "cpu", "cpus_allocated", "ram_gb", "os", "python", "torch", "cuda", "cudnn", "opencv", "n_params",
            "train_it_per_s", "train_peak_mem_gb", "fullres_hw", "fullres_forward_s", "fullres_peak_mem_gb",
            "asserts_passed", "note"]
    (REPO / "results" / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).reindex(columns=cols).to_csv(REPO / "results" / "tables" / "platform.csv", index=False)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
