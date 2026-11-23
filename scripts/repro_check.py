"""Reproduction check: two independent evaluate runs (fresh processes) of the same checkpoint.

python scripts/repro_check.py runs/pilot-eval-v1/<tsA> runs/pilot-eval-v1/<tsB> [--name M0]
Pass if every per-scene metric matches the value logged during training within 1e-4 dB
(PSNR; same tolerance applied to SSIM) and the float outputs are bit-identical.
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TOL = 1e-4


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--name", default="M0")
    ap.add_argument("--out", default=str(REPO / "results" / "tables" / "repro_check.json"))
    args = ap.parse_args()
    A, B = Path(args.a).resolve(), Path(args.b).resolve()
    ma, mb = json.loads((A / "metrics.json").read_text()), json.loads((B / "metrics.json").read_text())
    sa, sb = ma["selection"][args.name], mb["selection"][args.name]
    assert sa["step"] == sb["step"] and sa["run_dir"] == sb["run_dir"]
    identical = {}
    for sid in sa["output_sha256"]:
        xa = np.load(A / "outputs" / args.name / f"{sid}.npy")
        xb = np.load(B / "outputs" / args.name / f"{sid}.npy")
        identical[sid] = bool(np.array_equal(xa, xb))
    res = {
        "checkpoint": f"{sa['run_dir']}/checkpoints/step_{sa['step']:06d}.pt",
        "eval_run_a": rel(A), "eval_run_b": rel(B),
        "max_abs_diff_vs_training_log_a": ma["max_abs_diff_vs_training_val_log"],
        "max_abs_diff_vs_training_log_b": mb["max_abs_diff_vs_training_val_log"],
        "outputs_bit_identical_a_vs_b": identical,
        "sha256_equal": {s: sa["output_sha256"][s] == sb["output_sha256"][s] for s in sa["output_sha256"]},
        "tolerance_db": TOL,
    }
    within = all(v <= TOL for m in (res["max_abs_diff_vs_training_log_a"], res["max_abs_diff_vs_training_log_b"])
                 for v in m.values())
    res["metrics_within_tol"] = bool(within)
    res["pass"] = bool(within and all(identical.values()))
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
