"""Exp 5 outputs: fig_train_curve.png, val_comparison.csv, fig_val_comparison.png.

python scripts/pilot_report.py --pilot runs/pilot_M0/<ts> --eval runs/pilot-eval-v1/<ts> --classical runs/classical-v1/<ts>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hdr.data import load_scene_np  # noqa: E402
from hdr.display import DISPLAY_MAPPING, to_uint8, write_sidecar  # noqa: E402
from hdr.metrics import METRIC_KEYS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "results" / "figures"
TAB = REPO / "results" / "tables"


def smooth(x, k=50):
    x = np.asarray(x, float)
    if len(x) < k:
        return x
    c = np.cumsum(np.insert(x, 0, 0))
    out = (c[k:] - c[:-k]) / k
    return np.concatenate([x[: k - 1] * np.nan, out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--classical", required=True)
    args = ap.parse_args()
    pilot, ev, cl = (Path(p).resolve() for p in (args.pilot, args.eval, args.classical))
    tl = pd.read_csv(pilot / "train_log.csv")
    vl = pd.read_csv(pilot / "val_log.csv", dtype={"scene": str}).drop_duplicates(["step", "scene"], keep="last")
    vm = vl[vl.scene == "mean"].sort_values("step")
    evm = json.loads((ev / "metrics.json").read_text())
    sel_step = evm["selection"]["M0"]["step"]
    pm = json.loads((pilot / "metrics.json").read_text()) if (pilot / "metrics.json").exists() else {}
    status = "finished" if (pilot / "FINISHED").exists() else f"partial (last logged update {int(tl.step.max())})"

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].semilogy(tl.step, tl.loss, alpha=0.25, label="loss (mean of 10 updates)")
    ax[0].semilogy(tl.step, smooth(tl.loss, 50), label="moving average (500 updates)")
    ax[0].set_xlabel("update")
    ax[0].set_ylabel("mu-law L1 loss")
    ax[0].legend(fontsize=8)
    ax[0].set_title("M0 pilot: training loss")
    ax[1].plot(vm.step, vm.psnr_mu, "o-", ms=3, label="M0 val PSNR-mu (5 scenes, full image)")
    c = pd.read_csv(cl / "metrics.csv", dtype={"scene": str})
    cmean = c[c.scene == "mean"].set_index("method")["psnr_mu"]
    for mname, ls in (("C0", "--"), ("C3", ":")):
        ax[1].axhline(cmean[mname], color="gray", ls=ls, label=f"{mname} = {cmean[mname]:.2f} dB")
    ax[1].axvline(sel_step, color="red", lw=0.8, label=f"selected step {sel_step}")
    ax[1].set_xlabel("update")
    ax[1].set_ylabel("PSNR-mu (dB)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("validation PSNR-mu")
    fig.tight_layout()
    fp = FIG / "fig_train_curve.png"
    fig.savefig(fp, dpi=100)
    plt.close(fig)
    write_sidecar(fp, [
        f"M0 pilot ({status}), single seed 5561, batch 8, lr 1e-4, fp32; val = 5 validation scenes, full image, mu=5000.",
        "Left: training loss logged every 10 updates (mean of those 10) and its 50-point moving average. Right: mean val PSNR-mu every 2000 updates; C0/C3 levels from the classical run.",
        f"Selected step {sel_step} = best val PSNR-mu (val-selected). Sources: {pilot.relative_to(REPO)}/train_log.csv, val_log.csv; {cl.relative_to(REPO)}/metrics.csv.",
    ])

    # val comparison table
    em = pd.read_csv(ev / "metrics.csv", dtype={"scene": str})
    em = em[em.name == "M0"].rename(columns={"name": "method"})[["scene", "method"] + METRIC_KEYS]
    cc = c[["scene", "method"] + METRIC_KEYS]
    tab = pd.concat([cc, em], ignore_index=True)
    order = {"C0": 0, "C1": 1, "C2": 2, "C3": 3, "M0": 4}
    tab["o"] = tab.method.map(order)
    tab["so"] = tab.scene.map(lambda s: "zzz" if s == "mean" else s)
    tab = tab.sort_values(["so", "o"]).drop(columns=["o", "so"])
    with (TAB / "val_comparison.csv").open("w") as fh:
        fh.write(f"# 5 validation scenes, full image, no border excluded, mu = 5000; C0-C3 from {cl.relative_to(REPO)}; "
                 f"M0 = pilot, val-selected step {sel_step}, from {ev.relative_to(REPO)} (single seed)\n")
        tab.to_csv(fh, index=False)

    # figure: same scene/crop as ladder
    gc = json.loads((TAB / "ghost_crop.json").read_text())
    sid, y, x, sz = gc["scene"], gc["y"], gc["x"], gc["size"]
    gt = load_scene_np(sid)["target_hwc"]
    ims = {"C1": np.load(cl / "outputs" / f"{sid}_C1.npy"), "C3": np.load(cl / "outputs" / f"{sid}_C3.npy"),
           "M0": np.load(ev / "outputs" / "M0" / f"{sid}.npy"), "GT": gt}
    ps = tab[tab.scene == sid].set_index("method")["psnr_mu"]
    fig, ax = plt.subplots(2, 4, figsize=(17, 7.5))
    for k, (n, im) in enumerate(ims.items()):
        ax[0, k].imshow(to_uint8(im))
        ax[0, k].add_patch(plt.Rectangle((x, y), sz, sz, fill=False, ec="yellow", lw=1.5))
        ax[0, k].set_title(n if n == "GT" else f"{n} (PSNR-mu {ps[n]:.2f} dB)", fontsize=10)
        ax[1, k].imshow(to_uint8(im[y:y + sz, x:x + sz]))
        ax[1, k].set_title(f"{n} crop", fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(f"val scene {sid}: C1, C3, M0 (pilot, step {sel_step}, val-selected), GT", fontsize=11)
    fig.tight_layout()
    fp = FIG / "fig_val_comparison.png"
    fig.savefig(fp, dpi=90)
    plt.close(fig)
    write_sidecar(fp, [
        f"Validation scene {sid}, crop y={y}, x={x}, {sz}x{sz} - same scene and crop as fig_classical_ladder.png.",
        f"C1, C3 classical (no training); M0 = AHDRNet-style pilot, step {sel_step}, val-selected, single seed; GT.",
        f"Display: {DISPLAY_MAPPING}, identical for every panel. PSNR-mu from results/tables/val_comparison.csv.",
    ])
    print(tab[tab.scene == "mean"].to_string())


if __name__ == "__main__":
    main()
