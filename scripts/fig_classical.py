"""Exp 3 figures from a classical run: ladder, error maps, C3 good/failure cases.

python scripts/fig_classical.py runs/classical-v1/<ts> [--failure-class "..."] [--failure-note "..."]
Writes results/figures/fig_classical_{ladder,error,cases}.png + sidecars, and
results/tables/ghost_crop.json (scene + crop reused by later figures).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hdr.data import load_scene_np  # noqa: E402
from hdr.display import DISPLAY_MAPPING, to_uint8, write_sidecar  # noqa: E402
from hdr.metrics import mu_law  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "results" / "figures"
CROP = 300
ERR_VMAX = 0.3
METHODS = ["C0", "C1", "C2", "C3"]


def worst_window(err: np.ndarray, size: int) -> tuple[int, int]:
    box = cv2.boxFilter(err.astype(np.float32), -1, (size, size), normalize=True, borderType=cv2.BORDER_CONSTANT)
    h, w = err.shape
    half = size // 2
    box[:half, :] = box[h - half:, :] = -1
    box[:, :half] = box[:, w - half:] = -1
    cy, cx = np.unravel_index(np.argmax(box), box.shape)
    return int(cy - half), int(cx - half)


def rect(ax, y, x, s, color="yellow"):
    ax.add_patch(plt.Rectangle((x, y), s, s, fill=False, ec=color, lw=1.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--failure-class", default="(not yet classified)")
    ap.add_argument("--failure-note", default="")
    ap.add_argument("--good-note", default="")
    args = ap.parse_args()
    run = Path(args.run).resolve()
    od = run / "outputs"
    met = pd.read_csv(run / "metrics.csv", dtype={"scene": str})
    scenes = sorted(s for s in met.scene.unique() if s != "mean")
    data = {}
    for sid in scenes:
        gt = load_scene_np(sid)["target_hwc"]
        preds = {m: np.load(od / f"{sid}_{m}.npy") for m in METHODS}
        data[sid] = (gt, preds)
    ghost_score = {sid: float(np.abs(mu_law(p["C1"]) - mu_law(g)).mean()) for sid, (g, p) in data.items()}
    gs = max(ghost_score, key=ghost_score.get)
    gt, preds = data[gs]
    e1 = np.abs(mu_law(preds["C1"]) - mu_law(gt)).mean(-1)
    y, x = worst_window(e1, CROP)
    (REPO / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (REPO / "results" / "tables" / "ghost_crop.json").write_text(json.dumps(
        {"scene": gs, "y": y, "x": x, "size": CROP, "ghost_scores_mean_abs_T_C1_minus_T_GT": ghost_score,
         "source_run": str(run.relative_to(REPO))}, indent=2))
    sl = (slice(y, y + CROP), slice(x, x + CROP))
    psnr = met.set_index(["scene", "method"])["psnr_mu"]

    # ladder
    fig, ax = plt.subplots(2, 5, figsize=(20, 7.5))
    for k, m in enumerate(METHODS + ["GT"]):
        im = gt if m == "GT" else preds[m]
        ax[0, k].imshow(to_uint8(im))
        rect(ax[0, k], y, x, CROP)
        ax[0, k].set_title(m if m == "GT" else f"{m}  (PSNR-mu {psnr[(gs, m)]:.2f} dB)", fontsize=10)
        ax[1, k].imshow(to_uint8(im[sl]))
        ax[1, k].set_title(f"{m} crop", fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(f"val scene {gs}: classical merges; crop y={y}, x={x}, {CROP}x{CROP}", fontsize=11)
    fig.tight_layout()
    fp = FIG / "fig_classical_ladder.png"
    fig.savefig(fp, dpi=90)
    plt.close(fig)
    write_sidecar(fp, [
        f"Validation scene {gs} (largest mean|T(C1)-T(GT)| among the 5 val scenes: "
        + ", ".join(f"{k}={v:.4f}" for k, v in ghost_score.items()) + ").",
        f"Top row full frame with crop box; bottom row crop y={y}, x={x}, size {CROP}x{CROP} (window with largest mean C1 error).",
        "Methods: C0 reference only; C1 hat-weighted merge (no alignment); C2 reference-fallback deghosting; C3 C2 on DIS-aligned frames. Not learned; no training.",
        f"Display: {DISPLAY_MAPPING}, same mapping for every panel. PSNR-mu in titles from {run.relative_to(REPO)}/metrics.csv.",
    ])

    # error maps
    fig, ax = plt.subplots(2, 4, figsize=(17, 7.5))
    for k, m in enumerate(METHODS):
        e = np.abs(mu_law(preds[m]) - mu_law(gt)).mean(-1)
        h = ax[0, k].imshow(e, cmap="inferno", vmin=0, vmax=ERR_VMAX)
        rect(ax[0, k], y, x, CROP, "cyan")
        ax[0, k].set_title(f"{m}: |T(.)-T(GT)| mean {e.mean():.4f}", fontsize=10)
        ax[1, k].imshow(e[sl], cmap="inferno", vmin=0, vmax=ERR_VMAX)
        ax[1, k].set_title(f"{m} crop", fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    fig.colorbar(h, ax=ax, fraction=0.015, label="|T(H) - T(H_gt)| (channel mean)")
    fig.suptitle(f"val scene {gs}: mu-law absolute error, shared scale 0..{ERR_VMAX}", fontsize=11)
    fp = FIG / "fig_classical_error.png"
    fig.savefig(fp, dpi=90)
    plt.close(fig)
    write_sidecar(fp, [
        f"Validation scene {gs}; error = channel mean of |T(H) - T(H_gt)|, T = mu-law (mu=5000) on clip(H,0,1).",
        f"Shared colour scale 0..{ERR_VMAX} (inferno) for all panels; crop y={y}, x={x}, {CROP}x{CROP}.",
        "Methods C0-C3 as in fig_classical_ladder; classical, no training.",
    ])

    # C3 good / failure cases
    c3 = met[(met.method == "C3") & (met.scene != "mean")].set_index("scene")["psnr_mu"]
    good = c3.idxmax()
    worst = None
    for sid, (g, p) in data.items():
        e = np.abs(mu_law(p["C3"]) - mu_law(g)).mean(-1)
        yy, xx = worst_window(e, CROP)
        v = float(e[yy:yy + CROP, xx:xx + CROP].mean())
        if worst is None or v > worst[0]:
            worst = (v, sid, yy, xx)
    _, fs, fy, fx = worst
    gg, gp = data[good]
    eg = np.abs(mu_law(gp["C3"]) - mu_law(gg)).mean(-1)
    gy, gx = worst_window(eg, CROP)  # even in the good scene show its hardest window
    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    for r, (sid, yy, xx, tag) in enumerate([(good, gy, gx, "good case"), (fs, fy, fx, "failure")]):
        g, p = data[sid]
        s2 = (slice(yy, yy + CROP), slice(xx, xx + CROP))
        panels = [(to_uint8(p["C1"][s2]), "C1 crop"), (to_uint8(p["C3"][s2]), "C3 crop"), (to_uint8(g[s2]), "GT crop"),
                  (np.abs(mu_law(p["C3"][s2]) - mu_law(g[s2])).mean(-1), "|T(C3)-T(GT)|")]
        for k, (im, t) in enumerate(panels):
            if im.ndim == 2:
                ax[r, k].imshow(im, cmap="inferno", vmin=0, vmax=ERR_VMAX)
            else:
                ax[r, k].imshow(im)
            ax[r, k].set_title(f"{tag}: scene {sid} - {t}" if k == 0 else t, fontsize=9)
            ax[r, k].axis("off")
    fig.suptitle(f"C3 good case (scene {good}, PSNR-mu {c3[good]:.2f}) vs visible failure (scene {fs}, PSNR-mu {c3[fs]:.2f})",
                 fontsize=11)
    fig.tight_layout()
    fp = FIG / "fig_classical_cases.png"
    fig.savefig(fp, dpi=90)
    plt.close(fig)
    write_sidecar(fp, [
        f"Good case: val scene {good} (highest C3 PSNR-mu {c3[good]:.3f} dB), crop y={gy}, x={gx}, {CROP}x{CROP} (its largest-C3-error window). {args.good_note}",
        f"Failure: val scene {fs}, crop y={fy}, x={fx}, {CROP}x{CROP} = the window with the largest mean |T(C3)-T(GT)| over all 5 val scenes ({worst[0]:.4f}).",
        f"Failure classification: {args.failure_class}. {args.failure_note}",
        f"Display: {DISPLAY_MAPPING}; error maps shared scale 0..{ERR_VMAX}. Classical methods, no training.",
    ])
    print(json.dumps({"ghost_scene": gs, "crop": [y, x], "good": [good, gy, gx], "failure": [fs, fy, fx]}))


if __name__ == "__main__":
    main()
