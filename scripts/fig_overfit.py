"""Figures for overfit runs (Exp 4 M0, Exp 6c M3).

python scripts/fig_overfit.py runs/overfit_M0/<ts>            -> fig_overfit_curve.png, fig_overfit_before_after.png
python scripts/fig_overfit.py runs/overfit_M3/<ts> --gate     -> fig_overfit_M3_curve.png, fig_gate_maps.png
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
from hdr.display import DISPLAY_MAPPING, LDR_MAPPING, ldr_to_uint8, to_uint8, write_sidecar  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "results" / "figures"


def hwc(a):
    return np.transpose(a, (1, 2, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    run = Path(args.run).resolve()
    m = json.loads((run / "metrics.json").read_text())
    tl = pd.read_csv(run / "train_log.csv")
    pl = pd.read_csv(run / "psnr_log.csv")
    var = m["variant"]
    crops = m["crops"]
    status = f"overfit check ({var}), {m['n_updates']} updates on one fixed batch of {len(crops)} crops; not a generalization result"
    crop_txt = "; ".join(f"scene {c['scene']} y={c['y']} x={c['x']} {c['size']}x{c['size']}" for c in crops)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].semilogy(tl.step, tl.loss)
    ax[0].set_xlabel("update")
    ax[0].set_ylabel("mu-law L1 loss (log)")
    ax[0].set_title(f"{var} overfit: training loss")
    ax[1].plot(pl.step, pl.psnr_mu, "o-", label=f"{var} (mean of 4 crops)")
    ax[1].axhline(m["c0_psnr_mu"], color="gray", ls="--", label=f"C0 = {m['c0_psnr_mu']:.2f} dB")
    ax[1].axhline(m["c0_psnr_mu"] + 5, color="red", ls=":", label="C0 + 5 dB (pass line)")
    ax[1].set_xlabel("update")
    ax[1].set_ylabel("PSNR-mu (dB)")
    ax[1].set_title(f"final {m['psnr_mu_final']:.2f} dB (margin {m['margin_over_c0_db']:+.2f} dB)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fp = FIG / ("fig_overfit_curve.png" if var == "M0" else f"fig_overfit_{var}_curve.png")
    fig.savefig(fp, dpi=100)
    plt.close(fig)
    write_sidecar(fp, [status, f"Crops: {crop_txt}.", "Loss logged every 10 updates; PSNR-mu (mean over the 4 crops) every 250 updates; C0 = H_2 on the same crops.",
                       f"Source run: {run.relative_to(REPO)}; numbers from metrics.json, train_log.csv, psnr_log.csv."])

    snap = np.load(run / "overfit_snapshots.npz")
    if not args.gate:
        idx = [0, 1]
        fig, ax = plt.subplots(len(idx), 6, figsize=(20, 7))
        for r, k in enumerate(idx):
            ldr = snap["ldr"][k]
            panels = [(ldr_to_uint8(hwc(ldr[0])), "L_1"), (ldr_to_uint8(hwc(ldr[1])), "L_2 (ref)"),
                      (ldr_to_uint8(hwc(ldr[2])), "L_3"), (to_uint8(hwc(snap["pred_step0"][k])), "output, update 0"),
                      (to_uint8(hwc(snap["pred_final"][k])), f"output, update {m['n_updates']}"),
                      (to_uint8(hwc(snap["target"][k])), "GT")]
            for c, (im, t) in enumerate(panels):
                ax[r, c].imshow(im)
                ax[r, c].set_title(f"crop {k} (scene {crops[k]['scene']}): {t}" if c == 0 else t, fontsize=9)
                ax[r, c].axis("off")
        fig.tight_layout()
        fp = FIG / "fig_overfit_before_after.png"
        fig.savefig(fp, dpi=90)
        plt.close(fig)
        write_sidecar(fp, [status, "Crops shown: " + "; ".join(
            f"crop {k} scene {crops[k]['scene']} y={crops[k]['y']} x={crops[k]['x']}" for k in idx) + ".",
            f"Inputs: {LDR_MAPPING}. Outputs and GT: {DISPLAY_MAPPING}.",
            f"Per-crop final PSNR-mu {[round(x, 2) for x in m['psnr_mu_final_per_crop']]}, C0 {[round(x, 2) for x in m['c0_psnr_mu_per_crop']]}.",
            f"Source run: {run.relative_to(REPO)}"])
    else:
        idx = [0, 1]
        cues = snap["cues"]  # [B, 2, 4, h, w] = q_ref, q_w, v, r
        fig, ax = plt.subplots(len(idx), 8, figsize=(26, 7))
        for r, k in enumerate(idx):
            panels = [(ldr_to_uint8(hwc(snap["ldr"][k][1])), "L_2", None),
                      (snap["gate1"][k, 0], "g_1", (0, 1)), (snap["gate3"][k, 0], "g_3", (0, 1)),
                      (cues[k, 0, 2], "v_1", (0, 1)), (cues[k, 1, 2], "v_3", (0, 1)),
                      (cues[k, 0, 3], "r_1", (0, 1)), (cues[k, 1, 3], "r_3", (0, 1)),
                      (cues[k, 0, 0], "q_ref", (0, 1))]
            for c, (im, t, rng) in enumerate(panels):
                if rng is None:
                    ax[r, c].imshow(im)
                else:
                    h = ax[r, c].imshow(im, cmap="viridis", vmin=rng[0], vmax=rng[1])
                ttl = t if rng is None or t.startswith(("v", "q")) else f"{t} (mean {im.mean():.3f})"
                ax[r, c].set_title(f"crop {k} (scene {crops[k]['scene']}): {ttl}" if c == 0 else ttl, fontsize=9)
                ax[r, c].axis("off")
        fig.colorbar(h, ax=ax, fraction=0.01)
        fp = FIG / "fig_gate_maps.png"
        fig.savefig(fp, dpi=85)
        plt.close(fig)
        write_sidecar(fp, [
            f"Gate maps of M3 at the end of the overfit check ({m['n_updates']} updates, fixed batch of 4 crops); status: gate implemented and verified to train on a fixed batch.",
            "Crops shown: " + "; ".join(f"crop {k} scene {crops[k]['scene']} y={crops[k]['y']} x={crops[k]['x']}" for k in idx) + ".",
            "Panels: L_2 (" + LDR_MAPPING + "); g_1, g_3 (gate in [0,1]); v_i validity; r_i = exp(-e/2) FB cue; q_ref exposure cue; viridis 0..1.",
            "Cues are estimated (DIS flow, FB check), not ground-truth motion.",
            f"Source run: {run.relative_to(REPO)}"])
    print(fp)


if __name__ == "__main__":
    main()
