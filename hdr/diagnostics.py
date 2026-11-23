"""Exp 6a/6b: reliability cue maps and cue-vs-alignment-error statistics.

python -m hdr.diagnostics --out runs/cues-v1 [--scenes 062 025]
  6a -> results/figures/fig_reliability_maps.png (+ .txt)
  6b -> results/tables/cue_diagnostics_val.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from . import GAMMA  # noqa: E402
from .alignment import exposure_match, to_gray8  # noqa: E402
from .data import load_flow_np, load_scene_np, load_split  # noqa: E402
from .display import LDR_MAPPING, ldr_to_uint8, write_sidecar  # noqa: E402
from .metrics import mu_law  # noqa: E402
from .runlog import make_run_dir, write_notes  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
Q_MIN = 0.2
ERR_LARGE = 0.05
REF_CLIP = 0.95
ERR_VMAX = 0.3


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, n = int(label.sum()), label.size
    neg = n - pos
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def warped_error(f, s, k):
    t = s["t"]
    i = 0 if k == 0 else 2
    hw = f["Lw"][k].astype(np.float64) ** GAMMA / t[i]
    return np.abs(mu_law(hw) - mu_law(s["target_hwc"])).mean(-1)


def stats_rows(sid: str) -> list[dict]:
    s, f = load_scene_np(sid), load_flow_np(sid)
    m2 = s["ldr_hwc"][1].max(-1)
    rows = []
    for k, name in ((0, "L1"), (1, "L3")):
        err = warped_error(f, s, k)
        use = (f["v"][k] == 1) & (f["q_w"][k] > Q_MIN)
        for stratum, sm in (("all", use), ("ref_unclipped", use & (m2 <= REF_CLIP)), ("ref_clipped", use & (m2 > REF_CLIP))):
            e, inc, r = err[sm], f["incons"][k][sm].astype(bool), f["r"][k][sm]
            large = e > ERR_LARGE
            rows.append({
                "scene": sid, "frame": name, "stratum": stratum, "n_pix": int(sm.sum()),
                "frac_of_frame": float(sm.mean()), "frac_inconsistent": float(inc.mean()) if e.size else np.nan,
                "mean_err_consistent": float(e[~inc].mean()) if (~inc).any() else np.nan,
                "mean_err_inconsistent": float(e[inc].mean()) if inc.any() else np.nan,
                "frac_large_err": float(large.mean()) if e.size else np.nan,
                "frac_large_err_among_consistent": float(large[~inc].mean()) if (~inc).any() else np.nan,
                "auroc_1_minus_r": auroc(1 - r, large) if e.size else np.nan,
                "_e": e, "_inc": inc, "_r": r,
            })
    return rows


def pooled(rows: list[dict]) -> list[dict]:
    out = []
    df = pd.DataFrame(rows)
    for (frame, stratum), g in df.groupby(["frame", "stratum"]):
        e = np.concatenate(list(g["_e"]))
        inc = np.concatenate(list(g["_inc"]))
        r = np.concatenate(list(g["_r"]))
        large = e > ERR_LARGE
        out.append({"scene": "pooled", "frame": frame, "stratum": stratum, "n_pix": int(e.size),
                    "frac_of_frame": np.nan, "frac_inconsistent": float(inc.mean()) if e.size else np.nan,
                    "mean_err_consistent": float(e[~inc].mean()) if (~inc).any() else np.nan,
                    "mean_err_inconsistent": float(e[inc].mean()) if inc.any() else np.nan,
                    "frac_large_err": float(large.mean()) if e.size else np.nan,
                    "frac_large_err_among_consistent": float(large[~inc].mean()) if (~inc).any() else np.nan,
                    "auroc_1_minus_r": auroc(1 - r, large) if e.size else np.nan})
    return out


def cue_failure_box(sid: str, k: int, size: int = 96):
    """Window with most 'consistent but wrong' pixels in a clipped-reference region."""
    import cv2

    s, f = load_scene_np(sid), load_flow_np(sid)
    m2 = s["ldr_hwc"][1].max(-1)
    err = warped_error(f, s, k)
    bad = ((f["incons"][k] == 0) & (f["v"][k] == 1) & (m2 > REF_CLIP) & (err > ERR_LARGE)).astype(np.float32)
    box = cv2.boxFilter(bad, -1, (size, size), normalize=True, borderType=cv2.BORDER_CONSTANT)
    cy, cx = np.unravel_index(np.argmax(box), box.shape)
    return int(max(cy - size // 2, 0)), int(max(cx - size // 2, 0)), size, float(box[cy, cx]), int(bad.sum())


def fig_maps(scenes: list[str], fp: Path, rd: Path):
    fig, ax = plt.subplots(2 * len(scenes), 7, figsize=(28, 4.2 * len(scenes)))
    notes = []
    for si, sid in enumerate(scenes):
        s, f = load_scene_np(sid), load_flow_np(sid)
        L, t = s["ldr_hwc"], s["t"]
        for k, i in ((0, 0), (1, 2)):
            r = 2 * si + k
            if t[i] < t[1]:
                ref_m, src_m = L[1], exposure_match(L[i], t[i], t[1])
            else:
                ref_m, src_m = exposure_match(L[1], t[1], t[i]), L[i]
            g_ref, g_src = to_gray8(ref_m), to_gray8(src_m)
            pair = np.stack([g_ref, g_src, g_src], -1)  # red = reference, cyan = source
            err = warped_error(f, s, k)
            y, x, sz, dens, nbad = cue_failure_box(sid, k)
            panels = [(pair, f"exposure-matched pair (red=ref L_2, cyan=L_{i+1})", None),
                      (ldr_to_uint8(f["Lw"][k]), f"L_{i+1}^w (warped)", None),
                      (f["v"][k], f"v_{i+1} (valid {f['v'][k].mean():.3f})", (0, 1)),
                      (f["incons"][k], f"FB-inconsistent ({f['incons'][k].mean():.3f})", (0, 1)),
                      (f["q_ref"], "q_ref", (0, 1)), (f["q_w"][k], f"q_{i+1}^w", (0, 1)),
                      (err, f"|T(H_{i+1}^w)-T(H_gt)| (0..{ERR_VMAX})", (0, ERR_VMAX))]
            for c, (im, ttl, rg) in enumerate(panels):
                if rg is None:
                    ax[r, c].imshow(im)
                else:
                    ax[r, c].imshow(im, cmap="inferno" if c == 6 else "gray", vmin=rg[0], vmax=rg[1])
                if c in (3, 6) and nbad > 0:
                    ax[r, c].add_patch(plt.Rectangle((x, y), sz, sz, fill=False, ec="lime", lw=1.5))
                ax[r, c].set_title((f"scene {sid}: " if c == 0 else "") + ttl, fontsize=8)
                ax[r, c].axis("off")
            notes.append(f"scene {sid}, L_{i+1}: lime box y={y} x={x} {sz}x{sz} holds the densest 'consistent but wrong' "
                         f"pixels (FB-consistent, valid, m(L_2)>{REF_CLIP}, err>{ERR_LARGE}): {100*dens:.1f}% of the box; "
                         f"{nbad} such pixels in the frame")
    fig.tight_layout()
    fig.savefig(fp, dpi=70)
    fig.savefig(rd / "figures" / fp.name, dpi=70)
    plt.close(fig)
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", nargs="*", default=None, help="2 val scenes for 6a (ghost scene, most saturated)")
    args = ap.parse_args()
    split = load_split()
    val = split["val"]
    if args.scenes:
        fig_scenes = args.scenes
    else:
        ghost = json.loads((REPO / "results" / "tables" / "ghost_crop.json").read_text())["scene"]
        inv = pd.read_csv(REPO / "results" / "tables" / "dataset_inventory.csv", dtype={"scene": str})
        sat = inv[inv.scene.isin(val) & (inv.scene != ghost)].sort_values("pct_m2_gt_095").scene.iloc[-1]
        fig_scenes = [ghost, sat]
    rd = make_run_dir(args.out, {"q_min": Q_MIN, "err_large": ERR_LARGE, "ref_clip": REF_CLIP, "fig_scenes": fig_scenes,
                                 "val": val})
    rows = []
    for sid in val:
        rows += stats_rows(sid)
    allrows = rows + pooled(rows)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in allrows])
    df.to_csv(rd / "metrics.csv", index=False)
    tab = REPO / "results" / "tables" / "cue_diagnostics_val.csv"
    with tab.open("w") as fh:
        fh.write(f"# 5 validation scenes ({', '.join(val)}); pixels with v_i=1 and q_i^w>{Q_MIN}; err = mean_c |T(H_i^w)-T(H_gt)| "
                 f"(mu=5000); large error: err>{ERR_LARGE}; consistent = FB check passed; AUROC of (1-r_i) for large error; "
                 f"strata by m(L_2)>{REF_CLIP}; source run {rd.relative_to(REPO)}\n")
        df.to_csv(fh, index=False)
    fp = REPO / "results" / "figures" / "fig_reliability_maps.png"
    notes = fig_maps(fig_scenes, fp, rd)
    write_sidecar(fp, [
        f"Validation scenes {fig_scenes[0]} (ghost scene of Exp 3) and {fig_scenes[1]} (most saturated val scene other than it); full frames.",
        "One row per non-reference frame. These are ESTIMATED reliability cues (DIS flow + forward-backward check, exposure cues), not ground-truth motion.",
        "Panels: exposure-matched pair as a red/cyan composite (8-bit gray, used only for flow); warped L_i^w; validity v_i; FB-inconsistency mask; q_ref; q_i^w; mu-law error of the warped frame vs H_gt.",
        "Cue failure example (consistent but wrong): " + " | ".join(notes),
        f"Display: LDR panels {LDR_MAPPING}; masks/cues gray 0..1; error inferno 0..{ERR_VMAX}. Not a learned result.",
        f"Source run: {rd.relative_to(REPO)}",
    ])
    pp = df[df.scene == "pooled"]
    write_notes(rd, [
        f"6b on {len(val)} val scenes; pixels v=1 & q^w>{Q_MIN}; large err > {ERR_LARGE}",
        *[f"{r.frame} {r.stratum}: n={r.n_pix}, incons {r.frac_inconsistent:.3f}, err cons {r.mean_err_consistent:.4f} vs incons "
          f"{r.mean_err_inconsistent:.4f}, AUROC(1-r) {r.auroc_1_minus_r:.3f}" for r in pp.itertuples()],
        *notes,
    ])
    print(pp.to_string())


if __name__ == "__main__":
    main()
