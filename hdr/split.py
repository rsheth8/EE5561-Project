"""Train/val split with a related-scene check (EXPERIMENT_PLAN.md 2.5).

python -m hdr.split  -> configs/split.json, results/figures/related_scenes.png
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import SEED  # noqa: E402
from .data import CACHE, load_scene_np  # noqa: E402
from .display import LDR_MAPPING, ldr_to_uint8, write_sidecar  # noqa: E402
from .runlog import make_run_dir, write_notes  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NCC_THRESHOLD = 0.9
N_VAL = 5
THUMB_HW = (48, 72)


def thumb(l2: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(l2.astype(np.float32), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (THUMB_HW[1], THUMB_HW[0]), interpolation=cv2.INTER_AREA).astype(np.float64)
    return (g - g.mean()) / (g.std() + 1e-12)


def union_find_groups(n: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def main():
    ids = sorted(p.stem for p in CACHE.glob("*.npz"))
    assert len(ids) == 74, f"expected 74 cached scenes, found {len(ids)}"
    rd = make_run_dir(REPO / "runs" / "split", {"seed": SEED, "ncc_threshold": NCC_THRESHOLD, "n_val": N_VAL,
                                                 "thumb_hw": THUMB_HW})
    l2s = {i: load_scene_np(i)["ldr_hwc"][1] for i in ids}
    th = np.stack([thumb(l2s[i]).ravel() for i in ids])
    ncc = th @ th.T / th.shape[1]
    n = len(ids)
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n) if ncc[a, b] > NCC_THRESHOLD]
    groups = union_find_groups(n, pairs)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(groups))
    shuffled = [groups[k] for k in order]
    # prefer singletons, keep shuffled order within each class; groups never split
    cand = [g for g in shuffled if len(g) == 1] + [g for g in shuffled if len(g) > 1]
    val: list[int] = []
    for g in cand:
        if len(val) + len(g) <= N_VAL:
            val += g
        if len(val) == N_VAL:
            break
    assert len(val) == N_VAL
    val_ids = sorted(ids[i] for i in val)
    train_ids = sorted(i for i in ids if i not in val_ids)
    iu = np.triu_indices(n, 1)
    top = np.argsort(-ncc[iu])[:10]
    top_pairs = [[ids[iu[0][k]], ids[iu[1][k]], float(ncc[iu][k])] for k in top]
    split = dict(
        train=train_ids,
        val=val_ids,
        seed=SEED,
        ncc_threshold=NCC_THRESHOLD,
        thumb_hw=list(THUMB_HW),
        flagged_pairs=[[ids[a], ids[b], float(ncc[a, b])] for a, b in pairs],
        groups=[[ids[i] for i in g] for g in groups if len(g) > 1],
        top10_ncc_pairs=top_pairs,
    )
    (REPO / "configs" / "split.json").write_text(json.dumps(split, indent=2))

    # contact sheet
    fig_path = REPO / "results" / "figures" / "related_scenes.png"
    show = split["flagged_pairs"] if pairs else top_pairs[:4]
    rows = max(1, len(show))
    fig, axes = plt.subplots(rows, 2, figsize=(6, 2.2 * rows), squeeze=False)
    for r, (a, b, v) in enumerate(show):
        for c, sid in enumerate((a, b)):
            axes[r, c].imshow(ldr_to_uint8(l2s[sid]))
            axes[r, c].set_title(f"scene {sid}  (NCC {v:.3f})", fontsize=8)
            axes[r, c].axis("off")
    title = (f"{len(pairs)} pairs with NCC > {NCC_THRESHOLD}" if pairs else
             f"no pair exceeds NCC {NCC_THRESHOLD}; showing top-{len(show)} NCC pairs for reference")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=110)
    fig.savefig(rd / "figures" / fig_path.name, dpi=110)
    plt.close(fig)
    write_sidecar(fig_path, [
        f"Related-scene check: middle exposure L_2, {THUMB_HW[0]}x{THUMB_HW[1]} grayscale thumbnails, z-normalised; NCC threshold {NCC_THRESHOLD}.",
        title + ".",
        "Pairs shown: " + "; ".join(f"{a}-{b} NCC={v:.3f}" for a, b, v in show),
        "Groups (never split): " + (json.dumps(split["groups"]) if split["groups"] else "none (all singletons)"),
        "Display mapping: " + LDR_MAPPING + ". Full frames, no crop. Not a training result.",
        f"Source run: {rd.relative_to(REPO)}",
    ])
    (rd / "metrics.json").write_text(json.dumps(split, indent=2))
    write_notes(rd, [
        f"{len(pairs)} pairs flagged with NCC > {NCC_THRESHOLD}; {len(split['groups'])} multi-scene groups",
        f"top NCC pair {top_pairs[0]}",
        f"val scenes (seed {SEED}): {val_ids}",
        f"{len(train_ids)} train scenes",
        "written configs/split.json (frozen) and results/figures/related_scenes.png",
    ])
    print(json.dumps({k: split[k] for k in ("val", "groups", "top10_ncc_pairs")}, indent=1))


if __name__ == "__main__":
    main()
