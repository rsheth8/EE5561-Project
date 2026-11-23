"""results/figures/fig_saturation_mask.png: L_2 with over-/under-exposure (clipping) masks."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hdr.data import load_scene_np  # noqa: E402
from hdr.display import LDR_MAPPING, ldr_to_uint8, write_sidecar  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SCENES = sys.argv[1:] or ["027", "036"]
fig, ax = plt.subplots(len(SCENES), 2, figsize=(11, 3.7 * len(SCENES)), squeeze=False)
lines = []
for r, sid in enumerate(SCENES):
    l2 = load_scene_np(sid)["ldr_hwc"][1]
    m = l2.max(-1)
    over, under = m > 0.95, m < 0.05
    ov = ldr_to_uint8(l2).astype(np.float32)
    ov[over] = 0.45 * ov[over] + 0.55 * np.array([255, 0, 255])
    ov[under] = 0.45 * ov[under] + 0.55 * np.array([0, 200, 255])
    ax[r, 0].imshow(ldr_to_uint8(l2))
    ax[r, 0].set_title(f"scene {sid}: L_2", fontsize=9)
    ax[r, 1].imshow(ov.astype(np.uint8))
    ax[r, 1].set_title(f"clipping mask: magenta m(L_2)>0.95 ({100*over.mean():.1f}%), cyan m(L_2)<0.05 ({100*under.mean():.1f}%)", fontsize=8)
    lines.append(f"scene {sid}: over-exposed {100*over.mean():.2f}%, under-exposed {100*under.mean():.2f}% of pixels")
for a in ax.ravel():
    a.axis("off")
fig.tight_layout()
fp = REPO / "results" / "figures" / "fig_saturation_mask.png"
fig.savefig(fp, dpi=100)
write_sidecar(fp, [
    f"Scenes {', '.join(SCENES)} (training split), full frames, no crop.",
    "This is a CLIPPING mask computed from the reference exposure alone: m(L_2) = max over RGB; over-exposed m>0.95 (magenta), under-exposed m<0.05 (cyan). It is NOT a motion mask.",
    *lines,
    "Display: " + LDR_MAPPING + "; overlay alpha 0.55. Data inspection only, no method/training. Produced by scripts/fig_saturation.py.",
])
print("\n".join(lines))
