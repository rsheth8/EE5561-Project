import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hdr.alignment import backward_warp, dis_flow, fb_check, to_gray8, validity  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DX, DY = 5, -3


def textured(h=240, w=320, seed=0):
    import cv2

    rng = np.random.default_rng(seed)
    base = rng.random((h // 4, w // 4, 3)).astype(np.float32)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    img += 0.15 * cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 1.0)
    return np.clip(img / img.max(), 0, 1).astype(np.float32)


def shifted_pair():
    """Reference ref(x) = src(x + (DX, DY)): the source content sits +DX right, DY up."""
    big = textured(260, 340)
    pad = 10
    src = big[pad:pad + 240, pad:pad + 320]
    ref = big[pad + DY:pad + DY + 240, pad + DX:pad + DX + 320]
    # make the source carry zeros where the reference samples outside it is not needed;
    # out-of-bounds is decided purely by the map.
    return ref, src


def test_zero_flow_identity():
    img = textured()
    out = backward_warp(img, np.zeros(img.shape[:2] + (2,), np.float32))
    assert np.array_equal(out, img)
    assert validity(np.zeros(img.shape[:2] + (2,), np.float32)).all()


def test_translation():
    ref, src = shifted_pair()
    f = dis_flow(to_gray8(ref), to_gray8(src))
    med = np.median(f[20:-20, 20:-20].reshape(-1, 2), axis=0)
    assert abs(med[0] - DX) < 0.25 and abs(med[1] - DY) < 0.25, med
    w = backward_warp(src, f)
    inner = (slice(20, -20), slice(20, -20))
    assert np.abs(w[inner] - ref[inner]).mean() < 0.01
    v = validity(f)
    h, wd = v.shape
    # right DX columns and top |DY| rows must be invalid (allow 2 px slack)
    assert (v[10:-10, wd - DX + 2:] == 0).all()
    assert (v[: abs(DY) - 2 if abs(DY) > 2 else 1, 10:-10] == 0).all()
    # interior valid
    assert v[abs(DY) + 2: h - 2, 2: wd - DX - 2].mean() > 0.99
    # save the figure for the report
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from hdr.display import ldr_to_uint8, write_sidecar

        fig, ax = plt.subplots(1, 5, figsize=(16, 3))
        panels = [(ldr_to_uint8(ref), "reference"), (ldr_to_uint8(src), "source"),
                  (ldr_to_uint8(w), "backward-warped source"),
                  (np.abs(w - ref).mean(-1), "|warped - ref| (0..0.2)"), (v, "validity v")]
        for a, (im, t) in zip(ax, panels):
            a.imshow(im, cmap="gray", vmin=0, vmax=0.2 if "ref|" in t else None)
            a.set_title(t, fontsize=9)
            a.axis("off")
        fig.suptitle(f"synthetic translation (+{DX}, {DY}) px; DIS median flow ({med[0]:.3f}, {med[1]:.3f})", fontsize=9)
        fig.tight_layout()
        fp = REPO / "results" / "figures" / "fig_warp_check.png"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fp, dpi=100)
        plt.close(fig)
        write_sidecar(fp, [
            f"Synthetic textured image (seeded random texture), 240x320, shifted by (+{DX}, {DY}) px. Not dataset data.",
            "Method: DIS optical flow (PRESET_MEDIUM) ref->src, backward warp cv2.remap INTER_LINEAR, BORDER_CONSTANT 0.",
            f"Median DIS flow in interior: ({med[0]:.4f}, {med[1]:.4f}); mean |warp-ref| interior {np.abs(w[inner]-ref[inner]).mean():.5f}.",
            "Validity v = 1 where the sampling map lies inside the source; v=0 strip: right 5 columns, top 3 rows.",
            "Display: images shown as stored values x255; error map shown with a 0..0.2 gray scale. Produced by tests/test_warp.py.",
        ])
        (REPO / "results" / "tables").mkdir(parents=True, exist_ok=True)
        (REPO / "results" / "tables" / "warp_check.json").write_text(json.dumps(
            {"median_flow": med.tolist(), "target": [DX, DY],
             "interior_mae": float(np.abs(w[inner] - ref[inner]).mean())}, indent=1))
    except Exception as exc:  # figure is a side product; the test result is the asserts
        print("figure failed:", exc)


def test_fb_identity():
    img = textured()
    g = to_gray8(img)
    ff = dis_flow(g, g)
    fbk = dis_flow(g, g)
    r = fb_check(ff, fbk)
    assert np.median(r["e"]) < 1e-3
    assert r["incons"].mean() < 0.01
