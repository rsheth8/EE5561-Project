"""IO pairing + cache-vs-raw feature equality + metric sanity."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from hdr import GAMMA  # noqa: E402
from hdr.data import load_scene  # noqa: E402
from hdr.io import load_scene_raw  # noqa: E402
from hdr.metrics import all_metrics, mu_law  # noqa: E402

RAW = REPO / "raw" / "Training"
need_data = pytest.mark.skipif(not (RAW / "001").exists() or not (REPO / "cache" / "001.npz").exists(), reason="data")


@need_data
def test_pairing_scene001():
    s = load_scene_raw(RAW / "001")
    assert s["tif_names"] == sorted(s["tif_names"])
    assert s["e"].tolist() == [0, 2, 4] and s["t"].tolist() == [1, 4, 16]
    assert s["pairing_ok"] and np.all(np.diff(s["pairing_means"]) > 0)
    assert s["dtype"] == "uint16"


@need_data
@pytest.mark.parametrize("sid", ["001", "036"])
def test_cache_features_equal_raw(sid):
    d = load_scene(sid)
    s = load_scene_raw(RAW / sid)
    ldr = torch.from_numpy(s["ldr_raw"].astype(np.float32) / 65535.0).permute(0, 3, 1, 2)
    t = torch.tensor(s["t"], dtype=torch.float32)
    # same memory layout as the loader -> bit-identical
    rad = ldr.contiguous().pow(GAMMA) / t.view(3, 1, 1, 1)
    feat = torch.cat([ldr, rad], 1)
    assert d["features"].shape == (3, 6, 1000, 1500)
    assert torch.equal(d["features"], feat)
    # non-contiguous pow differs by at most float32 rounding (1 ulp)
    rad_nc = ldr.pow(GAMMA) / t.view(3, 1, 1, 1)
    assert (d["radiance"] - rad_nc).abs().max() < 1e-6
    assert torch.equal(d["target"], torch.from_numpy(s["hdr"]).permute(2, 0, 1))
    assert d["ldr"].dtype == torch.float32 and d["exposure"].tolist() == s["t"].tolist()


def test_metrics_basic():
    rng = np.random.default_rng(0)
    gt = rng.random((32, 48, 3)).astype(np.float32) * 0.5
    m = all_metrics(gt, gt)
    assert m["psnr_l"] == float("inf") and m["ssim_mu"] == pytest.approx(1.0)
    noisy = gt + 0.01
    m = all_metrics(noisy, gt)
    assert m["psnr_l"] == pytest.approx(40.0, abs=1e-3)
    assert mu_law(np.array([0.0, 1.0, 2.0])).tolist() == pytest.approx([0.0, 1.0, 1.0])
    # metrics never modify inputs
    g2 = gt.copy()
    all_metrics(noisy, gt)
    assert np.array_equal(g2, gt)


def test_split_frozen():
    p = REPO / "configs" / "split.json"
    if not p.exists():
        pytest.skip("split")
    s = json.loads(p.read_text())
    assert len(s["val"]) == 5 and len(s["train"]) + len(s["val"]) == 74
    assert not set(s["val"]) & set(s["train"])
    for g in s["groups"]:
        assert set(g) <= set(s["train"]) or set(g) <= set(s["val"])
