import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hdr.model import AHDRNet, build_model, count_params  # noqa: E402

B, H, W = 2, 48, 64


def inputs(seed=0, zero_nonref=False):
    g = torch.Generator().manual_seed(seed)
    feat = torch.rand(B, 3, 6, H, W, generator=g)
    featw = torch.rand(B, 3, 6, H, W, generator=g)
    featw[:, 1] = feat[:, 1]
    cues = torch.rand(B, 2, 4, H, W, generator=g)
    valid = (torch.rand(B, 2, 1, H, W, generator=g) > 0.3).float()
    valid[:, :, :, :, :5] = 0  # an invalid strip
    cues[:, :, 2:3] = valid
    if zero_nonref:
        feat[:, [0, 2]] = 0
        featw[:, [0, 2]] = 0
    return feat, featw, cues, valid


def test_m0_shapes_and_params():
    m = AHDRNet()
    y = m(torch.rand(B, 3, 6, H, W))
    assert y.shape == (B, 3, H, W) and (y >= 0).all() and (y <= 1).all()
    assert 1.3e6 < count_params(m) < 1.6e6


def test_gate_shapes_zero_where_invalid_and_init():
    torch.manual_seed(0)
    for var in ("M2", "M3"):
        m = build_model(var)
        feat, featw, cues, valid = inputs()
        h, (g1, g3) = m(feat, featw, cues, valid)
        assert h.shape == (B, 3, H, W)
        assert g1.shape == (B, 1, H, W) and g3.shape == (B, 1, H, W)
        assert (g1[valid[:, 0] == 0] == 0).all() and (g3[valid[:, 1] == 0] == 0).all()
        assert torch.allclose(g1[valid[:, 0] == 1], torch.tensor(0.5), atol=1e-6)
        assert torch.allclose(g3[valid[:, 1] == 1], torch.tensor(0.5), atol=1e-6)


def test_gate_zero_after_training_step_where_invalid():
    m = build_model("M3")
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    feat, featw, cues, valid = inputs()
    for _ in range(2):
        h, _ = m(feat, featw, cues, valid)
        loss = h.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    _, (g1, g3) = m(feat, featw, cues, valid)
    assert (g1[valid[:, 0] == 0] == 0).all() and (g3[valid[:, 1] == 0] == 0).all()


def test_m2_m3_same_params():
    a, b = build_model("M2"), build_model("M3")
    assert count_params(a) == count_params(b)
    assert [n for n, _ in a.named_parameters()] == [n for n, _ in b.named_parameters()]
    assert [p.shape for p in a.parameters()] == [p.shape for p in b.parameters()]


def test_m2_ignores_cue_values():
    m = build_model("M2")
    feat, featw, cues, valid = inputs()
    h1, _ = m(feat, featw, cues, valid)
    cues2 = torch.rand_like(cues)
    cues2[:, :, 2:3] = valid
    h2, _ = m(feat, featw, cues2, valid)
    assert torch.equal(h1, h2)


def test_finite_with_zero_nonref():
    for var in ("M2", "M3"):
        m = build_model(var)
        feat, featw, cues, valid = inputs(zero_nonref=True)
        h, (g1, g3) = m(feat, featw, cues, valid)
        assert torch.isfinite(h).all() and torch.isfinite(g1).all() and torch.isfinite(g3).all()
