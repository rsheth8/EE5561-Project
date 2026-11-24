"""M0 / M1: AHDRNet-style network (Yan et al., CVPR 2019, arXiv:1904.10293).

Written from scratch for this repo following the paper's description and the layer
layout of the authors' public release (github.com/qingsenyangit/AHDRNet, model.py).
No code was copied. Differences from that release:
  * attention: Conv3x3(128->128) -> LeakyReLU(0.2) -> Conv3x3(128->64) -> sigmoid, as in
    the release and EXPERIMENT_PLAN.md 6.1;
  * encoder activation LeakyReLU(0.2) (plan 6.1); DRDB dense layers use ReLU (release);
  * inputs X_i = concat(L_i, L_i**2.2 / t_i) with t_i = 2**e_i (relative times);
  * final sigmoid output; loss L1 on mu-law(clip(H,0,1)), mu = 5000.
M1 is the same module fed with the flow-aligned non-reference inputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, cin: int, growth: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, growth, 3, padding=dilation, dilation=dilation)

    def forward(self, x):
        return torch.cat([x, F.relu(self.conv(x))], dim=1)


class DRDB(nn.Module):
    """Dilated residual dense block: 6 dense layers (growth 32, dilation 2),
    local 1x1 fusion and a local residual."""

    def __init__(self, ch: int = 64, n_layers: int = 6, growth: int = 32, dilation: int = 2):
        super().__init__()
        layers, c = [], ch
        for _ in range(n_layers):
            layers.append(DenseLayer(c, growth, dilation))
            c += growth
        self.dense = nn.Sequential(*layers)
        self.fuse = nn.Conv2d(c, ch, 1)

    def forward(self, x):
        return x + self.fuse(self.dense(x))


class Attention(nn.Module):
    """A_i = sigmoid(Conv(128->64)(LeakyReLU(Conv(128->128)(cat(Z_i, Z_2))))), a 64xHxW mask."""

    def __init__(self, ch: int = 64):
        super().__init__()
        self.c1 = nn.Conv2d(2 * ch, 2 * ch, 3, padding=1)
        self.c2 = nn.Conv2d(2 * ch, ch, 3, padding=1)

    def forward(self, zi, z2):
        return torch.sigmoid(self.c2(F.leaky_relu(self.c1(torch.cat([zi, z2], 1)), 0.2)))


class Encoder(nn.Module):
    def __init__(self, cin: int = 6, ch: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(cin, ch, 3, padding=1)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), 0.2)


class MergeNet(nn.Module):
    """Attention + merge part, shared by M0-M3: takes Z_1, Z_2, Z_3 (each [B,64,H,W])."""

    def __init__(self, ch: int = 64, n_drdb: int = 3, n_layers: int = 6, growth: int = 32, dilation: int = 2):
        super().__init__()
        self.att1 = Attention(ch)
        self.att3 = Attention(ch)
        self.f0 = nn.Conv2d(3 * ch, ch, 3, padding=1)
        self.drdbs = nn.ModuleList([DRDB(ch, n_layers, growth, dilation) for _ in range(n_drdb)])
        self.gff1 = nn.Conv2d(n_drdb * ch, ch, 1)
        self.gff3 = nn.Conv2d(ch, ch, 3, padding=1)
        self.up = nn.Conv2d(ch, ch, 3, padding=1)
        self.out = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, z1, z2, z3, return_att: bool = False):
        a1 = self.att1(z1, z2)
        a3 = self.att3(z3, z2)
        f = self.f0(torch.cat([a1 * z1, z2, a3 * z3], 1))
        outs = []
        for blk in self.drdbs:
            f = blk(f)
            outs.append(f)
        f = self.gff3(self.gff1(torch.cat(outs, 1))) + z2
        h = torch.sigmoid(self.out(self.up(f)))
        return (h, (a1, a3)) if return_att else h


class AHDRNet(nn.Module):
    """M0 (original inputs) / M1 (aligned inputs). Input x: [B, 3, 6, H, W]."""

    def __init__(self, **kw):
        super().__init__()
        self.enc = Encoder(6, kw.get("ch", 64))
        self.merge = MergeNet(**kw)

    def forward(self, x):
        z1, z2, z3 = (self.enc(x[:, i]) for i in range(3))
        return self.merge(z1, z2, z3)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def receptive_field(n_drdb: int = 3, n_layers: int = 6, dilation: int = 2) -> dict:
    """Theoretical RF (pixels, one side = (RF-1)/2) along the deepest path input->output
    for the non-reference branch: enc 3x3, attention 2x 3x3, F0 3x3, n_drdb*n_layers
    dilated 3x3 (each adds 2*dilation), 1x1, gff 3x3, up 3x3, out 3x3."""
    k3 = 2  # a 3x3 conv with dilation 1 adds 2 px
    rf = 1 + k3 * (1 + 2 + 1) + n_drdb * n_layers * 2 * dilation + k3 * 3
    rf_ref = 1 + k3 * (1 + 1) + n_drdb * n_layers * 2 * dilation + k3 * 3  # reference branch (no attention)
    rf_nodil = 1 + k3 * (1 + 2 + 1) + n_drdb * n_layers * 2 * 1 + k3 * 3
    return {"rf_nonref_branch": rf, "rf_ref_branch": rf_ref, "rf_if_dilation_1": rf_nodil}


def build_model(variant: str) -> nn.Module:
    variant = variant.upper()
    if variant in ("M0", "M1"):
        return AHDRNet()
    if variant in ("M2", "M3"):
        from .gate import GatedAHDRNet

        return GatedAHDRNet(constant_cues=(variant == "M2"))
    raise ValueError(variant)


def model_forward(model: nn.Module, variant: str, batch: dict):
    """Returns (H_pred, extras). Picks inputs per variant."""
    variant = variant.upper()
    if variant == "M0":
        return model(batch["feat"]), None
    if variant == "M1":
        return model(batch["featw"]), None
    return model(batch["feat"], batch["featw"], batch["cues"], batch["valid"])
