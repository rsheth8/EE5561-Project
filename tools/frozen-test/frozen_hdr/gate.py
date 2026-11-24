"""M2 / M3: gated fusion of original and flow-aligned features (EXPERIMENT_PLAN.md 6.3).

For i in {1, 3}, shared encoder E:
    Z_i^o = E(X_i), Z_i^w = E(X_i^w)
    c_i   = [q_ref, q_i^w, v_i, r_i]  (M3)   |  constant 1.0, 4 channels  (M2)
    g_i   = v_i * sigmoid(Conv3x3(64->1)(LeakyReLU(Conv3x3(196->64)(cat(Z_2, Z_i^o, Z_i^w, c_i)))))
    Z_i   = g_i * Z_i^w + (1 - g_i) * Z_i^o
then the unchanged M0 attention + merge.

Design choices (ours): one spatial gate channel (interpretable, unlike AHDRNet's
CxHxW masks); separate gate parameters for i=1 and i=3 (mirrors AHDRNet's separate
attention modules); last gate conv initialised to weight 0, bias 0 so g = 0.5*v at
start. M2 and M3 have identical parameters; only the cue channels differ.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Encoder, MergeNet


class Gate(nn.Module):
    def __init__(self, ch: int = 64, n_cues: int = 4):
        super().__init__()
        self.c1 = nn.Conv2d(3 * ch + n_cues, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, 1, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, z2, zo, zw, c, v):
        h = F.leaky_relu(self.c1(torch.cat([z2, zo, zw, c], 1)), 0.2)
        return v * torch.sigmoid(self.c2(h))


class GatedAHDRNet(nn.Module):
    def __init__(self, constant_cues: bool, ch: int = 64, **kw):
        super().__init__()
        self.constant_cues = constant_cues
        self.enc = Encoder(6, ch)
        self.gate1 = Gate(ch)
        self.gate3 = Gate(ch)
        self.merge = MergeNet(ch=ch, **kw)

    def forward(self, feat, featw, cues, valid):
        """feat, featw: [B,3,6,H,W] (featw[:,1] is the reference); cues [B,2,4,H,W];
        valid [B,2,1,H,W]. Returns H_pred and (g_1, g_3), each [B,1,H,W]."""
        z2 = self.enc(feat[:, 1])
        out_z, gates = [], []
        for k, (i, gate) in enumerate(((0, self.gate1), (2, self.gate3))):
            zo = self.enc(feat[:, i])
            zw = self.enc(featw[:, i])
            c = torch.ones_like(cues[:, k]) if self.constant_cues else cues[:, k]
            g = gate(z2, zo, zw, c, valid[:, k])
            out_z.append(g * zw + (1 - g) * zo)
            gates.append(g)
        h = self.merge(out_z[0], z2, out_z[1])
        return h, (gates[0], gates[1])
