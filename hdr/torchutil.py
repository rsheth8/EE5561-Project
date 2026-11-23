from __future__ import annotations

import random

import numpy as np
import torch


def setup_torch(seed: int, deterministic: bool = False) -> torch.device:
    """fp32 everywhere: TF32 is disabled for matmul and cuDNN (plan 6.5 says fp32)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
