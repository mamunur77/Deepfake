"""
src.utils.seed  ·  Reproducibility helpers.
===========================================
Sets seeds for Python, NumPy, and PyTorch (CPU + CUDA).
Call once at the start of every script / notebook.

Usage:
    from src.utils.seed import seed_everything
    seed_everything(42)
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Fix all random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to set everywhere.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[SEED] All random seeds set to {seed}")
