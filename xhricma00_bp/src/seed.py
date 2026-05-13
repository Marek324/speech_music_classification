# seed.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.

RAND_SEED = 381


def seed_all() -> None:
    """Fix RNGs for ``datasets.shuffle``, numpy, torch (incl. CUDA if present)."""
    import random

    import numpy as np
    import torch

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RAND_SEED)
