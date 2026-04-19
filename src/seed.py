# Keep this value in sync with scripts/dataset/source_config.py — they live in
# separate uv subprojects so direct import isn't possible.
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
