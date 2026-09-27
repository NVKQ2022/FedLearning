"""
Reproducibility and Deterministic Seed Management for FL-IoT-IDS.

Adheres to:
- skills/experiment-orchestration/SKILL.md (Step 3: Deterministic Reproducibility Protocol)
- skills/methodology-audit/SKILL.md (Pillar 2 & 5: Reproducibility and Scientific Integrity)
"""

import os
import random
import logging
from typing import Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Enforces bit-level deterministic reproducibility across Python, NumPy,
    and PyTorch (CPU & CUDA).

    Args:
        seed: Integer random seed (default: 42).
    """
    logger.info(f"Setting global deterministic random seed: {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def set_deterministic_seed(seed: int = 42) -> None:
    """
    Alias for set_seed adhering to experiment-orchestration skill naming.
    """
    set_seed(seed)


if __name__ == "__main__":
    print("--- Testing Deterministic Seed Reproducibility ---")
    set_seed(42)
    rand_a = np.random.rand(3)
    set_seed(42)
    rand_b = np.random.rand(3)
    assert np.allclose(rand_a, rand_b), "Random numbers must be identical across identical seeds!"
    print(f"Seed test passed: {rand_a} == {rand_b}")
