import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int = 42, *, strict: bool = False) -> None:
    """
    Set seeds across random, numpy, and torch (CPU and CUDA) for reproducibility.

    When ``strict=True``, also disable TF32 and opt into deterministic algorithms
    so cross-machine reruns do not silently change math mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if strict:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
