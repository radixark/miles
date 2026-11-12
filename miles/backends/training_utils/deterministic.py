"""What ``--deterministic-mode`` turns on in torch, shared by the training backends."""

import os

import torch


def enable_deterministic_algorithms() -> None:
    """Have torch run its deterministic kernels, and raise on an op that has none.

    This governs torch-native ops only. A custom kernel (flash-attn, Transformer Engine) is opaque to
    it, so each backend also configures the kernels it brings.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    # deterministic mode otherwise fills every torch.empty, which only catches reads of
    # uninitialized memory and costs ~5% of forward kernel time
    torch.utils.deterministic.fill_uninitialized_memory = os.environ.get("MILES_FILL_UNINIT", "0") == "1"
