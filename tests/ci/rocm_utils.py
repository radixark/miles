"""Shared ROCm detection for CI tests."""

import torch

IS_ROCM = getattr(torch.version, "hip", None) is not None
