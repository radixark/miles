"""mbridge's FP8 dequant loader opens shards with `device="cuda"`, which safetensors maps to cuda:0 on every rank."""

import torch
from safetensors import safe_open

from mbridge.models.ext.deepseek_v3 import dequant_fp8_safetensor_io


def _safe_open_on_current_device(filename, framework, device="cpu"):
    if device == "cuda":
        device = f"cuda:{torch.cuda.current_device()}"
    return safe_open(filename, framework=framework, device=device)


# TODO: drop once mbridge opens the shards on the current device
dequant_fp8_safetensor_io.safe_open = _safe_open_on_current_device
