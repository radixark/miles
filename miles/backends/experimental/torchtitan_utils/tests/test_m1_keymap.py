"""M1 gate, GPU-free: the titan spec built from an HF config must map 1:1 onto the
checkpoint's HF weight names. Meta-builds the model (no weights, no GPU), runs the
state-dict adapter's to_hf, and diffs the resulting key set against
model.safetensors.index.json. Any missing/extra key means the mapper or adapter wiring
is wrong.

Usage: python3 -m miles.backends.experimental.torchtitan_utils.tests.test_m1_keymap <hf_ckpt_dir>
"""

import json
import os
import sys

import torch

from miles.backends.experimental.torchtitan_utils import models


def run(hf_dir: str) -> int:
    spec, hf = models.spec_from_hf(hf_dir)

    from torchtitan.models.qwen3.sharding import set_qwen3_sharding_config

    set_qwen3_sharding_config(spec.model, enable_sp=False, enable_ep=False)

    with torch.device("meta"):
        model = spec.model.build()

    adapter = spec.state_dict_adapter(spec.model, hf_dir)
    hf_sd = adapter.to_hf(model.state_dict())
    ours = set(hf_sd.keys())

    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            ckpt = set(json.load(f)["weight_map"].keys())
    else:
        from safetensors import safe_open

        with safe_open(os.path.join(hf_dir, "model.safetensors"), framework="pt") as f:
            ckpt = set(f.keys())

    missing = sorted(ckpt - ours)
    extra = sorted(ours - ckpt)
    print(f"model: {hf_dir}  |  titan->hf keys: {len(ours)}  |  checkpoint keys: {len(ckpt)}")
    for k in missing[:20]:
        print(f"  MISSING (in ckpt, not produced): {k}")
    for k in extra[:20]:
        print(f"  EXTRA   (produced, not in ckpt): {k}")

    fused_leak = [
        name
        for name, p in model.named_parameters()
        if "wqkv" in name  # FusedQKVLinear's param name; QKVLinear uses wq/wk/wv
    ]
    if fused_leak:
        print(f"  FUSED-QKV LEAK: {fused_leak[:5]}")

    ok = not missing and not extra and not fused_leak
    print("M1 KEYMAP:", "PASS (exact match)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
