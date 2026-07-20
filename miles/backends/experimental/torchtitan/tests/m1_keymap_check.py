"""M1 gate (GPU-free): titan spec built from HF config must map 1:1 onto the checkpoint.

Meta-builds the model (no weights, no GPU), runs the state-dict adapter's to_hf, and
compares the resulting HF key set against the checkpoint's model.safetensors.index.json.
Any missing/extra key means the HF-config->titan mapper or adapter wiring is wrong.

Usage: python3 -m miles.backends.experimental.torchtitan.tests.m1_keymap_check <hf_ckpt_dir>
"""

import json
import os
import sys

import torch

from miles.backends.experimental.torchtitan import titan_bridge


def main(hf_dir: str) -> int:
    spec = titan_bridge.spec_from_hf(hf_dir, seq_len=4096)

    from torchtitan.config import ParallelismConfig

    spec.model.update_from_config(config=titan_bridge._RuntimeConfig(ParallelismConfig()))
    with torch.device("meta"):
        model = spec.model.build()

    adapter = titan_bridge.make_adapter(spec, hf_dir)
    hf_sd = adapter.to_hf(model.state_dict())
    ours = set(hf_sd.keys())

    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            ckpt = set(json.load(f)["weight_map"].keys())
    else:  # single-shard checkpoints have no index
        from safetensors import safe_open

        with safe_open(os.path.join(hf_dir, "model.safetensors"), framework="pt") as f:
            ckpt = set(f.keys())

    missing = sorted(ckpt - ours)
    extra = sorted(ours - ckpt)
    print(f"model: {hf_dir}")
    print(f"titan->hf keys: {len(ours)} | checkpoint keys: {len(ckpt)}")
    for k in missing[:20]:
        print(f"  MISSING (in ckpt, not produced): {k}")
    for k in extra[:20]:
        print(f"  EXTRA   (produced, not in ckpt): {k}")
    if missing or extra:
        print("M1 KEYMAP: FAIL")
        return 1
    print("M1 KEYMAP: PASS (exact match)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
