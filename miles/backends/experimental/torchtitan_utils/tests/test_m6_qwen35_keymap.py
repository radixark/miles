"""M6 gate, GPU-free: same idea as test_m1_keymap.py but for qwen3_5 — meta-builds the
model (with vision_encoder pruned, matching the real text-only RL build path in
model.py), runs the state-dict adapter's to_hf, and diffs the resulting key set against
model.safetensors.index.json. Two prefixes are EXPECTED to be missing by design:
"model.visual.*" (vision tower, pruned — see build_and_load_model) and "mtp.*"
(multi-token-prediction head, not built by torchtitan's qwen3_5 model at all). Any other
missing/extra key means the mapper or adapter wiring is wrong.

Usage: python3 -m miles.backends.experimental.torchtitan_utils.tests.test_m6_qwen35_keymap <hf_ckpt_dir>
"""

import json
import os
import sys
from types import SimpleNamespace

import torch

from miles.backends.experimental.torchtitan_utils import models

EXPECTED_MISSING_PREFIXES = ("model.visual.", "mtp.")


def run(hf_dir: str) -> int:
    spec, hf = models.spec_from_hf(hf_dir)

    from torchtitan.config import ParallelismConfig

    spec.model.update_from_config(config=SimpleNamespace(parallelism=ParallelismConfig()))

    with torch.device("meta"):
        model = spec.model.build()

    assert model.vision_encoder is not None, "sanity: model should build a real vision_encoder before pruning"
    model.vision_encoder = None

    adapter = spec.state_dict_adapter(spec.model, hf_dir)
    hf_sd = adapter.to_hf(model.state_dict())
    ours = set(hf_sd.keys())

    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        ckpt = set(json.load(f)["weight_map"].keys())

    missing = sorted(ckpt - ours)
    extra = sorted(ours - ckpt)
    expected_missing = [k for k in missing if k.startswith(EXPECTED_MISSING_PREFIXES)]
    unexpected_missing = [k for k in missing if not k.startswith(EXPECTED_MISSING_PREFIXES)]

    print(f"model: {hf_dir}  |  titan->hf keys: {len(ours)}  |  checkpoint keys: {len(ckpt)}")
    print(f"  expected-missing (vision/mtp, pruned by design): {len(expected_missing)}")
    for k in unexpected_missing[:20]:
        print(f"  UNEXPECTED MISSING (in ckpt, not produced): {k}")
    for k in extra[:20]:
        print(f"  EXTRA   (produced, not in ckpt): {k}")

    fused_leak = [name for name, p in model.named_parameters() if "wqkv" in name]
    if fused_leak:
        print(f"  FUSED-QKV LEAK: {fused_leak[:5]}")

    n_full = sum(1 for layer_cfg in model.layers if getattr(layer_cfg, "attention", None) is not None)
    n_gdn = sum(1 for layer_cfg in model.layers if getattr(layer_cfg, "delta_net", None) is not None)
    print(f"  layers: {len(model.layers)} total, {n_full} full-attention, {n_gdn} GDN")

    ok = not unexpected_missing and not extra and not fused_leak and (n_full + n_gdn == len(model.layers))
    print("M6 QWEN3_5 KEYMAP:", "PASS (exact match modulo vision/mtp)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
