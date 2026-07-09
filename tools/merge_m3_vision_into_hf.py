"""Merge the original M3 vision tower + projector into an LM-only exported HF dir.

convert_torch_dist_to_hf.py exports only the trained LM weights
(``language_model.model.*``); text-only SFT never touches the vision tower. This
copies the untrained vision params (``vision_tower.*`` / ``multi_modal_projector.*``
/ ``patch_merge_mlp.*``) verbatim from the ORIGIN HF model into the exported dir and
extends model.safetensors.index.json, producing a complete servable VL checkpoint.

Usage:
  python tools/merge_m3_vision_into_hf.py \
    --origin-hf-dir /fsx/peng/models/MiniMax-M3 \
    --exported-hf-dir /fsx/peng/models/MiniMax-M3_sft_hf
"""

import argparse
import json
import os

import safetensors.torch
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin-hf-dir", required=True, help="original full M3 HF (has vision weights)")
    ap.add_argument("--exported-hf-dir", required=True, help="LM-only exported HF to complete in place")
    ap.add_argument("--out-shard", default="model-vision-00000-of-00001.safetensors")
    args = ap.parse_args()

    oidx = json.load(open(os.path.join(args.origin_hf_dir, "model.safetensors.index.json")))
    owm = oidx["weight_map"]
    vis_keys = [k for k in owm if not k.startswith("language_model.")]
    print(f"vision/non-LM keys to merge: {len(vis_keys)}")

    eidx_path = os.path.join(args.exported_hf_dir, "model.safetensors.index.json")
    eidx = json.load(open(eidx_path))
    ewm = eidx["weight_map"]
    overlap = [k for k in vis_keys if k in ewm]
    assert not overlap, f"exported dir already has {len(overlap)} vision keys, e.g. {overlap[:3]}"

    # load vision tensors (group reads by source shard for efficiency)
    by_shard = {}
    for k in vis_keys:
        by_shard.setdefault(owm[k], []).append(k)
    tensors, total_bytes = {}, 0
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(args.origin_hf_dir, shard), framework="pt") as f:
            for k in keys:
                t = f.get_tensor(k)
                tensors[k] = t
                total_bytes += t.numel() * t.element_size()
    print(f"loaded {len(tensors)} vision tensors, {total_bytes/1e9:.2f} GB")

    out_path = os.path.join(args.exported_hf_dir, args.out_shard)
    safetensors.torch.save_file(tensors, out_path)
    print(f"wrote {out_path}")

    # extend the index: point vision keys at the new shard, bump total_size
    for k in vis_keys:
        ewm[k] = args.out_shard
    eidx.setdefault("metadata", {})
    eidx["metadata"]["total_size"] = int(eidx["metadata"].get("total_size", 0)) + total_bytes
    json.dump(eidx, open(eidx_path, "w"), indent=2)
    print(f"updated index -> {len(ewm)} total keys ({len(vis_keys)} vision added)")


if __name__ == "__main__":
    main()
