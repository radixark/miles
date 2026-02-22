"""
Strip layers from an HF safetensors checkpoint.

Usage:
    python strip_hf_layers.py \
        --src  /root/shared/Qwen3-Next-80B-A3B-Thinking \
        --dst  /root/shared/Qwen3-Next-80B-A3B-Thinking-8L \
        --keep-layers 0 1 2 3 4 5 6 7

The script:
  1. Reads model.safetensors.index.json from src.
  2. Drops any weight key whose layer index is NOT in --keep-layers.
     Non-layer keys (embed_tokens, norm, lm_head, mtp.*, ...) are always kept.
  3. Re-packs the remaining weights into new shard files (≤ MAX_SHARD_BYTES each).
  4. Writes a new model.safetensors.index.json and copies all non-shard files
     (config.json, tokenizer files, etc.).
  5. Patches config.json: sets num_hidden_layers = len(keep_layers).
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

MAX_SHARD_BYTES = 5 * 1024**3  # 5 GB per shard


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source HF checkpoint directory")
    p.add_argument("--dst", required=True, help="Output directory")
    p.add_argument(
        "--keep-layers",
        nargs="+",
        type=int,
        required=True,
        help="Layer indices to keep (e.g. 0 1 2 3 4 5 6 7)",
    )
    return p.parse_args()


def should_keep(name: str, keep_set: set) -> bool:
    m = re.match(r"model\.layers\.(\d+)\.", name)
    if m:
        return int(m.group(1)) in keep_set
    return True  # non-layer weights: always keep


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    keep_set = set(args.keep_layers)

    dst.mkdir(parents=True, exist_ok=True)

    # ── 1. Load index ────────────────────────────────────────────────────────
    index_path = src / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]

    # ── 2. Figure out which keys to keep and which source shards we need ─────
    kept_keys = [k for k in weight_map if should_keep(k, keep_set)]
    dropped = len(weight_map) - len(kept_keys)
    print(f"Keeping {len(kept_keys)} / {len(weight_map)} weight keys  (dropped {dropped})")

    needed_shards = sorted(set(weight_map[k] for k in kept_keys))
    print(f"Reading from {len(needed_shards)} source shards …")

    # ── 3. Load all needed tensors (shard by shard to save RAM) ──────────────
    # Build: shard_file -> list of kept keys in that shard
    shard_to_keys: dict[str, list[str]] = {}
    for k in kept_keys:
        shard_to_keys.setdefault(weight_map[k], []).append(k)

    all_tensors: dict[str, torch.Tensor] = {}
    for shard_file in needed_shards:
        print(f"  Loading {shard_file} …", flush=True)
        full = load_file(src / shard_file)
        for k in shard_to_keys[shard_file]:
            all_tensors[k] = full[k]
        del full

    # ── 4. Split into new shards ──────────────────────────────────────────────
    new_shards: list[dict[str, torch.Tensor]] = []
    current_shard: dict[str, torch.Tensor] = {}
    current_bytes = 0

    for k in kept_keys:  # preserve original key order
        t = all_tensors[k]
        nbytes = t.numel() * t.element_size()
        if current_shard and current_bytes + nbytes > MAX_SHARD_BYTES:
            new_shards.append(current_shard)
            current_shard = {}
            current_bytes = 0
        current_shard[k] = t
        current_bytes += nbytes

    if current_shard:
        new_shards.append(current_shard)

    total_shards = len(new_shards)
    print(f"Writing {total_shards} output shard(s) …")

    new_weight_map: dict[str, str] = {}
    for i, shard in enumerate(new_shards, 1):
        fname = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        print(f"  Saving {fname}  ({len(shard)} tensors) …", flush=True)
        save_file(shard, dst / fname)
        for k in shard:
            new_weight_map[k] = fname

    # ── 5. Write new index ───────────────────────────────────────────────────
    new_index = {
        "metadata": index.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)
    print("Wrote model.safetensors.index.json")

    # ── 6. Copy non-shard files (config, tokenizer, …) ───────────────────────
    skip_patterns = {
        re.compile(r"model-\d+-of-\d+\.safetensors"),
        re.compile(r"model\.safetensors\.index\.json"),
    }
    for item in src.iterdir():
        if any(pat.match(item.name) for pat in skip_patterns):
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
        print(f"  Copied {item.name}")

    # ── 7. Patch config.json ──────────────────────────────────────────────────
    cfg_path = dst / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        old_val = cfg.get("num_hidden_layers")
        cfg["num_hidden_layers"] = len(keep_set)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Patched config.json: num_hidden_layers {old_val} → {len(keep_set)}")

    print("Done.")


if __name__ == "__main__":
    main()
