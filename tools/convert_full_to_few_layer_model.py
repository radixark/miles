#!/usr/bin/env python3
import os
import re
import json
import shutil
import argparse
from safetensors.torch import load_file, save_file

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--layers", type=int, default=5)
    return p.parse_args()

def copy_non_weight_files(src, dst):
    os.makedirs(dst, exist_ok=True)
    for fname in os.listdir(src):
        if fname.endswith(".safetensors"):
            continue
        if fname.endswith(".index.json"):
            continue
        s = os.path.join(src, fname)
        d = os.path.join(dst, fname)
        if os.path.isfile(s):
            shutil.copy(s, d)

def load_index(src):
    """Load the model.safetensors.index.json if it exists."""
    index_path = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return json.load(f)
    return None

def patch_config(dst, n_layers):
    cfg_path = os.path.join(dst, "config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    for k in ["num_hidden_layers", "n_layer"]:
        if k in cfg:
            cfg[k] = n_layers
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

def make_keep_fn(max_layers):
    pat = re.compile(r"layers\.(\d+)\.")
    def keep(name: str) -> bool:
        if not name.startswith("layers."):
            return True
        m = pat.match(name)
        if not m:
            return False
        return int(m.group(1)) < max_layers
    return keep

def prune_weights(src, dst, max_layers):
    keep = make_keep_fn(max_layers)
    
    # Load original index if exists
    orig_index = load_index(src)
    
    # Track which weights go to which files
    weight_map = {}  # weight_name -> filename
    file_weights = {}  # filename -> {weight_name: tensor}
    
    # Process all safetensor files
    safetensor_files = sorted([f for f in os.listdir(src) if f.endswith(".safetensors")])
    
    for fname in safetensor_files:
        path = os.path.join(src, fname)
        tensors = load_file(path)
        
        kept_in_file = {}
        for k, v in tensors.items():
            if keep(k):
                if k in weight_map:
                    raise RuntimeError(f"duplicate key {k}")
                print(f"Keeping: {k} from {fname}")
                kept_in_file[k] = v
                weight_map[k] = fname
        
        if kept_in_file:
            file_weights[fname] = kept_in_file
    
    # Save the pruned safetensor files
    for fname, weights in file_weights.items():
        out_path = os.path.join(dst, fname)
        save_file(weights, out_path)
        print(f"Saved {len(weights)} tensors to {fname}")
    
    # Create new index.json
    if orig_index and "metadata" in orig_index:
        metadata = orig_index["metadata"].copy()
        # Update total_size if present
        if "total_size" in metadata:
            total_size = sum(
                sum(t.numel() * t.element_size() for t in weights.values())
                for weights in file_weights.values()
            )
            metadata["total_size"] = total_size
    else:
        metadata = {}
    
    new_index = {
        "metadata": metadata,
        "weight_map": weight_map
    }
    
    index_path = os.path.join(dst, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(new_index, f, indent=2)
    print(f"Saved index with {len(weight_map)} weights")

def main():
    args = parse_args()
    copy_non_weight_files(args.src, args.dst)
    patch_config(args.dst, args.layers)
    prune_weights(args.src, args.dst, args.layers)

if __name__ == "__main__":
    main()

