import os
import re
import glob
import torch
import argparse
from collections import defaultdict
from typing import Dict, Set, Tuple

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

def get_weight_files(directory: str) -> list:
    """Finds all weight files in the directory (safetensors or bin)."""
    # Priority 1: Safetensors
    files = glob.glob(os.path.join(directory, "*.safetensors"))
    if files:
        return sorted(files), "safetensors"
    
    # Priority 2: PyTorch Bin
    files = glob.glob(os.path.join(directory, "*.bin"))
    if files:
        return sorted(files), "torch"
    
    return [], None

def analyze_weights(model_path: str):
    print(f"--- Analyzing Checkpoint: {model_path} ---")
    
    weight_files, file_type = get_weight_files(model_path)
    
    if not weight_files:
        print("❌ No checkpoint files (.safetensors or .bin) found in the directory.")
        return

    print(f"Detected format: {file_type}")
    print(f"Found {len(weight_files)} file(s). Scanning headers...")

    # Store normalized_name -> set of (shape, dtype)
    # Using a set to handle cases where different layers might have different shapes (rare but possible)
    grouped_params: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

    for file_path in weight_files:
        try:
            if file_type == "safetensors":
                if safe_open is None:
                    print("❌ Error: 'safetensors' library is required. Install via `pip install safetensors`")
                    return
                
                # Context manager to open file without loading data to RAM
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        # Get metadata (shape and dtype) without loading the tensor
                        slice_obj = f.get_slice(key)
                        shape = str(tuple(slice_obj.get_shape()))
                        dtype = str(slice_obj.get_dtype()).split(".")[-1] # Clean string (e.g. 'float16')
                        
                        # --- Normalization Logic ---
                        # 1. Normalize Layers: layers.0, h.0, blocks.0 -> layers.{idx}
                        normalized_key = re.sub(r'\b(layers|h|blocks)\.\d+', r'\1.{idx}', key)
                        
                        # 2. Normalize Experts: experts.0 -> experts.{idx}
                        normalized_key = re.sub(r'\b(experts)\.\d+', r'\1.{idx}', normalized_key)
                        
                        grouped_params[normalized_key].add((shape, dtype))
            
            elif file_type == "torch":
                # For .bin files, we must load the header. 
                # map_location='meta' helps avoid OOM but isn't always supported by basic torch.load
                # map_location='cpu' is safer for compatibility.
                weights = torch.load(file_path, map_location="cpu")
                for key, tensor in weights.items():
                    shape = str(tuple(tensor.shape))
                    dtype = str(tensor.dtype).replace("torch.", "")
                    
                    # --- Normalization Logic ---
                    normalized_key = re.sub(r'\b(layers|h|blocks)\.\d+', r'\1.{idx}', key)
                    normalized_key = re.sub(r'\b(experts)\.\d+', r'\1.{idx}', normalized_key)
                    
                    grouped_params[normalized_key].add((shape, dtype))
                
                # Clean up memory immediately
                del weights
                
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    # --- Print Results ---
    print(f"\n{'Weight Name (Normalized)':<60} | {'Shape':<25} | {'Dtype':<10}")
    print("-" * 100)

    for name in sorted(grouped_params.keys()):
        infos = grouped_params[name]
        for shape, dtype in infos:
            print(f"{name:<60} | {shape:<25} | {dtype:<10}")

    print("-" * 100)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect weights of a local HuggingFace checkpoint.")
    parser.add_argument("path", type=str, help="Path to the folder containing .safetensors or .bin files")
    
    args = parser.parse_args()
    
    if os.path.exists(args.path):
        analyze_weights(args.path)
    else:
        print(f"❌ Path does not exist: {args.path}")