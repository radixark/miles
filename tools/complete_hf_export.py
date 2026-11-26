"""Complete a raw ``--save-hf`` export into a standalone HF checkpoint.

Copies every base-checkpoint tensor missing from the export (a frozen vision tower, an untrained
MTP layer) into completion shards, under the export's HF naming scheme, and adds them to the
export's ``model.safetensors.index.json``. Exported tensors are never touched, so it is safe on
any raw export, and a second run is a no-op.

The base checkpoint must store its weights in the same format as the export: completing a bf16
export from an fp8-quantized base would copy orphaned quantization scale tensors. Cast such a
base first (see tools/fp8_cast_bf16.py).

Usage:
    python tools/complete_hf_export.py --export-dir /path/to/hf/step_3 \
        --hf-checkpoint /path/to/Qwen3.5-2B
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import safetensors.torch
from safetensors import safe_open

from miles.utils.hf_utils.weight_mapping import get_checkpoint_weight_map, missing_base_tensors

COMPLETION_SHARD_FORMAT = "model-completion-{:05d}.safetensors"


def complete_export(export_dir: str | Path, hf_checkpoint: str | Path) -> list[str]:
    """Copy base-checkpoint tensors missing from the export into completion shards.

    :returns: The names of the copied tensors, in the export's naming scheme.
    """
    export_dir, hf_checkpoint = Path(export_dir), Path(hf_checkpoint)
    index_path = export_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"{index_path} not found; the directory is not a raw --save-hf export.")
    export_map = get_checkpoint_weight_map(export_dir)
    missing_by_name = missing_base_tensors(export_map.keys(), hf_checkpoint)
    if not missing_by_name:
        print(f"{export_dir} already holds every tensor of {hf_checkpoint}; nothing to do.")
        return []
    referenced_completions = sorted({shard for shard in export_map.values() if shard.startswith("model-completion-")})
    if referenced_completions:
        raise ValueError(
            f"{export_dir} already holds completion shards ({', '.join(referenced_completions)}) but is still "
            f"missing {len(missing_by_name)} tensors of {hf_checkpoint}; it was likely completed from a different "
            "base checkpoint. Refusing to mix completions from two bases."
        )
    base_map = get_checkpoint_weight_map(hf_checkpoint)
    missing_by_base_shard: dict[str, list[str]] = defaultdict(list)
    for name, base_name in missing_by_name.items():
        missing_by_base_shard[base_map[base_name]].append(name)

    # One completion shard per base shard keeps only that shard's tensors in memory at a time.
    # An unreferenced completion shard left by a crashed run is overwritten.
    index = json.loads(index_path.read_text(encoding="utf-8"))
    added_bytes = 0
    for shard_index, (base_shard, names) in enumerate(sorted(missing_by_base_shard.items()), start=1):
        completion_shard = COMPLETION_SHARD_FORMAT.format(shard_index)
        with safe_open(hf_checkpoint / base_shard, framework="pt", device="cpu") as reader:
            tensors = {name: reader.get_tensor(missing_by_name[name]) for name in sorted(names)}
        safetensors.torch.save_file(tensors, export_dir / completion_shard, metadata={"format": "pt"})
        index["weight_map"].update({name: completion_shard for name in names})
        added_bytes += sum(t.numel() * t.element_size() for t in tensors.values())
        del tensors

    # Written to a temporary file and moved over the old index, so a crash mid-write leaves the
    # export's previous index intact.
    metadata = index.setdefault("metadata", {})
    if "total_size" in metadata:
        metadata["total_size"] += added_bytes
    tmp_index_path = index_path.with_name(index_path.name + ".tmp")
    with tmp_index_path.open("w", encoding="utf-8") as tmp_index_file:
        json.dump(index, tmp_index_file, indent=2)
        tmp_index_file.flush()
        os.fsync(tmp_index_file.fileno())
    os.replace(tmp_index_path, index_path)
    print(
        f"Copied {len(missing_by_name)} tensors from {hf_checkpoint} into {len(missing_by_base_shard)} "
        f"completion shard(s); {index_path} now lists {len(index['weight_map'])} tensors."
    )
    return sorted(missing_by_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", required=True, help="A raw --save-hf export directory.")
    parser.add_argument("--hf-checkpoint", required=True, help="The base HF checkpoint the run trained from.")
    args = parser.parse_args()
    complete_export(args.export_dir, args.hf_checkpoint)


if __name__ == "__main__":
    main()
