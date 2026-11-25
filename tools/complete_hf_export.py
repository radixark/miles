"""Complete a raw ``--save-hf`` export into a standalone HF checkpoint.

Copies every base-checkpoint tensor missing from the export (a frozen vision tower, an untrained
MTP layer) into one completion shard and adds it to the export's ``model.safetensors.index.json``.
Exported tensors are never touched, so it is safe on any raw export, and a second run is a no-op.

Usage:
    python tools/complete_hf_export.py --export-dir /path/to/hf/step_3 \
        --hf-checkpoint /path/to/Qwen3.5-2B
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import safetensors.torch
from safetensors import safe_open

from miles.utils.hf_utils.weight_mapping import get_checkpoint_weight_map

COMPLETION_SHARD = "model-completion-00001.safetensors"


def complete_export(export_dir: str | Path, hf_checkpoint: str | Path) -> list[str]:
    """Copy base-checkpoint tensors missing from the export into a completion shard.

    :returns: The names of the copied tensors.
    """
    export_dir, hf_checkpoint = Path(export_dir), Path(hf_checkpoint)
    index_path = export_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"{index_path} not found; the directory is not a raw --save-hf export.")
    export_map = get_checkpoint_weight_map(export_dir)
    base_map = get_checkpoint_weight_map(hf_checkpoint)
    missing = sorted(set(base_map) - set(export_map))
    if not missing:
        print(f"{export_dir} already holds every tensor of {hf_checkpoint}; nothing to do.")
        return []
    assert COMPLETION_SHARD not in set(export_map.values()), f"{COMPLETION_SHARD} is already in the index"

    # The missing tensors are the small side of the checkpoint; they are held in memory until the
    # shard is written. Each base shard is opened once.
    missing_by_shard: dict[str, list[str]] = defaultdict(list)
    for name in missing:
        missing_by_shard[base_map[name]].append(name)
    tensors = {}
    for shard_name, names in sorted(missing_by_shard.items()):
        with safe_open(hf_checkpoint / shard_name, framework="pt", device="cpu") as reader:
            for name in names:
                tensors[name] = reader.get_tensor(name)

    # An unreferenced completion shard left by a crashed run is overwritten. The index is updated
    # last, so a crash mid-write leaves the export's previous index intact.
    safetensors.torch.save_file(tensors, export_dir / COMPLETION_SHARD)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"].update({name: COMPLETION_SHARD for name in missing})
    metadata = index.setdefault("metadata", {})
    metadata["total_size"] = metadata.get("total_size", 0) + sum(
        t.numel() * t.element_size() for t in tensors.values()
    )
    index_path.write_text(json.dumps(index, indent=2))
    print(f"Copied {len(missing)} tensors from {hf_checkpoint}; {index_path} now lists {len(index['weight_map'])}.")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", required=True, help="A raw --save-hf export directory.")
    parser.add_argument("--hf-checkpoint", required=True, help="The base HF checkpoint the run trained from.")
    args = parser.parse_args()
    complete_export(args.export_dir, args.hf_checkpoint)


if __name__ == "__main__":
    main()
