import shutil
from pathlib import Path

# every weight format the hub ships; copying one would duplicate the model next to the written shards
_HF_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".h5", ".msgpack", ".ckpt")


def copy_hf_metadata_files(source_dir: str | Path, dest_dir: str | Path) -> None:
    """Copy what a written HF checkpoint needs besides its weights: config, tokenizer, remote code."""
    for path in Path(source_dir).iterdir():
        # the source's weight index must not overwrite the written shard mapping
        if path.is_file() and path.suffix not in _HF_WEIGHT_SUFFIXES and not path.name.endswith(".index.json"):
            shutil.copy2(path, Path(dest_dir) / path.name)
