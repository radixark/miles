import json
import os
import shutil
import stat
import tempfile
from collections import defaultdict
from pathlib import Path

import safetensors


DSPARK_SUBDIR = "dspark"
DSPARK_RETROFIT_BACKUP_SUBDIR = "dspark-converted-backup"
MODEL_INDEX_FILENAME = "model.safetensors.index.json"


def _is_dspark_tensor(name: str) -> bool:
    return name.startswith("mtp.")


def _link_or_copy_file(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        mode = "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        mode = "copy"
    size = os.path.getsize(destination)
    print(f"DSpark checkpoint {mode}: {source} -> {destination} ({size} bytes)")
    return destination


def _read_model_index(checkpoint_path: Path) -> dict:
    index_path = checkpoint_path / MODEL_INDEX_FILENAME
    if not index_path.is_file():
        raise ValueError(f"DSpark checkpoint requires {index_path}.")
    with index_path.open() as f:
        model_index = json.load(f)
    if not isinstance(model_index.get("weight_map"), dict):
        raise ValueError(f"DSpark checkpoint index has no weight_map: {index_path}.")
    return model_index


def _validate_indexed_shards(checkpoint_path: Path, weight_map: dict[str, str]) -> None:
    indexed_keys_by_shard: dict[str, set[str]] = defaultdict(set)
    for name, filename in weight_map.items():
        indexed_keys_by_shard[filename].add(name)

    for filename, indexed_keys in indexed_keys_by_shard.items():
        shard_path = checkpoint_path / filename
        if not shard_path.is_file():
            raise ValueError(f"DSpark checkpoint index references missing shard {shard_path}.")
        with safetensors.safe_open(shard_path, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
        if actual_keys != indexed_keys:
            missing_from_index = sorted(actual_keys - indexed_keys)
            missing_from_shard = sorted(indexed_keys - actual_keys)
            raise ValueError(
                f"DSpark shard/index mismatch for {filename}: "
                f"missing_from_index={missing_from_index}, missing_from_shard={missing_from_shard}."
            )


def _copy_nonweight_files(source_path: Path, output_path: Path) -> None:
    for entry in source_path.iterdir():
        if not entry.is_file():
            continue
        if entry.name == MODEL_INDEX_FILENAME or entry.suffix in {
            ".bin",
            ".gguf",
            ".pt",
            ".pth",
            ".safetensors",
        }:
            continue
        _link_or_copy_file(str(entry), str(output_path / entry.name))


def _root_dspark_tensors(checkpoint_path: Path) -> dict[str, list[str]]:
    tensors_by_shard: dict[str, list[str]] = {}
    for shard_path in sorted(checkpoint_path.glob("*.safetensors")):
        with safetensors.safe_open(shard_path, framework="pt", device="cpu") as handle:
            names = [name for name in handle.keys() if _is_dspark_tensor(name)]
        if names:
            tensors_by_shard[shard_path.name] = names
    return tensors_by_shard


def extract_native_dspark_subcheckpoint(
    source_path: str,
    output_path: str,
    model_index: dict,
) -> dict[str, str]:
    """Split native DeepSeek-V4 MTP-only shards into ``output_path/dspark``.

    The returned map contains only target-model tensors. Draft shard bytes and
    auxiliary files are linked when possible and copied otherwise.
    """
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    config_path = source / "config.json"
    with config_path.open() as f:
        config = json.load(f)

    weight_map = model_index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Source model index has no weight_map.")
    is_native_dsv4 = "DeepseekV4ForCausalLM" in config.get("architectures", []) and "embed.weight" in weight_map
    if not is_native_dsv4:
        raise ValueError("--preserve-native-dspark-checkpoint requires a native DeepSeek-V4 checkpoint.")

    draft_weight_map, draft_shards = _split_pure_dspark_shards(source, weight_map)
    draft_output = output / DSPARK_SUBDIR
    if draft_output.exists():
        raise FileExistsError(f"Refusing to overwrite existing DSpark subcheckpoint: {draft_output}.")
    draft_output.mkdir(parents=True)
    _copy_nonweight_files(source, draft_output)
    for filename in sorted(draft_shards):
        destination = draft_output / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy_file(str(source / filename), str(destination))

    metadata = dict(model_index.get("metadata", {}))
    metadata.pop("total_size", None)
    with (draft_output / MODEL_INDEX_FILENAME).open("w") as f:
        json.dump({"metadata": metadata, "weight_map": draft_weight_map}, f, indent=2)

    validate_dspark_subcheckpoint(str(draft_output))
    return {name: filename for name, filename in weight_map.items() if filename not in draft_shards}


def _split_pure_dspark_shards(
    checkpoint_path: Path,
    weight_map: dict[str, str],
) -> tuple[dict[str, str], set[str]]:
    draft_weight_map = {name: filename for name, filename in weight_map.items() if _is_dspark_tensor(name)}
    if not draft_weight_map:
        raise ValueError(f"Checkpoint has no mtp.* tensors to split: {checkpoint_path}.")

    draft_shards = set(draft_weight_map.values())
    mixed_by_shard: dict[str, list[str]] = defaultdict(list)
    for name, filename in weight_map.items():
        if filename in draft_shards and not _is_dspark_tensor(name):
            mixed_by_shard[filename].append(name)
    if mixed_by_shard:
        details = {
            filename: {"count": len(names), "examples": sorted(names)[:5]}
            for filename, names in sorted(mixed_by_shard.items())
        }
        raise ValueError(f"MTP and target tensors share shards in {checkpoint_path}: {details}.")

    _validate_indexed_shards(
        checkpoint_path,
        {name: filename for name, filename in weight_map.items() if filename in draft_shards},
    )
    return draft_weight_map, draft_shards


def _write_filtered_index_atomic(output_path: Path, model_index: dict, weight_map: dict[str, str]) -> None:
    metadata = dict(model_index.get("metadata", {}))
    metadata.pop("total_size", None)
    index_path = output_path / MODEL_INDEX_FILENAME
    index_mode = stat.S_IMODE(index_path.stat().st_mode)
    with tempfile.NamedTemporaryFile(mode="w", dir=output_path, prefix=".dspark-index-", delete=False) as handle:
        temporary_path = Path(handle.name)
        json.dump({"metadata": metadata, "weight_map": weight_map}, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary_path, index_mode)
    try:
        os.replace(temporary_path, index_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def retrofit_native_dspark_subcheckpoint(source_path: str, output_path: str) -> None:
    """Retrofit a converted checkpoint without rewriting its target shards.

    Converted MTP-only shards are moved out of the root through recoverable
    hardlinks/copies under ``dspark-converted-backup`` before the root index is
    atomically filtered.
    """
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    if not output.is_dir():
        raise ValueError(f"Existing converted checkpoint does not exist: {output}.")
    draft_output = output / DSPARK_SUBDIR
    backup_path = output / DSPARK_RETROFIT_BACKUP_SUBDIR
    for reserved_path in (draft_output, backup_path):
        if reserved_path.exists():
            raise FileExistsError(f"Refusing DSpark retrofit because reserved path exists: {reserved_path}.")

    output_index = _read_model_index(output)
    output_weight_map = output_index["weight_map"]
    _draft_weight_map, converted_draft_shards = _split_pure_dspark_shards(output, output_weight_map)
    filtered_output_weight_map = {
        name: filename for name, filename in output_weight_map.items() if filename not in converted_draft_shards
    }

    source_index = _read_model_index(source)
    extract_native_dspark_subcheckpoint(
        source_path=str(source),
        output_path=str(output),
        model_index=source_index,
    )

    backup_path.mkdir()
    _link_or_copy_file(str(output / MODEL_INDEX_FILENAME), str(backup_path / MODEL_INDEX_FILENAME))
    for filename in sorted(converted_draft_shards):
        destination = backup_path / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy_file(str(output / filename), str(destination))

    _write_filtered_index_atomic(output, output_index, filtered_output_weight_map)
    for filename in sorted(converted_draft_shards):
        (output / filename).unlink()

    if _root_dspark_tensors(output):
        raise RuntimeError(f"DSpark retrofit left root mtp.* tensors in {output}.")
    validate_dspark_subcheckpoint(str(output / DSPARK_SUBDIR))
    print(f"Retrofitted {output} with a raw DSpark subcheckpoint; recoverable converted shards are in {backup_path}.")


def validate_dspark_subcheckpoint(checkpoint_path: str) -> dict[str, str]:
    """Validate that a nested checkpoint is self-contained and MTP-only."""
    checkpoint = Path(checkpoint_path).resolve()
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise ValueError(f"DSpark checkpoint requires {config_path}.")
    with config_path.open() as f:
        json.load(f)

    model_index = _read_model_index(checkpoint)
    weight_map = model_index["weight_map"]
    if not weight_map:
        raise ValueError(f"DSpark checkpoint has an empty weight map: {checkpoint}.")
    unexpected_keys = sorted(name for name in weight_map if not _is_dspark_tensor(name))
    if unexpected_keys:
        raise ValueError(f"Nested DSpark index contains non-MTP tensors: {unexpected_keys}.")
    _validate_indexed_shards(checkpoint, weight_map)
    shard_sizes = {filename: (checkpoint / filename).stat().st_size for filename in sorted(set(weight_map.values()))}
    summary = f"Validated DSpark checkpoint {checkpoint}: {len(weight_map)} MTP tensors, shard_bytes={shard_sizes}"
    print(summary)
    return weight_map


def propagate_dspark_subcheckpoint(source_path: str, output_path: str) -> bool:
    """Propagate an existing nested DSpark checkpoint without rewriting it."""
    source_root = Path(source_path).resolve()
    draft_source = source_root / DSPARK_SUBDIR
    if not draft_source.exists():
        return False
    if not draft_source.is_dir():
        raise ValueError(f"Expected DSpark subcheckpoint directory, got {draft_source}.")

    root_index_path = source_root / MODEL_INDEX_FILENAME
    if root_index_path.is_file():
        root_weight_map = _read_model_index(source_root)["weight_map"]
        root_dspark_keys = sorted(name for name in root_weight_map if _is_dspark_tensor(name))
        if root_dspark_keys:
            raise ValueError(
                "Checkpoint has both a nested DSpark checkpoint and root mtp.* tensors in its index: "
                f"count={len(root_dspark_keys)}, examples={root_dspark_keys[:5]}."
            )
    root_dspark_tensors = _root_dspark_tensors(source_root)
    if root_dspark_tensors:
        details = {
            filename: {"count": len(names), "examples": names[:5]} for filename, names in root_dspark_tensors.items()
        }
        message = f"Checkpoint has both a nested DSpark checkpoint and root mtp.* shard tensors: {details}."
        raise ValueError(message)

    validate_dspark_subcheckpoint(str(draft_source))

    draft_output = Path(output_path).resolve() / DSPARK_SUBDIR
    if draft_output.exists():
        raise FileExistsError(f"Refusing to overwrite existing DSpark subcheckpoint: {draft_output}.")
    shutil.copytree(draft_source, draft_output, copy_function=_link_or_copy_file)
    validate_dspark_subcheckpoint(str(draft_output))
    return True
