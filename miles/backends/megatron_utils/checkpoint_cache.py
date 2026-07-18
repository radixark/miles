import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


_SENTINEL = ".miles_cache_manifest.json"
_MIN_FREE_BYTES = 256 * 1024**3


def _checkpoint_dir(checkpoint_root: Path) -> Path:
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    value = tracker.read_text().strip()
    if value == "release":
        return checkpoint_root / "release"
    return checkpoint_root / f"iter_{int(value):07d}"


def _manifest(source_root: Path, files: list[Path]) -> dict[str, object]:
    entries = []
    for path in files:
        stat = path.stat()
        entries.append(
            {
                "path": str(path.relative_to(source_root)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    serialized = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {"digest": hashlib.sha256(serialized.encode()).hexdigest(), "files": entries}


def _is_complete(target_root: Path, manifest: dict[str, object]) -> bool:
    sentinel = target_root / _SENTINEL
    if not sentinel.is_file() or json.loads(sentinel.read_text()) != manifest:
        return False
    return all(
        (target_root / entry["path"]).is_file() and (target_root / entry["path"]).stat().st_size == entry["size"]
        for entry in manifest["files"]
    )


def prepare_rank_local_checkpoint(source: str, cache_root: str, rank: int) -> str:
    source_root = Path(source).resolve()
    source_checkpoint_dir = _checkpoint_dir(source_root)
    if not source_checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {source_checkpoint_dir}")

    rank_files = sorted(source_checkpoint_dir.glob(f"__{rank}_*.distcp"))
    if not rank_files:
        raise FileNotFoundError(
            f"Checkpoint {source_root} has no distcp files for rank {rank}; "
            "the checkpoint must use the same world size and parallel topology as training"
        )

    metadata_files = sorted(
        path for path in source_checkpoint_dir.iterdir() if path.is_file() and path.suffix != ".distcp"
    )
    source_files = [source_root / "latest_checkpointed_iteration.txt", *metadata_files, *rank_files]
    manifest = _manifest(source_root, source_files)

    cache_base = Path(cache_root)
    cache_base.mkdir(parents=True, exist_ok=True)
    target_root = cache_base / f"rank-{rank}"
    lock_path = cache_base / f".rank-{rank}.lock"

    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _is_complete(target_root, manifest):
            return str(target_root)

        temporary_root = cache_base / f".rank-{rank}.tmp"
        if temporary_root.exists():
            shutil.rmtree(temporary_root)

        required_bytes = sum(entry["size"] for entry in manifest["files"])
        existing_bytes = (
            sum(path.stat().st_size for path in target_root.rglob("*") if path.is_file())
            if target_root.exists()
            else 0
        )
        free_bytes = shutil.disk_usage(cache_base).free + existing_bytes
        if free_bytes < required_bytes + _MIN_FREE_BYTES:
            raise OSError(
                f"Insufficient local checkpoint cache space at {cache_base}: "
                f"free={free_bytes}, required={required_bytes}, reserve={_MIN_FREE_BYTES}"
            )

        temporary_root.mkdir()

        for source_path in source_files:
            relative_path = source_path.relative_to(source_root)
            target_path = temporary_root / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)

        if not all((temporary_root / entry["path"]).stat().st_size == entry["size"] for entry in manifest["files"]):
            raise OSError(f"Local checkpoint cache verification failed for rank {rank}")
        (temporary_root / _SENTINEL).write_text(json.dumps(manifest, sort_keys=True))

        if target_root.exists():
            shutil.rmtree(target_root)
        os.replace(temporary_root, target_root)
        return str(target_root)
