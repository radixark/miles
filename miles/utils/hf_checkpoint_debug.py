from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover - optional dependency
    safe_open = None


@dataclass
class CheckpointIndex:
    root: str
    weight_map: dict[str, str]
    fmt: str
    keys: list[str]


class TensorFileCache:
    def __init__(self, fmt: str) -> None:
        self.fmt = fmt
        self._cache: dict[str, object] = {}

    def _open_file(self, path: str) -> object:
        if self.fmt == "safetensors":
            if safe_open is None:
                raise RuntimeError(
                    "safetensors is required to read .safetensors checkpoints. Install with: pip install safetensors"
                )
            return safe_open(path, framework="pt", device="cpu")
        if self.fmt == "bin":
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
                obj = obj["state_dict"]
            if not isinstance(obj, dict):
                raise ValueError(f"Unsupported checkpoint content in {path}")
            return obj
        raise ValueError(f"Unsupported checkpoint format: {self.fmt}")

    def get_tensor(self, path: str, key: str) -> torch.Tensor:
        if path not in self._cache:
            self._cache[path] = self._open_file(path)
        file_obj = self._cache[path]
        if self.fmt == "safetensors":
            return file_obj.get_tensor(key)
        return file_obj[key]


class Checkpoint:
    def __init__(self, index: CheckpointIndex) -> None:
        self.index = index
        self.cache = TensorFileCache(index.fmt)

    @property
    def keys(self) -> list[str]:
        return self.index.keys

    def get_tensor(self, key: str) -> torch.Tensor:
        return self.cache.get_tensor(self.index.weight_map[key], key)


def detect_format(paths: Iterable[str]) -> str:
    extensions = {os.path.splitext(p)[1].lower() for p in paths}
    has_safetensors = ".safetensors" in extensions
    has_bin = any(ext in {".bin", ".pt", ".pth"} for ext in extensions)
    if has_safetensors and has_bin:
        raise ValueError("Mixed checkpoint formats are not supported in one index")
    if has_safetensors:
        return "safetensors"
    if has_bin:
        return "bin"
    raise ValueError(f"Unsupported checkpoint file extensions: {sorted(extensions)}")


def find_index_file(directory: str) -> str | None:
    preferred = [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    for name in preferred:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    index_files = [f for f in os.listdir(directory) if f.endswith(".index.json")]
    if len(index_files) == 1:
        return os.path.join(directory, index_files[0])
    if len(index_files) > 1:
        raise ValueError(f"Multiple index files found in {directory}. Specify the index file directly.")
    return None


def find_single_weight_file(directory: str) -> str | None:
    preferred = [
        "model.safetensors",
        "pytorch_model.bin",
    ]
    for name in preferred:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    candidates: list[str] = []
    for suffix in (".safetensors", ".bin", ".pt", ".pth"):
        candidates.extend(os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(suffix))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Multiple weight files found in {directory}. Specify a file explicitly.")
    return None


def load_index(index_path: str) -> CheckpointIndex:
    with open(index_path, encoding="utf-8") as handle:
        data = json.load(handle)
    if "weight_map" not in data or not isinstance(data["weight_map"], dict):
        raise ValueError(f"Invalid index file: {index_path}")
    root = os.path.dirname(index_path)
    raw_map: dict[str, str] = data["weight_map"]
    weight_map = {key: os.path.join(root, value) for key, value in raw_map.items()}
    fmt = detect_format(weight_map.values())
    missing = [path for path in weight_map.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing shard files: {missing[:5]}")
    keys = sorted(weight_map.keys())
    return CheckpointIndex(root=root, weight_map=weight_map, fmt=fmt, keys=keys)


def load_single_file(path: str) -> Checkpoint:
    fmt = detect_format([path])
    index = CheckpointIndex(
        root=os.path.dirname(path),
        weight_map={},
        fmt=fmt,
        keys=[],
    )
    checkpoint = Checkpoint(index)
    if fmt == "safetensors":
        if safe_open is None:
            raise RuntimeError(
                "safetensors is required to read .safetensors checkpoints. Install with: pip install safetensors"
            )
        reader = safe_open(path, framework="pt", device="cpu")
        keys = sorted(reader.keys())
        index.weight_map.update({key: path for key in keys})
        index.keys.extend(keys)
        checkpoint.cache._cache[path] = reader
        return checkpoint
    state_dict = checkpoint.cache._open_file(path)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint content in {path}")
    keys = sorted(state_dict.keys())
    index.weight_map.update({key: path for key in keys})
    index.keys.extend(keys)
    checkpoint.cache._cache[path] = state_dict
    return checkpoint


def build_checkpoint(path: str | Path) -> Checkpoint:
    path = os.path.abspath(str(path))
    if os.path.isdir(path):
        index_path = find_index_file(path)
        if index_path:
            return Checkpoint(load_index(index_path))
        weight_file = find_single_weight_file(path)
        if weight_file:
            return load_single_file(weight_file)
        raise FileNotFoundError(f"No checkpoint files found in {path}")
    if os.path.isfile(path):
        if path.endswith(".index.json"):
            return Checkpoint(load_index(path))
        return load_single_file(path)
    raise FileNotFoundError(f"Checkpoint path does not exist: {path}")


def resolve_checkpoint_path(path_or_repo: str | None) -> str | None:
    if not path_or_repo:
        return None
    if os.path.exists(path_or_repo):
        return os.path.abspath(path_or_repo)
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning(
            "Cannot resolve HF repo id %s (huggingface_hub unavailable): %s",
            path_or_repo,
            exc,
        )
        return None
    try:
        return snapshot_download(
            repo_id=path_or_repo,
            allow_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth", "*.index.json"],
        )
    except Exception as exc:
        logger.warning("Failed to download HF checkpoint %s: %s", path_or_repo, exc)
        return None


def _avg_relative_diff(t1: torch.Tensor, t2: torch.Tensor, mismatch_mask: torch.Tensor, eps: float) -> float:
    if mismatch_mask.numel() == 0:
        return 0.0
    mismatch_count = mismatch_mask.sum().item()
    if mismatch_count == 0:
        return 0.0
    t1_f = t1
    t2_f = t2
    if torch.is_complex(t1_f):
        t1_f = t1_f.abs()
        t2_f = t2_f.abs()
    t1_f = t1_f.float()
    t2_f = t2_f.float()
    rel = (t1_f - t2_f).abs() / (t2_f.abs() + eps)
    rel_sum = (rel * mismatch_mask).sum().item()
    return rel_sum / mismatch_count


def compare_checkpoints_and_report(
    checkpoint_a: Checkpoint,
    checkpoint_b: Checkpoint,
    *,
    label_a: str,
    label_b: str,
    eps: float = 1e-8,
) -> None:
    keys_a = set(checkpoint_a.keys)
    keys_b = set(checkpoint_b.keys)
    missing_in_b = sorted(keys_a - keys_b)
    missing_in_a = sorted(keys_b - keys_a)
    for key in missing_in_b:
        logger.info("Weight missing in %s: %s", label_b, key)
    for key in missing_in_a:
        logger.info("Weight missing in %s: %s", label_a, key)

    mismatches = 0
    for key in sorted(keys_a & keys_b):
        t1 = checkpoint_a.get_tensor(key)
        t2 = checkpoint_b.get_tensor(key)
        if t1.shape != t2.shape or t1.dtype != t2.dtype:
            logger.info(
                "Weight mismatch %s: shape/dtype %s/%s vs %s/%s",
                key,
                tuple(t1.shape),
                t1.dtype,
                tuple(t2.shape),
                t2.dtype,
            )
            mismatches += 1
            continue
        if torch.equal(t1, t2):
            continue
        mismatch_mask = t1 != t2
        mismatch_count = mismatch_mask.sum().item()
        total = t1.numel()
        mismatch_pct = (mismatch_count / total) * 100 if total else 0.0
        avg_rel_diff = _avg_relative_diff(t1, t2, mismatch_mask, eps)
        logger.info(
            "Weight mismatch %s: mismatch_pct=%.6f avg_rel_diff=%.6g",
            key,
            mismatch_pct,
            avg_rel_diff,
        )
        mismatches += 1

    logger.info(
        "First weight sync comparison complete: %d mismatched tensors, %d missing in %s, %d missing in %s",
        mismatches,
        len(missing_in_a),
        label_a,
        len(missing_in_b),
        label_b,
    )


class DebugFirstWeightSync:
    def __init__(self, output_dir: str, source_checkpoint: str | None, write_rank: bool) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source_checkpoint = source_checkpoint
        self.write_rank = write_rank
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self._shard_idx = 0
        self._weight_map: dict[str, str] = {}

    def write_chunk(self, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        if not self.write_rank:
            return
        if not named_tensors:
            return
        state_dict = {name: tensor.detach().cpu() for name, tensor in named_tensors}
        self._shard_idx += 1
        file_name = f"pytorch_model.debug-rank{self.rank:05d}-{self._shard_idx:05d}.bin"
        file_path = self.output_dir / file_name
        torch.save(state_dict, file_path)
        for name in state_dict:
            if name in self._weight_map and self._weight_map[name] != file_name:
                logger.warning(
                    "Duplicate weight key %s encountered in debug first weight sync; keeping first entry.",
                    name,
                )
                continue
            self._weight_map[name] = file_name
        del state_dict

    def finalize_and_compare(self, group: dist.ProcessGroup | None = None) -> None:
        if not dist.is_initialized():
            return
        dist.barrier(group=group)
        local_map = self._weight_map if self.write_rank else {}
        all_maps = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(local_map, object_gather_list=all_maps, dst=0, group=group)

        if dist.get_rank() == 0:
            merged: dict[str, str] = {}
            for rank_map in all_maps or []:
                if not rank_map:
                    continue
                for key, value in rank_map.items():
                    if key in merged and merged[key] != value:
                        logger.warning(
                            "Duplicate weight key %s across ranks; keeping first entry.",
                            key,
                        )
                        continue
                    merged[key] = value
            self._write_index(merged)
            self._compare_with_source()

        dist.barrier(group=group)

    def _write_index(self, weight_map: dict[str, str]) -> None:
        index_path = self.output_dir / "pytorch_model.bin.index.json"
        total_size = 0
        for file_name in set(weight_map.values()):
            file_path = self.output_dir / file_name
            if file_path.exists():
                total_size += file_path.stat().st_size
        payload = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        with open(index_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        logger.info("Saved debug first weight sync checkpoint to %s", self.output_dir)

    def _compare_with_source(self) -> None:
        if not self.source_checkpoint:
            logger.warning("debug-first-weight-sync: --hf-checkpoint not set; skip compare.")
            return
        source_path = resolve_checkpoint_path(self.source_checkpoint)
        if source_path is None:
            logger.warning(
                "debug-first-weight-sync: could not resolve hf checkpoint %s; skip compare.",
                self.source_checkpoint,
            )
            return
        try:
            debug_checkpoint = build_checkpoint(self.output_dir)
            source_checkpoint = build_checkpoint(source_path)
        except Exception as exc:
            logger.warning("debug-first-weight-sync: failed to load checkpoints: %s", exc)
            return
        logger.info(
            "Comparing first weight sync checkpoint (%s) with sglang init checkpoint (%s)",
            self.output_dir,
            source_path,
        )
        compare_checkpoints_and_report(
            debug_checkpoint,
            source_checkpoint,
            label_a="debug_first_weight_sync",
            label_b="sglang_hf_checkpoint",
        )
