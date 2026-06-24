from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import socket
import sys
import time
from dataclasses import dataclass
from typing import Any, cast

import ray
import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, LoadPlanner, ReadItem
from torch.distributed.checkpoint.planner_helpers import create_read_items_for_chunk_list
from torch.distributed.checkpoint.utils import _create_file_view
from torch.futures import Future
from tqdm.auto import tqdm
from transformers import AutoConfig
from typing_extensions import override

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miles.backends.megatron_utils import megatron_to_hf as m2hf

DEFAULT_DIRECT_MOE_GROUP_SIZE = 2 * 1024**3
CHECKSUM_DIRNAME = ".ray-convert-checksums"
CHECKSUM_READ_BYTES = 64 * 1024**2


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


pickle.Unpickler = UnpicklerWrapper


@dataclass(frozen=True)
class Args:
    input_dir: str
    output_dir: str
    origin_hf_dir: str | None
    model_name: str | None
    force: bool
    max_file_bytes: int
    concurrency: int | None
    task_group_bytes: int
    source_key_regex: str | None
    dry_run_plan: bool
    sha1sum_output: bool
    progress: bool
    progress_interval_seconds: float


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    keys: tuple[str, ...]
    estimated_source_bytes: int
    moe_blocks: tuple[MoeBlockSpec, ...] = ()


@dataclass(frozen=True)
class MoeBlockSpec:
    source_key: str
    relative_path: str
    storage_indices: tuple[MetadataIndex, ...]
    layer_idx: int
    linear_name: str
    hf_prefix: str


@dataclass(frozen=True)
class PreparedTensorGroup:
    source_name: str
    tensors: tuple[tuple[str, torch.Tensor], ...]


@dataclass(frozen=True)
class TaskLoadStats:
    read_items: int
    files: int
    storage_bytes: int


@dataclass(frozen=True)
class PreparedTaskTensors:
    groups: tuple[PreparedTensorGroup, ...]
    load_stats: TaskLoadStats


@dataclass(frozen=True)
class ShardManifest:
    temp_filename: str
    final_filename: str | None
    weight_keys: tuple[str, ...]
    bytes: int
    tensors: int
    checksum_filename: str | None = None


@dataclass(frozen=True)
class TaskResult:
    task_id: int
    actor_id: int
    node: str
    pid: int
    shards: tuple[ShardManifest, ...]
    source_bytes: int
    output_bytes: int
    weights: int
    source_keys: tuple[str, ...]
    dcp_read_items: int
    dcp_files: int
    dcp_storage_bytes: int
    cuda_device_id: int | None
    ray_node_id: str


@dataclass(frozen=True)
class DcpLoadResult:
    state_dict: dict[str, torch.Tensor]
    read_items: int
    files: int
    storage_bytes: int


@dataclass(frozen=True)
class DirectMoeLoadResult:
    tensor_groups: tuple[PreparedTensorGroup, ...]
    read_items: int
    files: int
    storage_bytes: int


@dataclass(frozen=True)
class PlannedShard:
    temp_filename: str
    final_filename: str
    weight_keys: tuple[str, ...]
    bytes: int
    checksum_filename: str | None


class ProgressReporter:
    def __init__(
        self,
        tasks: list[TaskSpec],
        enabled: bool,
        interval_seconds: float,
        stream: Any = sys.stderr,
    ) -> None:
        self.enabled = enabled
        self.interval_seconds = max(interval_seconds, 0.1)
        self.total_tasks = len(tasks)
        self.total_bytes = sum(task.estimated_source_bytes for task in tasks)
        self.completed_tasks = 0
        self.last_refresh_time = 0.0
        self.progress = tqdm(
            total=self.total_bytes or self.total_tasks,
            desc="Converting",
            unit="B" if self.total_bytes else "task",
            unit_scale=bool(self.total_bytes),
            unit_divisor=1000,
            mininterval=self.interval_seconds,
            disable=not enabled,
            file=stream,
        )
        self._set_postfix()

    def complete(self, result: TaskResult) -> None:
        self.completed_tasks += 1
        self.progress.update(result.source_bytes if self.total_bytes else 1)
        self._set_postfix()

    def tick(self) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self.last_refresh_time >= self.interval_seconds:
            self.progress.refresh()
            self.last_refresh_time = now

    def finish(self) -> None:
        self._set_postfix()
        self.progress.close()

    def _set_postfix(self) -> None:
        if not self.enabled:
            return
        self.progress.set_postfix_str(f"tasks={self.completed_tasks}/{self.total_tasks}", refresh=False)


def sha1sum_file(path: str, read_bytes: int = CHECKSUM_READ_BYTES) -> str:
    digest = hashlib.sha1()
    buffer = bytearray(read_bytes)
    view = memoryview(buffer)
    with open(path, "rb", buffering=0) as f:
        while True:
            n = f.readinto(buffer)
            if not n:
                break
            digest.update(view[:n])
    return digest.hexdigest()


def write_shard_checksum(staging_dir: str, shard_filename: str, shard_bytes: int) -> str:
    checksum_dir = os.path.join(staging_dir, CHECKSUM_DIRNAME)
    os.makedirs(checksum_dir, exist_ok=True)
    checksum_filename = os.path.join(CHECKSUM_DIRNAME, f"{shard_filename}.sha1.json")
    checksum_path = os.path.join(staging_dir, checksum_filename)
    tmp_path = f"{checksum_path}.tmp-{os.getpid()}"
    payload = {
        "filename": shard_filename,
        "algorithm": "sha1",
        "sha1": sha1sum_file(os.path.join(staging_dir, shard_filename)),
        "bytes": os.path.getsize(os.path.join(staging_dir, shard_filename)),
        "tensor_bytes": shard_bytes,
    }
    with open(tmp_path, "w") as f:
        json.dump(payload, f, sort_keys=True)
    os.replace(tmp_path, checksum_path)
    return checksum_filename


class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = make_storage_meta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class ChunkedStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    def __init__(self, keys_to_load: set[str]):
        super().__init__()
        self.keys_to_load = keys_to_load

    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        if metadata is None:
            raise ValueError("DCP metadata is required")
        for key, value in metadata.state_dict_metadata.items():
            if key not in self.keys_to_load:
                continue
            if isinstance(value, dist_cp.metadata.TensorStorageMetadata):
                value = torch.empty(value.size, dtype=value.properties.dtype)  # type: ignore[assignment]
            state_dict[key] = value
        super().set_up_planner(state_dict, metadata, is_coordinator)


class AccountingStorageReader(WrappedStorageReader):
    def __init__(self, path: str):
        super().__init__(path)
        self.read_items = 0
        self.files = 0
        self.storage_bytes = 0

    @override
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        self.read_items, self.files, self.storage_bytes = compute_dcp_load_accounting(self.storage_data, plan)
        per_file: dict[str, list[Any]] = {}
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            per_file.setdefault(item_md.relative_path, []).append(read_item)

        for relative_path, reqs in per_file.items():
            new_path = self.fs.concat_path(self.path, relative_path)
            with self.fs.create_stream(new_path, "rb") as stream:
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = cast(io.IOBase, _create_file_view(stream, item_md.offset, item_md.length))
                    transform_from = self.transforms.transform_load_stream(
                        req,
                        item_md.transform_descriptors or (),
                        file_slice,
                    )

                    if req.type == LoadItemType.BYTE_IO:
                        read_bytes = io.BytesIO(transform_from.read(-1))
                        read_bytes.seek(0)
                        planner.load_bytes(req, read_bytes)
                        continue

                    seekable = transform_from if transform_from.seekable() else io.BytesIO(transform_from.read(-1))
                    seekable.seek(0)
                    tensor = cast(torch.Tensor, torch.load(seekable, map_location="cpu", weights_only=True))
                    tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
                    target_tensor = planner.resolve_tensor(req).detach()
                    if target_tensor.size() != tensor.size():
                        raise AssertionError(
                            f"DCP tensor size mismatch for {req.storage_index}: "
                            f"{target_tensor.size()} vs {tensor.size()}"
                        )
                    target_tensor.copy_(tensor)
                    planner.commit_tensor(req, target_tensor)

        fut: Future[None] = Future()
        fut.set_result(None)
        return fut


def make_storage_meta():
    storage_meta = getattr(dist_cp, "StorageMeta", None)
    if storage_meta is not None:
        return storage_meta()
    return dist_cp.metadata.StorageMeta()


def compute_dcp_load_accounting(storage_data: dict[Any, Any], plan: LoadPlan) -> tuple[int, int, int]:
    files = set()
    storage_bytes = 0
    for read_item in plan.items:
        item_md = storage_data[read_item.storage_index]
        files.add(item_md.relative_path)
        storage_bytes += int(item_md.length)
    return len(plan.items), len(files), storage_bytes


def prepare_cached_metadata_for_reader(
    metadata: dist_cp.metadata.Metadata,
    storage_reader: WrappedStorageReader,
) -> dist_cp.metadata.Metadata:
    if getattr(metadata, "storage_meta", None) is None:
        metadata.storage_meta = make_storage_meta()
    metadata.storage_meta.load_id = storage_reader.load_id
    if metadata.planner_data is None:
        metadata.planner_data = {}
    return metadata


def load_tensor_chunk(
    input_dir: str,
    keys_to_load: set[str],
    metadata: dist_cp.metadata.Metadata,
) -> DcpLoadResult:
    state_dict: dict[str, torch.Tensor] = {}
    storage_reader = AccountingStorageReader(input_dir)
    metadata = prepare_cached_metadata_for_reader(metadata, storage_reader)
    planner = ChunkedStateDictLoadPlanner(keys_to_load)
    planner.set_up_planner(state_dict, metadata, is_coordinator=True)
    storage_reader.set_up_storage_reader(metadata, is_coordinator=True)
    local_plan = planner.create_local_plan()
    local_plan = storage_reader.prepare_local_plan(local_plan)
    global_plan = planner.create_global_plan([local_plan])
    global_plan = storage_reader.prepare_global_plan(global_plan)
    final_local_plan = planner.finish_plan(global_plan[0])
    storage_reader.read_data(final_local_plan, planner).wait()
    return DcpLoadResult(
        state_dict=state_dict,
        read_items=storage_reader.read_items,
        files=storage_reader.files,
        storage_bytes=storage_reader.storage_bytes,
    )


def get_expert_param(args: Any, name: str, param: torch.Tensor):
    if ".experts." not in name:
        yield name, param
        return

    num_experts = args.num_experts
    match = re.search(r"mlp.experts\.(.+)\.weight(\d+)", name)
    if not match:
        if param.shape[0] != num_experts:
            raise AssertionError(f"Expected {num_experts} experts for {name}, got {param.shape}")
        for expert_id in range(num_experts):
            expert_name = name.replace(".experts.experts.", ".experts.") + str(expert_id)
            yield expert_name, param[expert_id]
    else:
        yield name, param


def get_layer_param(args: Any, name: str, param: torch.Tensor):
    if ".layers." not in name:
        yield name, param
        return

    num_layers = args.num_layers
    match = re.search(r"\.layers\.(\d+)\.", name)
    if not match:
        if param.shape[0] != num_layers:
            raise AssertionError(f"Expected {num_layers} layers for {name}, got {param.shape}")
        for layer_id in range(num_layers):
            layer_name = name.replace(".layers.", f".layers.{layer_id}.")
            yield from get_expert_param(args, layer_name, param[layer_id])
    else:
        yield from get_expert_param(args, name, param)


def get_named_params(args: Any, state_dict: dict[str, torch.Tensor]):
    for name, param in state_dict.items():
        yield from get_layer_param(args, f"module.module.{name}", param)


_MOE_EXPERT_KEY_RE = re.compile(
    r"^(?P<prefix>language_model\.)?decoder\.layers\.(?P<layer>\d+)\."
    r"mlp\.experts\.experts\.(?P<linear>linear_fc[12])\.weight$"
)


def parse_moe_expert_key(source_key: str) -> tuple[int, str, str] | None:
    match = _MOE_EXPERT_KEY_RE.match(source_key)
    if not match:
        return None
    hf_prefix = "language_model." if match.group("prefix") else ""
    return int(match.group("layer")), match.group("linear"), hf_prefix


def is_supported_moe_read_item(read_item: Any, source_key: str, tensor_size: torch.Size) -> bool:
    parsed = parse_moe_expert_key(source_key)
    if parsed is None or len(tensor_size) != 3:
        return False
    _, linear_name, _ = parsed

    offsets = tuple(int(value) for value in read_item.storage_index.offset)
    lengths = tuple(int(value) for value in read_item.lengths)
    if len(offsets) != 3 or len(lengths) != 3:
        return False

    expert_offset, ffn_offset, hidden_offset = offsets
    expert_count, ffn_length, hidden_length = lengths
    num_experts, ffn_size, hidden_size = (int(value) for value in tensor_size)
    if expert_offset < 0 or expert_offset + expert_count > num_experts:
        return False
    if hidden_offset != 0 or hidden_length != hidden_size:
        return False
    if linear_name == "linear_fc2":
        return ffn_offset == 0 and ffn_length == ffn_size
    if ffn_size % 2 != 0:
        return False
    half_ffn = ffn_size // 2
    return (ffn_offset == 0 and ffn_length == ffn_size) or (ffn_offset in {0, half_ffn} and ffn_length == half_ffn)


def create_full_tensor_read_items(source_key: str, metadata: dist_cp.metadata.Metadata) -> list[Any]:
    md = metadata.state_dict_metadata[source_key]
    if not isinstance(md, dist_cp.metadata.TensorStorageMetadata):
        raise TypeError(f"{source_key} is not tensor metadata")
    return create_read_items_for_chunk_list(source_key, md, list(md.chunks))


def create_direct_moe_read_items(block: MoeBlockSpec, md: dist_cp.metadata.TensorStorageMetadata) -> list[ReadItem]:
    chunk_by_index = {
        MetadataIndex(block.source_key, chunk.offsets, idx): chunk for idx, chunk in enumerate(md.chunks)
    }
    read_items: list[ReadItem] = []
    for storage_index in block.storage_indices:
        chunk = chunk_by_index[storage_index]
        read_items.append(
            ReadItem(
                type=LoadItemType.TENSOR,
                dest_index=MetadataIndex(block.source_key),
                dest_offsets=chunk.offsets,
                storage_index=storage_index,
                storage_offsets=torch.Size([0 for _ in chunk.offsets]),
                lengths=chunk.sizes,
            )
        )
    read_items.sort(key=lambda item: tuple(item.storage_index.offset))
    return read_items


def create_moe_block_specs(source_key: str, metadata: dist_cp.metadata.Metadata) -> list[MoeBlockSpec] | None:
    parsed = parse_moe_expert_key(source_key)
    if parsed is None:
        return None
    layer_idx, linear_name, hf_prefix = parsed
    md = metadata.state_dict_metadata[source_key]
    if not isinstance(md, dist_cp.metadata.TensorStorageMetadata):
        return None
    read_items = create_full_tensor_read_items(source_key, metadata)
    if not read_items or any(not is_supported_moe_read_item(item, source_key, md.size) for item in read_items):
        return None
    if metadata.storage_data is None:
        return None

    by_file: dict[str, list[MetadataIndex]] = {}
    for read_item in read_items:
        item_md = metadata.storage_data[read_item.storage_index]
        by_file.setdefault(item_md.relative_path, []).append(read_item.storage_index)

    return [
        MoeBlockSpec(
            source_key=source_key,
            relative_path=relative_path,
            storage_indices=tuple(sorted(indices, key=lambda index: tuple(index.offset))),
            layer_idx=layer_idx,
            linear_name=linear_name,
            hf_prefix=hf_prefix,
        )
        for relative_path, indices in sorted(by_file.items())
    ]


def expert_source_name(source_key: str, expert_id: int) -> str:
    return "module.module." + source_key.replace(".experts.experts.", ".experts.") + str(expert_id)


def hf_expert_weight_name(hf_prefix: str, layer_idx: int, expert_id: int, projection: str) -> str:
    return f"{hf_prefix}model.layers.{layer_idx}.mlp.experts.{expert_id}.{projection}.weight"


def contiguous_if_needed(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def converted_moe_tensors_from_chunk(
    source_key: str,
    layer_idx: int,
    linear_name: str,
    hf_prefix: str,
    read_item: Any,
    tensor: torch.Tensor,
    tensor_size: torch.Size,
) -> list[PreparedTensorGroup]:
    offsets = tuple(int(value) for value in read_item.storage_index.offset)
    lengths = tuple(int(value) for value in read_item.lengths)
    expert_offset = offsets[0]
    expert_count = lengths[0]
    if tensor.shape[0] != expert_count:
        raise AssertionError(f"Expected {expert_count} experts in {source_key} chunk, got {tensor.shape}")

    groups: list[PreparedTensorGroup] = []
    if linear_name == "linear_fc2":
        for local_expert_idx in range(expert_count):
            expert_id = expert_offset + local_expert_idx
            groups.append(
                PreparedTensorGroup(
                    source_name=expert_source_name(source_key, expert_id),
                    tensors=(
                        (
                            hf_expert_weight_name(hf_prefix, layer_idx, expert_id, "down_proj"),
                            contiguous_if_needed(tensor[local_expert_idx]),
                        ),
                    ),
                )
            )
        return groups

    half_ffn = int(tensor_size[1]) // 2
    second_dim_offset = offsets[1]
    second_dim_length = lengths[1]
    for local_expert_idx in range(expert_count):
        expert_id = expert_offset + local_expert_idx
        expert_tensor = tensor[local_expert_idx]
        if second_dim_offset == 0 and second_dim_length == int(tensor_size[1]):
            gate_weight, up_weight = expert_tensor.chunk(2, dim=0)
            named_tensors = (
                (
                    hf_expert_weight_name(hf_prefix, layer_idx, expert_id, "gate_proj"),
                    contiguous_if_needed(gate_weight),
                ),
                (hf_expert_weight_name(hf_prefix, layer_idx, expert_id, "up_proj"), contiguous_if_needed(up_weight)),
            )
        elif second_dim_offset == 0 and second_dim_length == half_ffn:
            named_tensors = (
                (
                    hf_expert_weight_name(hf_prefix, layer_idx, expert_id, "gate_proj"),
                    contiguous_if_needed(expert_tensor),
                ),
            )
        elif second_dim_offset == half_ffn and second_dim_length == half_ffn:
            named_tensors = (
                (
                    hf_expert_weight_name(hf_prefix, layer_idx, expert_id, "up_proj"),
                    contiguous_if_needed(expert_tensor),
                ),
            )
        else:
            raise AssertionError(
                f"Unsupported {linear_name} chunk for {source_key}: "
                f"offsets={offsets}, lengths={lengths}, size={tuple(tensor_size)}"
            )
        groups.append(
            PreparedTensorGroup(source_name=expert_source_name(source_key, expert_id), tensors=named_tensors)
        )
    return groups


def load_moe_block_direct(
    input_dir: str,
    block: MoeBlockSpec,
    metadata: dist_cp.metadata.Metadata,
) -> DirectMoeLoadResult:
    md = metadata.state_dict_metadata[block.source_key]
    if not isinstance(md, dist_cp.metadata.TensorStorageMetadata):
        raise TypeError(f"{block.source_key} is not tensor metadata")

    storage_reader = AccountingStorageReader(input_dir)
    metadata = prepare_cached_metadata_for_reader(metadata, storage_reader)
    storage_reader.set_up_storage_reader(metadata, is_coordinator=True)
    read_items = create_direct_moe_read_items(block, md)
    read_count, file_count, storage_bytes = compute_dcp_load_accounting(
        storage_reader.storage_data, LoadPlan(read_items)
    )

    tensor_groups: list[PreparedTensorGroup] = []
    new_path = storage_reader.fs.concat_path(storage_reader.path, block.relative_path)
    with storage_reader.fs.create_stream(new_path, "rb") as stream:
        for read_item in read_items:
            item_md = storage_reader.storage_data[read_item.storage_index]
            file_slice = cast(io.IOBase, _create_file_view(stream, item_md.offset, item_md.length))
            transform_from = storage_reader.transforms.transform_load_stream(
                read_item,
                item_md.transform_descriptors or (),
                file_slice,
            )
            seekable = transform_from if transform_from.seekable() else io.BytesIO(transform_from.read(-1))
            seekable.seek(0)
            tensor = cast(torch.Tensor, torch.load(seekable, map_location="cpu", weights_only=True))
            tensor = narrow_tensor_by_index(tensor, read_item.storage_offsets, read_item.lengths)
            tensor_groups.extend(
                converted_moe_tensors_from_chunk(
                    block.source_key,
                    block.layer_idx,
                    block.linear_name,
                    block.hf_prefix,
                    read_item,
                    tensor,
                    md.size,
                )
            )

    return DirectMoeLoadResult(tuple(tensor_groups), read_count, file_count, storage_bytes)


def tensor_metadata_from_checkpoint_metadata(
    metadata: dist_cp.metadata.Metadata,
) -> dict[str, tuple[torch.Size, torch.dtype]]:
    tensor_metadata = {}
    for key, value in metadata.state_dict_metadata.items():
        if "optimizer" in key or "_state" in key:
            continue
        if isinstance(value, dist_cp.metadata.TensorStorageMetadata):
            tensor_metadata[key] = (value.size, value.properties.dtype)
    return tensor_metadata


def tensor_nbytes(shape: torch.Size, dtype: torch.dtype) -> int:
    element_bits = torch.finfo(dtype).bits if dtype.is_floating_point else torch.iinfo(dtype).bits
    return shape.numel() * (element_bits // 8)


def filter_tensor_metadata(
    tensor_metadata: dict[str, tuple[torch.Size, torch.dtype]],
    source_key_regex: str | None,
) -> dict[str, tuple[torch.Size, torch.dtype]]:
    if not source_key_regex:
        return tensor_metadata
    pattern = re.compile(source_key_regex)
    return {key: value for key, value in tensor_metadata.items() if pattern.search(key)}


def _mla_pair_group(key: str) -> str:
    return key.replace("self_attention.linear_q_down_proj.weight", "self_attention.MLA_A_PAIR.weight").replace(
        "self_attention.linear_kv_down_proj.weight",
        "self_attention.MLA_A_PAIR.weight",
    )


def group_small_tasks(
    atomic_tasks: list[tuple[int, tuple[str, ...]]],
    task_group_bytes: int,
) -> list[tuple[int, tuple[str, ...]]]:
    if task_group_bytes <= 0:
        return atomic_tasks

    large_tasks = [task for task in atomic_tasks if task[0] >= task_group_bytes]
    small_tasks = [task for task in atomic_tasks if task[0] < task_group_bytes]
    small_tasks.sort(key=lambda item: (-item[0], item[1]))
    grouped_tasks: list[tuple[int, tuple[str, ...]]] = []
    current_keys: list[str] = []
    current_bytes = 0
    for estimated_bytes, keys in small_tasks:
        if current_keys and current_bytes + estimated_bytes > task_group_bytes:
            grouped_tasks.append((current_bytes, tuple(sorted(current_keys))))
            current_keys = []
            current_bytes = 0
        current_keys.extend(keys)
        current_bytes += estimated_bytes
    if current_keys:
        grouped_tasks.append((current_bytes, tuple(sorted(current_keys))))
    return large_tasks + grouped_tasks


def plan_whole_source_tasks(
    tensor_metadata: dict[str, tuple[torch.Size, torch.dtype]],
    q_lora_rank: int | None,
    task_group_bytes: int,
) -> list[TaskSpec]:
    grouped: dict[str, list[str]] = {}
    for key in tensor_metadata:
        group = _mla_pair_group(key) if q_lora_rank is not None else key
        grouped.setdefault(group, []).append(key)

    atomic_tasks = []
    for keys in grouped.values():
        sorted_keys = tuple(sorted(keys))
        estimated_bytes = sum(tensor_nbytes(tensor_metadata[key][0], tensor_metadata[key][1]) for key in sorted_keys)
        atomic_tasks.append((estimated_bytes, sorted_keys))
    raw_tasks = group_small_tasks(atomic_tasks, task_group_bytes)
    raw_tasks.sort(key=lambda item: (-item[0], item[1]))
    return [TaskSpec(idx, keys, estimated_bytes) for idx, (estimated_bytes, keys) in enumerate(raw_tasks)]


def collect_moe_blocks_by_file(
    tensor_metadata: dict[str, tuple[torch.Size, torch.dtype]],
    metadata: dist_cp.metadata.Metadata,
) -> tuple[list[tuple[int, MoeBlockSpec]], dict[str, tuple[torch.Size, torch.dtype]]]:
    moe_blocks: list[tuple[int, MoeBlockSpec]] = []
    whole_source_metadata = dict(tensor_metadata)
    if metadata.storage_data is None:
        return moe_blocks, whole_source_metadata

    for source_key in sorted(tensor_metadata):
        block_specs = create_moe_block_specs(source_key, metadata)
        if block_specs is None:
            continue
        for block in block_specs:
            estimated_bytes = sum(int(metadata.storage_data[index].length) for index in block.storage_indices)
            moe_blocks.append((estimated_bytes, block))
        del whole_source_metadata[source_key]
    return moe_blocks, whole_source_metadata


def group_moe_block_tasks(moe_blocks: list[tuple[int, MoeBlockSpec]], task_group_bytes: int) -> list[TaskSpec]:
    if not moe_blocks:
        return []
    target_bytes = task_group_bytes or DEFAULT_DIRECT_MOE_GROUP_SIZE
    moe_blocks.sort(
        key=lambda item: (
            item[1].source_key,
            item[1].relative_path,
            tuple(item[1].storage_indices[0].offset) if item[1].storage_indices else (),
        )
    )

    tasks: list[TaskSpec] = []
    current_blocks: list[MoeBlockSpec] = []
    current_bytes = 0

    def flush_current() -> None:
        nonlocal current_blocks, current_bytes
        if not current_blocks:
            return
        tasks.append(
            TaskSpec(
                task_id=-1,
                keys=tuple(sorted({block.source_key for block in current_blocks})),
                estimated_source_bytes=current_bytes,
                moe_blocks=tuple(current_blocks),
            )
        )
        current_blocks = []
        current_bytes = 0

    for estimated_bytes, block in moe_blocks:
        if current_blocks and current_bytes + estimated_bytes > target_bytes:
            flush_current()
        current_blocks.append(block)
        current_bytes += estimated_bytes
    flush_current()
    return tasks


def plan_conversion_tasks(
    tensor_metadata: dict[str, tuple[torch.Size, torch.dtype]],
    metadata: dist_cp.metadata.Metadata,
    q_lora_rank: int | None,
    task_group_bytes: int,
) -> list[TaskSpec]:
    moe_blocks, whole_source_metadata = collect_moe_blocks_by_file(tensor_metadata, metadata)
    tasks = group_moe_block_tasks(moe_blocks, task_group_bytes)
    tasks.extend(
        TaskSpec(-1, task.keys, task.estimated_source_bytes)
        for task in plan_whole_source_tasks(whole_source_metadata, q_lora_rank, task_group_bytes)
    )
    tasks.sort(
        key=lambda task: (
            -task.estimated_source_bytes,
            task.moe_blocks[0].source_key if task.moe_blocks else task.keys[0],
            task.moe_blocks[0].relative_path if task.moe_blocks else "",
            task.keys,
        )
    )
    return [TaskSpec(idx, task.keys, task.estimated_source_bytes, task.moe_blocks) for idx, task in enumerate(tasks)]


def summarize_plan(tasks: list[TaskSpec], model_name: str, concurrency: int, output_dir: str) -> dict[str, Any]:
    source_bytes = [task.estimated_source_bytes for task in tasks]
    direct_moe_tasks = [task for task in tasks if task.moe_blocks]
    return {
        "model_name": model_name,
        "concurrency": concurrency,
        "output_dir": output_dir,
        "tasks": len(tasks),
        "source_keys": sum(len(task.keys) for task in tasks),
        "direct_moe_tasks": len(direct_moe_tasks),
        "direct_moe_blocks": sum(len(task.moe_blocks) for task in direct_moe_tasks),
        "estimated_source_bytes": sum(source_bytes),
        "largest_task_bytes": max(source_bytes, default=0),
        "smallest_task_bytes": min(source_bytes, default=0),
        "top_tasks": [
            {
                "task_id": task.task_id,
                "keys": task.keys,
                "estimated_source_bytes": task.estimated_source_bytes,
                "moe_blocks": len(task.moe_blocks),
            }
            for task in tasks[:10]
        ],
    }


def load_hf_config(origin_hf_dir: str | None) -> Any | None:
    if origin_hf_dir is None:
        return None
    return AutoConfig.from_pretrained(origin_hf_dir, trust_remote_code=True)


def load_quantization_config(hf_config: Any | None) -> dict[str, Any] | None:
    if hf_config is None:
        return None
    quantization_config = getattr(hf_config, "quantization_config", None)
    if quantization_config is not None:
        return dict(quantization_config)
    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None:
        nested = getattr(text_config, "quantization_config", None)
        if nested is not None:
            return dict(nested)
    return None


def get_hf_vocab_size(hf_config: Any | None) -> int | None:
    if hf_config is None:
        return None
    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None and hasattr(text_config, "vocab_size"):
        return int(text_config.vocab_size)
    vocab_size = getattr(hf_config, "vocab_size", None)
    return int(vocab_size) if vocab_size is not None else None


def copy_assets(origin_hf_dir: str | None, output_dir: str) -> None:
    if origin_hf_dir is None:
        return
    for filename in os.listdir(origin_hf_dir):
        if filename == "model.safetensors.index.json" or filename.endswith(".safetensors"):
            continue
        src = os.path.join(origin_hf_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, filename))


def _flush_shard(
    staging_dir: str,
    task_id: int,
    shard_idx: int,
    tensors: dict[str, torch.Tensor],
    current_size: int,
    sha1sum_output: bool,
) -> ShardManifest:
    filename = f"worker-{socket.gethostname()}-task-{task_id:05d}-shard-{shard_idx:05d}.safetensors"
    safetensors.torch.save_file(tensors, os.path.join(staging_dir, filename))
    checksum_filename = write_shard_checksum(staging_dir, filename, current_size) if sha1sum_output else None
    return ShardManifest(filename, None, tuple(tensors.keys()), current_size, len(tensors), checksum_filename)


def append_to_shards(
    staging_dir: str,
    task_id: int,
    shard_idx: int,
    current_tensors: dict[str, torch.Tensor],
    current_size: int,
    converted_named_tensors: tuple[tuple[str, torch.Tensor], ...] | list[tuple[str, torch.Tensor]],
    max_file_bytes: int,
    shards: list[ShardManifest],
    sha1sum_output: bool,
) -> tuple[int, int, int]:
    total_size = 0
    for converted_name, converted_param in converted_named_tensors:
        tensor_size = converted_param.numel() * converted_param.element_size()
        if tensor_size + current_size > max_file_bytes and current_tensors:
            shards.append(_flush_shard(staging_dir, task_id, shard_idx, current_tensors, current_size, sha1sum_output))
            shard_idx += 1
            current_tensors.clear()
            current_size = 0
        current_tensors[converted_name] = converted_param
        current_size += tensor_size
        total_size += tensor_size
    return shard_idx, current_size, total_size


def prepare_moe_block_task_tensors(
    task: TaskSpec,
    input_dir: str,
    metadata: dist_cp.metadata.Metadata,
) -> PreparedTaskTensors:
    groups: list[PreparedTensorGroup] = []
    total_read_items = 0
    total_files = 0
    total_storage_bytes = 0
    for block in task.moe_blocks:
        result = load_moe_block_direct(input_dir, block, metadata)
        groups.extend(result.tensor_groups)
        total_read_items += result.read_items
        total_files += result.files
        total_storage_bytes += result.storage_bytes
    return PreparedTaskTensors(tuple(groups), TaskLoadStats(total_read_items, total_files, total_storage_bytes))


def prepare_whole_source_task_tensors(
    task: TaskSpec,
    input_dir: str,
    megatron_args: Any,
    model_name: str,
    metadata: dist_cp.metadata.Metadata,
) -> PreparedTaskTensors:
    load_result = load_tensor_chunk(input_dir, set(task.keys), metadata)
    state_dict = load_result.state_dict

    groups: list[PreparedTensorGroup] = []
    try:
        for name, param in get_named_params(megatron_args, state_dict):
            if getattr(megatron_args, "vocab_size", None) is not None:
                param = m2hf.remove_padding(name, param, megatron_args.vocab_size)
            converted_named_tensors = m2hf._convert_to_hf_core(megatron_args, model_name, name, param)
            groups.append(PreparedTensorGroup(name, tuple(converted_named_tensors)))
        return PreparedTaskTensors(
            tuple(groups),
            TaskLoadStats(load_result.read_items, load_result.files, load_result.storage_bytes),
        )
    finally:
        del state_dict


def write_prepared_tensor_groups(
    staging_dir: str,
    task_id: int,
    groups: tuple[PreparedTensorGroup, ...],
    megatron_args: Any,
    quantization_config: dict[str, Any] | None,
    max_file_bytes: int,
    cuda_device_id: int | None,
    sha1sum_output: bool,
) -> tuple[tuple[ShardManifest, ...], int]:
    current_tensors: dict[str, torch.Tensor] = {}
    current_size = 0
    shard_idx = 0
    shards: list[ShardManifest] = []
    total_size = 0

    for group in groups:
        converted_named_tensors = group.tensors
        if quantization_config is not None:
            if cuda_device_id is not None:
                torch.cuda.set_device(cuda_device_id)
            converted_named_tensors = tuple(
                m2hf.quantize_params(
                    megatron_args, group.source_name, list(converted_named_tensors), quantization_config
                )
            )
        shard_idx, current_size, added_size = append_to_shards(
            staging_dir,
            task_id,
            shard_idx,
            current_tensors,
            current_size,
            converted_named_tensors,
            max_file_bytes,
            shards,
            sha1sum_output,
        )
        total_size += added_size

    if current_tensors:
        shards.append(_flush_shard(staging_dir, task_id, shard_idx, current_tensors, current_size, sha1sum_output))
    return tuple(shards), total_size


def assign_cuda_device_id(actor_id: int, device_count: int) -> int:
    if device_count <= 0:
        raise RuntimeError("Quantization requires CUDA, but no CUDA devices are visible")
    return actor_id % device_count


def initialize_worker_cuda_device(actor_id: int, quantization_config: dict[str, Any] | None) -> int | None:
    if quantization_config is None:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("Quantization requires CUDA, but CUDA is unavailable")
    cuda_device_id = assign_cuda_device_id(actor_id, torch.cuda.device_count())
    torch.cuda.set_device(cuda_device_id)
    return cuda_device_id


class ConversionWorker:
    def __init__(
        self,
        actor_id: int,
        input_dir: str,
        staging_dir: str,
        megatron_args: Any,
        model_name: str,
        quantization_config: dict[str, Any] | None,
        max_file_bytes: int,
        sha1sum_output: bool,
        metadata_ref: Any,
    ) -> None:
        self.actor_id = actor_id
        self.ray_node_id = ray.get_runtime_context().get_node_id()
        self.cuda_device_id = initialize_worker_cuda_device(actor_id, quantization_config)
        self.input_dir = input_dir
        self.staging_dir = staging_dir
        self.megatron_args = megatron_args
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.max_file_bytes = max_file_bytes
        self.sha1sum_output = sha1sum_output
        self.metadata = metadata_ref if isinstance(metadata_ref, dist_cp.metadata.Metadata) else ray.get(metadata_ref)

    def convert(self, task: TaskSpec) -> TaskResult:
        node = socket.gethostname()
        pid = os.getpid()
        if task.moe_blocks:
            prepared = prepare_moe_block_task_tensors(task, self.input_dir, self.metadata)
        else:
            prepared = prepare_whole_source_task_tensors(
                task, self.input_dir, self.megatron_args, self.model_name, self.metadata
            )
        shards, total_size = write_prepared_tensor_groups(
            self.staging_dir,
            task.task_id,
            prepared.groups,
            self.megatron_args,
            self.quantization_config,
            self.max_file_bytes,
            self.cuda_device_id,
            self.sha1sum_output,
        )
        return TaskResult(
            task_id=task.task_id,
            actor_id=self.actor_id,
            node=node,
            pid=pid,
            shards=shards,
            source_bytes=task.estimated_source_bytes,
            output_bytes=total_size,
            weights=sum(len(shard.weight_keys) for shard in shards),
            source_keys=task.keys,
            dcp_read_items=prepared.load_stats.read_items,
            dcp_files=prepared.load_stats.files,
            dcp_storage_bytes=prepared.load_stats.storage_bytes,
            cuda_device_id=self.cuda_device_id,
            ray_node_id=self.ray_node_id,
        )


def initialize_ray() -> None:
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    if ray.is_initialized():
        return
    try:
        ray.init(address="auto", ignore_reinit_error=True)
    except ConnectionError:
        ray.init(ignore_reinit_error=True)


def live_ray_node_ids() -> list[str]:
    node_ids = [str(node["NodeID"]) for node in ray.nodes() if node.get("Alive", False)]
    node_ids.sort()
    if not node_ids:
        raise RuntimeError("Ray has no live nodes")
    return node_ids


def make_conversion_actor():
    return ray.remote(
        num_cpus=0,
        num_gpus=0,
        runtime_env={
            "env_vars": {
                "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            }
        },
    )(ConversionWorker)


def collect_ray_results(
    tasks: list[TaskSpec],
    input_dir: str,
    staging_dir: str,
    megatron_args: Any,
    model_name: str,
    quantization_config: dict[str, Any] | None,
    max_file_bytes: int,
    concurrency: int,
    metadata_ref: Any,
    sha1sum_output: bool,
    progress: bool,
    progress_interval_seconds: float,
) -> list[TaskResult]:
    worker_count = min(concurrency, len(tasks))
    if worker_count < 1:
        return []

    worker_cls = make_conversion_actor()
    node_ids = live_ray_node_ids()
    workers = []
    for actor_id in range(worker_count):
        node_id = node_ids[actor_id % len(node_ids)]
        actor_cls = worker_cls.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False))
        workers.append(
            actor_cls.remote(
                actor_id,
                input_dir,
                staging_dir,
                megatron_args,
                model_name,
                quantization_config,
                max_file_bytes,
                sha1sum_output,
                metadata_ref,
            )
        )

    pending: dict[Any, int] = {}
    submitted = 0
    results: list[TaskResult] = []
    progress_reporter = ProgressReporter(tasks, progress, progress_interval_seconds)
    progress_reporter.tick()
    for worker_idx, worker in enumerate(workers):
        if submitted >= len(tasks):
            break
        pending[worker.convert.remote(tasks[submitted])] = worker_idx
        submitted += 1

    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1, timeout=progress_reporter.interval_seconds)
        if not ready:
            progress_reporter.tick()
            continue
        ready_ref = ready[0]
        worker_idx = pending.pop(ready_ref)
        result = ray.get(ready_ref)
        results.append(result)
        progress_reporter.complete(result)
        if not progress:
            print(
                f"task {result.task_id} finished on {result.node}: "
                f"{result.output_bytes / 1e9:.2f} GB output, {result.weights} tensors"
            )
        if submitted < len(tasks):
            pending[workers[worker_idx].convert.remote(tasks[submitted])] = worker_idx
            submitted += 1

    progress_reporter.finish()
    results.sort(key=lambda result: result.task_id)
    return results


def plan_global_shards(task_results: list[TaskResult]) -> tuple[tuple[PlannedShard, ...], dict[str, Any]]:
    all_shards = [shard for result in sorted(task_results, key=lambda item: item.task_id) for shard in result.shards]
    if not all_shards:
        raise ValueError("No HF tensor shards were emitted")
    weight_map: dict[str, str] = {}
    planned_shards: list[PlannedShard] = []
    total_size = 0
    for idx, shard in enumerate(all_shards):
        final_filename = f"model-{idx:05d}-of-{len(all_shards):05d}.safetensors"
        for key in shard.weight_keys:
            if key in weight_map:
                raise ValueError(f"Duplicate HF tensor emitted during finalization: {key}")
            weight_map[key] = final_filename
        planned_shards.append(
            PlannedShard(shard.temp_filename, final_filename, shard.weight_keys, shard.bytes, shard.checksum_filename)
        )
        total_size += shard.bytes
    return tuple(planned_shards), {"metadata": {"total_size": total_size}, "weight_map": weight_map}


def merge_checksum_files(staging_dir: str, output_dir: str, planned_shards: tuple[PlannedShard, ...]) -> None:
    checksums: dict[str, dict[str, Any]] = {}
    for shard in planned_shards:
        if shard.checksum_filename is None:
            raise ValueError(f"Missing checksum file for {shard.temp_filename}")
        checksum_path = os.path.join(staging_dir, shard.checksum_filename)
        with open(checksum_path) as f:
            checksum = json.load(f)
        if checksum.get("filename") != shard.temp_filename:
            raise ValueError(f"Checksum filename mismatch in {checksum_path}")
        checksums[shard.final_filename] = {
            "sha1": checksum["sha1"],
            "bytes": checksum["bytes"],
            "tensor_bytes": shard.bytes,
        }

    output_path = os.path.join(output_dir, "checksum.json")
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump({"algorithm": "sha1", "files": checksums}, f, indent=2, sort_keys=True)
    os.replace(tmp_path, output_path)


def finalize_output(
    staging_dir: str,
    output_dir: str,
    origin_hf_dir: str | None,
    task_results: list[TaskResult],
    sha1sum_output: bool,
) -> None:
    planned_shards, index = plan_global_shards(task_results)
    copy_assets(origin_hf_dir, output_dir)
    for shard in planned_shards:
        os.replace(os.path.join(staging_dir, shard.temp_filename), os.path.join(output_dir, shard.final_filename))
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    if sha1sum_output:
        merge_checksum_files(staging_dir, output_dir, planned_shards)
    shutil.rmtree(staging_dir)


def reject_cloud_path(path: str, label: str) -> None:
    if "://" in path:
        raise ValueError(f"{label} must be a local filesystem path, got {path}")


def prepare_output_dir(output_dir: str, force: bool) -> str:
    reject_cloud_path(output_dir, "output_dir")
    if os.path.exists(output_dir):
        if not force:
            raise FileExistsError(f"{output_dir} exists; pass --force to overwrite")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    staging_dir = os.path.join(output_dir, ".ray-convert-staging")
    os.makedirs(staging_dir, exist_ok=True)
    return staging_dir


def load_megatron_args(input_dir: str, model_name_override: str | None, vocab_size: int | None) -> tuple[Any, str]:
    megatron_args = torch.load(os.path.join(input_dir, "common.pt"), weights_only=False)["args"]
    model_name = model_name_override or getattr(megatron_args, "original_hf_model_name", None)
    if model_name is None:
        raise ValueError("Model name is required when common.pt does not include original_hf_model_name")
    if vocab_size is not None:
        megatron_args.vocab_size = vocab_size
    if not hasattr(megatron_args, "sglang_enable_ep_moe"):
        megatron_args.sglang_enable_ep_moe = False
    return megatron_args, model_name


def read_metadata_and_plan(args: Args, megatron_args: Any) -> tuple[dist_cp.metadata.Metadata, list[TaskSpec]]:
    metadata = WrappedStorageReader(args.input_dir).read_metadata()
    tensor_metadata = tensor_metadata_from_checkpoint_metadata(metadata)
    tensor_metadata = filter_tensor_metadata(tensor_metadata, args.source_key_regex)
    if args.source_key_regex and not tensor_metadata:
        raise ValueError(f"No checkpoint keys matched {args.source_key_regex}")
    tasks = plan_conversion_tasks(
        tensor_metadata, metadata, getattr(megatron_args, "q_lora_rank", None), args.task_group_bytes
    )
    return metadata, tasks


def convert_torch_dist_to_hf_ray(args: Args) -> str:
    reject_cloud_path(args.input_dir, "input_dir")
    if args.origin_hf_dir is not None:
        reject_cloud_path(args.origin_hf_dir, "origin_hf_dir")
    common_pt = os.path.join(args.input_dir, "common.pt")
    if not os.path.exists(common_pt):
        raise FileNotFoundError(f"Expected {common_pt}")

    hf_config = load_hf_config(args.origin_hf_dir)
    vocab_size = get_hf_vocab_size(hf_config)
    megatron_args, model_name = load_megatron_args(args.input_dir, args.model_name, vocab_size)
    quantization_config = load_quantization_config(hf_config)

    metadata, tasks = read_metadata_and_plan(args, megatron_args)
    concurrency = args.concurrency or min(max(len(tasks), 1), 16)
    print(json.dumps(summarize_plan(tasks, model_name, concurrency, args.output_dir), indent=2, default=str))
    if args.dry_run_plan:
        return args.output_dir
    if not tasks:
        raise ValueError("No checkpoint tensor tasks were planned")

    staging_dir = prepare_output_dir(args.output_dir, args.force)
    initialize_ray()
    metadata_ref = ray.put(metadata)
    task_results = collect_ray_results(
        tasks,
        args.input_dir,
        staging_dir,
        megatron_args,
        model_name,
        quantization_config,
        args.max_file_bytes,
        concurrency,
        metadata_ref,
        args.sha1sum_output,
        args.progress,
        args.progress_interval_seconds,
    )
    finalize_output(staging_dir, args.output_dir, args.origin_hf_dir, task_results, args.sha1sum_output)
    return args.output_dir


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--origin-hf-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--max-file-bytes", type=int, default=20 * 1024**3)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--task-group-bytes", type=int, default=0)
    parser.add_argument("--source-key-regex", default=None)
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--sha1sum-output", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
    parser.set_defaults(progress=True)
    ns = parser.parse_args()
    return Args(
        input_dir=ns.input_dir,
        output_dir=ns.output_dir,
        origin_hf_dir=ns.origin_hf_dir,
        model_name=ns.model_name,
        force=ns.force,
        max_file_bytes=ns.max_file_bytes,
        concurrency=ns.concurrency,
        task_group_bytes=ns.task_group_bytes,
        source_key_regex=ns.source_key_regex,
        dry_run_plan=ns.dry_run_plan,
        sha1sum_output=ns.sha1sum_output,
        progress=ns.progress,
        progress_interval_seconds=ns.progress_interval_seconds,
    )


def main() -> None:
    convert_torch_dist_to_hf_ray(parse_args())


if __name__ == "__main__":
    main()
