"""Slice-based HF initialization for Miles offline checkpoint conversion."""

import json
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core.distributed import DistributedDataParallel
from megatron.core.extensions.transformer_engine import TEDotProductAttention, TEGroupedLinear, TELinear
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.module import Float16Module
from safetensors import safe_open

from miles_plugins.models.muse_spark.config import parse_muse_spark_1x_config
from miles_plugins.models.muse_spark.model import MuseSpark1xGPTModel
from miles_plugins.models.muse_spark.provider import validate_model_config
from miles_plugins.models.muse_spark.weight_mapping import (
    NON_TEXT_PREFIXES,
    TensorSpec,
    build_mappings,
    hf_tensor_specs,
)


def _read_shard_metadata(shard_path):
    tensor_specs = {}
    with safe_open(shard_path, framework="pt", device="cpu") as shard_reader:
        for tensor_name in shard_reader.keys():
            tensor_slice = shard_reader.get_slice(tensor_name)
            dtype = {"BF16": torch.bfloat16, "F32": torch.float32}.get(tensor_slice.get_dtype())
            if dtype is None:
                raise ValueError(f"unsupported checkpoint dtype for {tensor_name}: {tensor_slice.get_dtype()}")
            tensor_specs[tensor_name] = TensorSpec(tuple(tensor_slice.get_shape()), dtype)
    return tensor_specs


class SafetensorsCheckpoint:
    """Read safetensors metadata and local tensor slices."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.config = json.loads((self.path / "config.json").read_text())
        index_path = self.path / "model.safetensors.index.json"
        if index_path.is_file():
            checkpoint_index = json.loads(index_path.read_text())
            self.weight_map = checkpoint_index.get("weight_map")
            if not isinstance(self.weight_map, dict) or not self.weight_map:
                raise ValueError("checkpoint index requires a nonempty weight_map")
        else:
            single_shard_path = self.path / "model.safetensors"
            if not single_shard_path.is_file():
                raise ValueError("checkpoint requires a safetensors index or model.safetensors")
            self.weight_map = None
        shard_filenames = set(self.weight_map.values()) if self.weight_map else {"model.safetensors"}
        if any(not isinstance(shard_filename, str) for shard_filename in shard_filenames):
            raise ValueError("safetensors shard names must be strings")
        self.tensor_metadata = {}
        shard_weight_map = {}
        for shard_filename in sorted(shard_filenames):
            shard_path = (self.path / shard_filename).resolve()
            if not shard_path.is_relative_to(self.path) or not shard_path.is_file():
                raise ValueError(f"invalid safetensors shard path: {shard_filename}")
            shard_specs = _read_shard_metadata(shard_path)
            duplicate_names = set(shard_specs) & set(shard_weight_map)
            if duplicate_names:
                raise ValueError(f"duplicate tensors across safetensors shards: {sorted(duplicate_names)}")
            shard_weight_map.update(dict.fromkeys(shard_specs, shard_filename))
            self.tensor_metadata.update(shard_specs)
        if self.weight_map is not None and self.weight_map != shard_weight_map:
            raise ValueError("checkpoint index and safetensors headers have different tensor coverage")
        self.weight_map = shard_weight_map

    def read_slice(self, name, selection):
        with safe_open(self.path / self.weight_map[name], framework="pt", device="cpu") as shard_reader:
            return shard_reader.get_slice(name)[selection]


def _unwrap_model(model):
    if isinstance(model, (list, tuple)):
        if len(model) != 1:
            raise ValueError("Spark checkpoint loading requires exactly one local pipeline stage")
        model = model[0]
    wrappers = (Float16Module, DistributedDataParallel, torch.nn.parallel.DistributedDataParallel)
    while isinstance(model, wrappers):
        model = model.module
    if not isinstance(model, MuseSpark1xGPTModel):
        raise ValueError("Spark checkpoint loading requires a MuseSpark1xGPTModel")
    validate_model_config(model.config)
    return model


def _get_megatron_tensor_spec(mapping, hf_specs):
    if mapping.kind == "qgkv":
        specs = [hf_specs[name] for name in mapping.hf_param.values()]
        return TensorSpec((sum(spec.shape[0] for spec in specs), specs[0].shape[1]), specs[0].dtype)
    if mapping.kind == "gated_mlp":
        spec = hf_specs[mapping.hf_param["gate"]]
        return TensorSpec((2 * spec.shape[0], spec.shape[1]), spec.dtype)
    spec = hf_specs[mapping.hf_param]
    if mapping.kind == "transpose":
        return TensorSpec(tuple(reversed(spec.shape)), spec.dtype)
    if mapping.kind == "expert":
        return TensorSpec(spec.shape[1:], spec.dtype)
    return spec


def _prepare_model_weights(model):
    model = _unwrap_model(model)
    spark_config = model.config.spark
    mappings = build_mappings(spark_config, model=model)
    mapped_names = {mapping.megatron_param for mapping in mappings}
    if len(mapped_names) != len(mappings):
        raise ValueError("duplicate Megatron parameter mapping")
    model_state = model.state_dict(keep_vars=True)
    framework_state_names = {
        f"{param_name}._extra_state"
        for param_name, module in model.named_modules()
        if isinstance(
            module,
            (TELinear, TEGroupedLinear, TEDotProductAttention, ColumnParallelLinear),
        )
    }
    target_weights = {}
    for param_name, target_weight in model_state.items():
        if param_name in framework_state_names and (
            target_weight is None
            or (
                isinstance(target_weight, torch.Tensor)
                and target_weight.dtype == torch.uint8
                and target_weight.numel() == 0
            )
        ):
            continue
        target_weights[param_name] = target_weight
    if set(target_weights) != mapped_names:
        raise ValueError(
            f"Megatron state coverage differs: missing={sorted(mapped_names - set(target_weights))}, "
            f"unexpected={sorted(set(target_weights) - mapped_names)}"
        )
    hf_specs = hf_tensor_specs(spark_config)
    for mapping in mappings:
        target_weight = target_weights[mapping.megatron_param]
        expected_spec = _get_megatron_tensor_spec(mapping, hf_specs)
        if mapping.partition_size > 1:
            local_shape = list(expected_spec.shape)
            partition_dim = mapping.partition_dim
            if local_shape[partition_dim] % mapping.partition_size:
                raise ValueError(f"indivisible global tensor for {mapping.megatron_param}")
            local_shape[partition_dim] //= mapping.partition_size
            expected_spec = TensorSpec(tuple(local_shape), expected_spec.dtype)
        if (
            not isinstance(target_weight, torch.Tensor)
            or TensorSpec(tuple(target_weight.shape), target_weight.dtype) != expected_spec
        ):
            raise ValueError(f"Megatron shape or dtype differs for {mapping.megatron_param}: expected {expected_spec}")
        if target_weight.device.type == "meta":
            raise ValueError("Spark checkpoint loading requires materialized target parameters")
    return model, mappings, target_weights


def _validate_hf_checkpoint(model, checkpoint):
    checkpoint_config = parse_muse_spark_1x_config(checkpoint.config)
    target_config = model.config.spark
    if checkpoint_config != target_config:
        raise ValueError("checkpoint text semantics differ from the target Spark model")
    expected_specs = hf_tensor_specs(target_config)
    checkpoint_text_names = {
        tensor_name for tensor_name in checkpoint.weight_map if not tensor_name.startswith(NON_TEXT_PREFIXES)
    }
    if checkpoint_text_names != set(expected_specs):
        raise ValueError(
            f"HF tensor coverage differs: missing={sorted(set(expected_specs) - checkpoint_text_names)}, "
            f"unexpected={sorted(checkpoint_text_names - set(expected_specs))}"
        )
    for tensor_name, expected_spec in expected_specs.items():
        if checkpoint.tensor_metadata[tensor_name] != expected_spec:
            raise ValueError(
                f"HF shape or dtype differs for {tensor_name}: expected {expected_spec}, got {checkpoint.tensor_metadata[tensor_name]}"
            )


def _read_mapped_weight(checkpoint, mapping):
    def read_hf_slice(hf_name):
        selection = mapping.hf_selection(hf_name, checkpoint.tensor_metadata[hf_name])
        return checkpoint.read_slice(hf_name, selection)

    if isinstance(mapping.hf_param, dict):
        hf_weights = {projection: read_hf_slice(hf_name) for projection, hf_name in mapping.hf_param.items()}
    else:
        hf_weights = read_hf_slice(mapping.hf_param)
    return mapping.hf_slices_to_megatron(hf_weights)


def _raise_if_any_rank_failed(local_error):
    if dist.is_initialized():
        rank_errors = [None] * dist.get_world_size()
        dist.all_gather_object(rank_errors, local_error)
    else:
        rank_errors = [local_error]
    rank_failures = [f"rank {rank}: {rank_error}" for rank, rank_error in enumerate(rank_errors) if rank_error]
    if rank_failures:
        raise ValueError("Spark checkpoint loading failed: " + "; ".join(rank_failures))


def _load_local_weights(mappings, target_weights, checkpoint):
    for mapping in mappings:
        hf_weight = _read_mapped_weight(checkpoint, mapping)
        target_weight = target_weights[mapping.megatron_param]
        if hf_weight.shape != target_weight.shape or hf_weight.dtype != target_weight.dtype:
            raise ValueError(f"selected HF payload differs from target {mapping.megatron_param}")
        target_weight.copy_(hf_weight)
        del hf_weight


@torch.no_grad()
def load_hf_checkpoint(model, path):
    """Load local HF slices without casting parameters or buffers."""
    local_error = None
    # Catch every preflight failure so no rank starts copying after a peer rejects the checkpoint.
    try:
        model, mappings, target_weights = _prepare_model_weights(model)
        checkpoint = path if isinstance(path, SafetensorsCheckpoint) else SafetensorsCheckpoint(path)
        _validate_hf_checkpoint(model, checkpoint)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_if_any_rank_failed(local_error)

    local_error = None
    # A local read or copy failure must reach peers before they enter native checkpoint saving.
    try:
        _load_local_weights(mappings, target_weights, checkpoint)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_if_any_rank_failed(local_error)
