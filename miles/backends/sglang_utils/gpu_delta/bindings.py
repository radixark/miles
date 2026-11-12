"""Shared canonical byte bindings and admitted FlashInfer storage layouts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from functools import cached_property

import torch

from miles.backends.sglang_utils.gpu_delta.flashinfer import (
    DestinationSegment,
    _byte_view,
    _DtypeMismatch,
    expert_destinations,
    expert_scale_target,
    expert_slices,
    identify_layout,
)

_DTYPES = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int32: "I32",
    torch.int64: "I64",
}

_TORCH_DTYPES = {name: dtype for dtype, name in _DTYPES.items()}


def _dtype_name(dtype):
    return _DTYPES.get(dtype, str(dtype))


def _full_slices(shape):
    return [[0, size] for size in shape]


@dataclass
class TensorBinding:
    name: str
    dtype: str
    shape: tuple[int, ...]
    slices: list[list[int]]
    storage: tuple[torch.Tensor, ...]
    encoding: str = "xor_bytes"
    destinations: tuple[torch.Tensor, ...] = ()
    segments: tuple[DestinationSegment, ...] = ()
    format: str | None = None

    def __post_init__(self):
        # The admitted binding is immutable for the session. Keep dtype and
        # slicing metadata off the paused per-tensor application path.
        self.torch_dtype = _TORCH_DTYPES[self.dtype]
        if self.encoding == "xor_raw_bytes":
            self.is_raw = True
        elif self.encoding == "xor_bytes":
            self.is_raw = False
        else:
            raise ValueError(f"unsupported GPU delta tensor encoding: {self.encoding}")
        self.raw_ue8m0 = self.format == "fp8-block128-ue8m0" and self.dtype == "U8"
        self._selection = tuple(slice(start, stop) for start, stop in self.slices)
        layer = re.search(r"(?:^|\.)layers\.(\d+)\.", self.name)
        self.layer = int(layer[1]) if layer else None

    @cached_property
    def storage_pointers(self):
        return tuple((tensor.device, tensor.untyped_storage().data_ptr()) for tensor in self.storage)

    def raw_bindings(self):
        """Reuse one small-mask binding per destination device."""
        return self._raw_views

    @cached_property
    def _raw_views(self):
        devices = {}
        for target in self.storage:
            devices.setdefault(target.device, []).append(target)
        if len(devices) == 1:
            return (self,)
        return tuple(replace(self, storage=tuple(targets)) for targets in devices.values())

    def apply_bindings(self):
        """Return receiver-local byte-layout pieces of one canonical tensor."""
        return self._apply_views

    @cached_property
    def _apply_views(self):
        if not self.segments:
            return (self,)
        return tuple(
            replace(
                self,
                shape=self.shape if segment.source_shape is None else segment.source_shape,
                slices=[list(axis) for axis in segment.slices],
                destinations=segment.destinations,
                segments=(),
            )
            for segment in self.segments
        )

    def selected_bytes(self, canonical_bytes):
        value = canonical_bytes.view(self.torch_dtype).reshape(self.shape)
        # Keep column-sharded selections as views of reusable decoded scratch.
        # The decoder fills this storage after preparation; never cache a copy.
        return _byte_view(value[self._selection])


def prepare_raw_xor(items, derived_targets=()):
    """Materialize small physical XOR masks before pausing, without live writes.

    Served values are lossless checkpoint images. Dtype admission permits BF16
    scalar/vector widening and the explicit UE8M0-to-FP32 scale representation.
    Ordinary byte-identical targets reuse their mask without reading live values
    unless a derived update needs the predicted value.
    """
    groups, prepared = {}, []
    for binding, payload in items:
        selected = binding.selected_bytes(payload)
        for target in binding.storage:
            mask = selected.reshape(-1)
            if not binding.raw_ue8m0 and binding.torch_dtype == target.dtype and id(target) not in derived_targets:
                prepared.append((target, mask, None))
            else:
                key = binding.torch_dtype, target.dtype, binding.raw_ue8m0, target.device
                groups.setdefault(key, []).append((target, mask))
    for (canonical_dtype, target_dtype, ue8m0, _), members in groups.items():
        targets, masks = zip(*members, strict=True)
        mask = torch.cat(masks)
        if canonical_dtype == torch.bfloat16 and target_dtype == torch.float32:
            # Lossless BF16 widening embeds its bits in the upper FP32 half.
            # Widen the XOR mask directly: no live read or numerical round trip.
            physical_mask = (mask.view(torch.int16).to(torch.int32) << 16).view(torch.uint8)
            new = None
            if any(id(target) in derived_targets for target in targets):
                old = torch.cat([target.detach().reshape(-1) for target in targets])
                new = old.view(torch.uint8).bitwise_xor(physical_mask).view(target_dtype)
        elif (ue8m0 and canonical_dtype == torch.uint8 and target_dtype == torch.float32) or (
            not ue8m0 and canonical_dtype == target_dtype
        ):
            old = torch.cat([target.detach().reshape(-1) for target in targets])
            canonical = old.to(torch.float8_e8m0fnu if ue8m0 else canonical_dtype)
            updated = canonical.view(torch.uint8).bitwise_xor(mask)
            new = updated.view(canonical.dtype).to(target_dtype)
            physical_mask = old.view(torch.uint8).bitwise_xor(new.view(torch.uint8)) if ue8m0 else mask
        else:
            raise ValueError(f"unsupported raw XOR representation: {canonical_dtype} -> {target_dtype}")
        offset = 0
        for target in targets:
            count = target.numel()
            byte_start, byte_stop = offset * target.element_size(), (offset + count) * target.element_size()
            prepared.append(
                (
                    target,
                    physical_mask[byte_start:byte_stop],
                    None if new is None else new[offset : offset + count].view(target.shape),
                )
            )
            offset += count
    return prepared


def _direct_binding(name, meta, target, slices=None):
    slices = _full_slices(meta["shape"]) if slices is None else slices
    view_shape = tuple(stop - start for start, stop in slices)
    if target.numel() != math.prod(view_shape):
        raise ValueError(
            f"canonical/live dtype or shape mismatch for {name}: "
            f"canonical={meta['dtype']}{view_shape}, "
            f"live={_dtype_name(target.dtype)}{tuple(target.shape)}"
        )
    direct = len(meta["shape"]) <= 1
    return TensorBinding(
        name,
        meta["dtype"],
        tuple(meta["shape"]),
        slices,
        (target,),
        encoding="xor_raw_bytes" if direct else "xor_bytes",
        destinations=() if direct else (_byte_view(target),),
        format=meta.get("format"),
    )


def _normalize_expert_meta(meta, layout, suffix):
    if layout.format != "fp8-block128-ue8m0":
        return
    if suffix == "weight":
        if meta["dtype"] != "F8_E4M3":
            raise _DtypeMismatch(f"checkpoint={meta['dtype']}; block128 FP8 requires canonical E4M3 weights")
    elif suffix == "weight_scale_inv":
        if meta["dtype"] == "F32":
            meta["checkpoint_dtype"] = "F32"
            meta["dtype"] = "U8"
        elif meta["dtype"] != "U8":
            raise _DtypeMismatch(f"checkpoint={meta['dtype']}; block128 FP8 requires exact UE8M0 scale values")
        meta["encoding"] = "xor_raw_bytes"
    else:
        raise ValueError(f"unclassified block128 FP8 expert tensor: {suffix}")
    meta["format"] = layout.format


def _moe_binding(name, meta, layer, expert, projection, suffix, layout):
    _normalize_expert_meta(meta, layout, suffix)
    if not layout.gated and projection == "gate":
        raise ValueError("non-gated experts have no gate projection")
    # Scalar metadata stays gate-first.
    if projection == "gate":
        stem, half = "w13", 0
    elif projection == "up":
        stem, half = "w13", 1
    elif projection == "down":
        stem, half = "w2", None
    else:
        raise ValueError(f"unsupported expert projection: {projection}")
    if suffix == "input_scale" and layout.format == "nvfp4":
        # Activation calibration belongs to the immutable checkpoint prior.
        # Only weight-derived scales change during training.
        return None
    local = layer._map_global_expert_id_to_local_expert_id(expert)
    if local < 0:
        return None
    if suffix == "weight_scale_2" and layout.format == "nvfp4":
        target = getattr(
            layer, f"{stem}_weight_scale_2"
        )  # config-access-exempt: Select the admitted gate/up or down expert scale image.
        target = target[local, half] if projection != "down" and layout.gated else target[local]
        return _direct_binding(name, meta, target)
    slices = expert_slices(layer, layout, projection, suffix, meta["shape"])
    local_shape = tuple(stop - start for start, stop in slices)
    if not all(local_shape):
        return None
    # The FlashInfer layout helpers rely on this one check of the local projection.
    if len(local_shape) != 2 or any(type(value) is not int for value in (local, *local_shape)):
        raise ValueError(f"expert byte layout requires an integer matrix and local expert: {name}")
    if layout.format == "fp8-block128-ue8m0" and suffix == "weight_scale_inv":
        target = expert_scale_target(layer, layout, local, projection, local_shape)
        return TensorBinding(
            name,
            meta["dtype"],
            tuple(meta["shape"]),
            slices,
            (target,),
            encoding="xor_raw_bytes",
            format=meta["format"],
        )
    byte_suffix = suffix
    if layout.format == "mxfp8" and suffix == "weight_scale_inv":
        byte_suffix = "weight_scale"
    elif layout.format == "mxfp4" and suffix == "weight_packed":
        byte_suffix = "weight"
    segments = expert_destinations(
        layer,
        layout,
        local,
        projection,
        byte_suffix,
        local_shape,
        meta["dtype"],
    )
    segments = tuple(
        DestinationSegment(
            tuple(
                (start + origin, stop + origin)
                for (start, stop), (origin, _) in zip(segment.slices, slices, strict=False)
            ),
            segment.destinations,
        )
        for segment in segments
    )
    storage = {}
    for segment in segments:
        for target in segment.destinations:
            storage.setdefault(target.untyped_storage().data_ptr(), target)
    return TensorBinding(
        name,
        meta["dtype"],
        tuple(meta["shape"]),
        slices,
        tuple(storage.values()),
        segments=segments,
        format=layout.format,
    )


def _identify_moe_layout(layer):
    # All ordinary expert methods initialize runner. The direct TRTLLM wrapper
    # sets it to None and dispatches using the explicit runtime selection.
    runner = layer.quant_method.runner
    if runner is None:
        from sglang.srt.runtime_context import get_exec

        backend = get_exec().moe.moe_runner_backend
    else:
        backend = runner.runner_backend
    return identify_layout(layer, backend)


@dataclass
class DerivedImage:
    """An independent consumer refreshed from a live canonical storage view."""

    name: str
    destination: torch.Tensor
    source: torch.Tensor

    def __post_init__(self):
        if self.source.shape != self.destination.shape or self.source.dtype != self.destination.dtype:
            raise ValueError(f"unsupported derived delta buffer geometry: {self.name}")

    @cached_property
    def source_pointer(self):
        return self.source.device, self.source.untyped_storage().data_ptr()


def _tensor_identity(tensor):
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def dense_target(name, meta, module, target, slices):
    quant_name = type(
        getattr(module, "quant_method", None)
    ).__name__  # config-access-exempt: Plain norms and SGLang linear modules expose different quantization metadata.
    if quant_name not in {
        "NoneType",
        "UnquantizedLinearMethod",
        "UnquantizedEmbeddingMethod",
    }:
        raise ValueError(f"unsupported dense delta quantization {quant_name}: {name}")
    if tuple(target.shape) != tuple(stop - start for start, stop in slices):
        if hasattr(
            module, "shard_indices"
        ):  # config-access-exempt: Only vocabulary-parallel modules carry vocabulary shard metadata.
            indices = module.shard_indices
            if indices.num_added_elements:
                raise ValueError("added-vocabulary delta layout is not supported")
            start, stop = indices.org_vocab_start_index, indices.org_vocab_end_index
            slices[0] = [start, stop]
            target = target[: stop - start]
        else:
            mismatches = [
                i for i, (local, full) in enumerate(zip(target.shape, meta["shape"], strict=False)) if local != full
            ]
            if len(mismatches) != 1 or len(target.shape) != len(meta["shape"]):
                raise ValueError(f"unclassified dense shape mapping: {name}")
            dim = mismatches[0]
            from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear

            if isinstance(module, RowParallelLinear):
                allowed_dim = getattr(
                    target, "input_dim", None
                )  # config-access-exempt: SGLang parameter loaders attach the row-parallel sharding axis.
            elif isinstance(module, ColumnParallelLinear):
                allowed_dim = getattr(
                    target, "output_dim", None
                )  # config-access-exempt: SGLang parameter loaders attach the column-parallel sharding axis.
            else:
                raise ValueError(f"unclassified tensor-parallel mapping: {name}")
            if dim != allowed_dim:
                raise ValueError(f"unclassified tensor-parallel mapping: {name}")
            from sglang.srt.distributed.utils import get_group_rank_size

            rank, _ = get_group_rank_size(module.tp_group)
            size = target.shape[dim]
            slices[dim] = [rank * size, (rank + 1) * size]
    return target, slices


class ConsumerSnapshot:
    """Read live consumer fields without retaining a stale replacement tensor."""

    def __init__(self, read):
        self.read = read
        self.objects = read()
        self.tensors = [
            (index, _tensor_identity(value))
            for index, value in enumerate(self.objects)
            if isinstance(value, torch.Tensor)
        ]

    def check_objects(self):
        current = self.read()
        if len(current) != len(self.objects) or any(a is not b for a, b in zip(current, self.objects, strict=False)):
            raise RuntimeError("GPU delta consumer storage changed; readmission required")


def gate_up_target(target, module, shard, slices):
    from sglang.srt.distributed.utils import get_group_rank_size

    rank, _ = get_group_rank_size(module.tp_group)
    dim = target.output_dim
    half = target.shape[dim] // 2
    target = target.narrow(dim, shard * half, half)
    slices[dim] = [rank * half, (rank + 1) * half]
    return target


class ParameterBindings:
    """Shared routed-expert and ordinary dense binding context at admission."""

    def __init__(self, model, stacked_params=()):
        self.modules = dict(model.named_modules())
        self.params = dict(model.named_parameters(remove_duplicate=False))
        self.gate_up_params = [(source, shard) for target, source, shard in stacked_params if target == "gate_up_proj"]
        self.excluded = {}
        self.moe_layers = {}  # prefix -> (layer, identified FlashInferLayout)

    def bind(self, name, meta, target_name, target=None):
        if meta["dtype"] not in _DTYPES.values():
            raise _DtypeMismatch(f"unsupported checkpoint dtype: {meta['dtype']}")
        match = re.fullmatch(r"(.+\.experts)\.(\d+)\.(gate|up|down)_proj\.(.+)", target_name)
        if match:
            prefix, expert, projection, suffix = match.groups()
            layer = self.modules.get(prefix)
            if layer is None:
                raise ValueError(f"canonical expert has no runtime layer: {name}")
            if prefix not in self.moe_layers:
                if layer.use_presharded_weights:
                    raise ValueError("GPU delta requires an unsharded canonical checkpoint")
                self.moe_layers[prefix] = layer, _identify_moe_layout(layer)
            binding = _moe_binding(
                name,
                meta,
                layer,
                int(expert),
                projection,
                suffix,
                self.moe_layers[prefix][1],
            )
            if binding is None:
                self.excluded[name] = (
                    "static activation calibration"
                    if suffix == "input_scale"
                    else "expert projection owned by another EP/TP rank"
                )
            return binding
        if target is None:
            target = self.params.get(target_name)
        slices = _full_slices(meta["shape"])
        if target is None:
            for source_stem, shard in self.gate_up_params:
                if source_stem not in target_name or ".experts." in target_name:
                    continue
                candidate = target_name.replace(source_stem, "gate_up_proj")
                target = self.params.get(candidate)
                if target is None:
                    continue
                module = self.modules[candidate.rsplit(".", 1)[0]]
                target = gate_up_target(target, module, shard, slices)
                target_name = candidate
                break
        if target is None:
            raise ValueError(f"unclassified canonical tensor: {name}")
        module = self.modules[target_name.rsplit(".", 1)[0]]
        target, slices = dense_target(name, meta, module, target, slices)
        return _direct_binding(name, meta, target, slices)
