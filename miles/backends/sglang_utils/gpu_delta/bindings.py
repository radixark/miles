"""Shared canonical byte bindings (CUDA receiver only).

Dense TP/vocabulary slicing, routed-expert dispatch to kernel-backend layouts, consumer snapshots,
derived and computed consumer images, receiver-owned canonical shadows, and the binding helpers
model mappings share: SGLang fused-projection loader shards and verified alias groups.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cached_property

import torch

from miles.backends.sglang_utils.gpu_delta.byte_views import (
    DestinationSegment,
    _byte_view,
    _DtypeMismatch,
    _storage_key,
)
from miles.backends.sglang_utils.gpu_delta.kernel_backends import identify_expert_layout
from miles.utils.gpu_delta.publication import _is_raw_encoding

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

STATIC_CALIBRATION = "static activation calibration"
REMOTE_EXPERT = "expert projection owned by another EP/TP rank"


def _dtype_name(dtype):
    return _DTYPES.get(dtype, str(dtype))


def _full_slices(shape):
    return [[0, size] for size in shape]


def _by_device(tensors):
    devices = {}
    for tensor in tensors:
        devices.setdefault(tensor.device, []).append(tensor)
    return tuple(tuple(group) for group in devices.values())


def _require_canonical_dtype(meta):
    if meta["dtype"] not in _DTYPES.values():
        raise _DtypeMismatch(f"unsupported checkpoint dtype: {meta['dtype']}")


def _require_slices(name, meta, slices):
    shape = meta["shape"]
    if len(slices) != len(shape) or any(
        not 0 <= start < stop <= extent for (start, stop), extent in zip(slices, shape, strict=True)
    ):
        raise ValueError(f"canonical selection {slices} is outside {name}{list(shape)}")


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
    # The live parameter ParameterBindings.bind resolved, which the receiver batches a non-layer binding by;
    # None when it is the canonical name.
    live_name: str | None = None

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
        return tuple(_storage_key(tensor) for tensor in self.storage)

    def raw_bindings(self):
        """One raw-mask binding per raw segment (or the whole tensor) and destination device.

        prepare_raw_xor takes only these views, never a segmented binding itself.
        """
        return self._raw_views

    @cached_property
    def _raw_views(self):
        if self.segments:
            # Raw segments are typed live pieces; each view applies only its own canonical range.
            return tuple(
                replace(
                    self,
                    shape=self.shape if segment.source_shape is None else segment.source_shape,
                    slices=[list(axis) for axis in segment.slices],
                    storage=targets,
                    segments=(),
                )
                for segment in self.segments
                for targets in _by_device(segment.destinations)
            )
        devices = _by_device(self.storage)
        if len(devices) == 1:
            return (self,)
        return tuple(replace(self, storage=targets) for targets in devices)

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
        if binding.segments:
            raise ValueError(f"raw XOR applies the raw_bindings() views of a segmented binding: {binding.name}")
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


def _direct_binding(name, meta, target, slices=None, live_name=None):
    """Bind canonical ``slices`` to one live tensor in the encoding the canonical plan publishes.

    ``meta["encoding"]`` comes from ``canonical_encoding``: a raw mask XORs the typed target, a
    decoded matrix mask its byte view.
    """
    slices = _full_slices(meta["shape"]) if slices is None else slices
    view_shape = tuple(stop - start for start, stop in slices)
    if target.numel() != math.prod(view_shape):
        raise ValueError(
            f"canonical/live dtype or shape mismatch for {name}: "
            f"canonical={meta['dtype']}{view_shape}, "
            f"live={_dtype_name(target.dtype)}{tuple(target.shape)}"
        )
    raw = _is_raw_encoding(meta["encoding"])
    return TensorBinding(
        name,
        meta["dtype"],
        tuple(meta["shape"]),
        slices,
        (target,),
        encoding=meta["encoding"],
        destinations=() if raw else (_byte_view(target),),
        format=meta.get("format"),
        live_name=live_name,
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
    slices = layout.expert_slices(layer, projection, suffix, meta["shape"])
    local_shape = tuple(stop - start for start, stop in slices)
    if not all(local_shape):
        return None
    # Kernel-backend layouts rely on this one check of the local projection.
    if len(local_shape) != 2 or any(type(value) is not int for value in (local, *local_shape)):
        raise ValueError(f"expert byte layout requires an integer matrix and local expert: {name}")
    if layout.format == "fp8-block128-ue8m0" and suffix == "weight_scale_inv":
        target = layout.expert_scale_target(layer, local, projection, local_shape)
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
    segments = layout.expert_destinations(
        layer,
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
            storage.setdefault(_storage_key(target), target)
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
        # MoeRunnerBackend and extension-registered runners both carry their exact SGLang value.
        backend = runner.runner_backend.value
    return identify_expert_layout(layer, backend)


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
    def source_pointers(self):
        return (_storage_key(self.source),)

    def refresh(self, ok):
        # Apply stream, after the matrix batches; the sticky error keeps the old copy. The pause runs under no_grad;
        # detached views of the same storage keep the out= selection valid when a test or manual driver refreshes
        # with grad enabled from a source that requires grad, as SGLang's plain nn.Parameter weights do.
        destination = self.destination.detach()
        torch.where(ok, self.source.detach(), destination, out=destination)


def _bits(tensor):
    # Same-width integer view: selection and comparison stay bitwise for FP8, BF16, FP32 and strided views.
    size = tensor.element_size()
    if size == 1:
        dtype = torch.uint8
    elif size == 2:
        dtype = torch.int16
    elif size == 4:
        dtype = torch.int32
    elif size == 8:
        dtype = torch.int64
    else:
        raise ValueError(f"unsupported computed image element size: {size}")
    return tensor.detach().view(dtype)


@dataclass
class ComputedImage:
    """Served values the loader computes from canonical bytes, recomputed in place during the pause.

    ``compute(*sources)`` mirrors the loader's own function: it takes the live sources as separate
    arguments (``DerivedUpdate.compute`` takes one tuple of values) and returns one tensor per
    destination. Sources are canonical-bound storage (live parameters or canonical shadows), never
    a derived destination. Each destination is a module parameter or a tensor a consumer snapshot
    pins. Admission requires CUDA residency and bytewise reproduction of every destination.
    """

    name: str
    sources: tuple[torch.Tensor, ...]
    destinations: tuple[torch.Tensor, ...]
    compute: Callable[..., tuple[torch.Tensor, ...]]

    def values(self):
        with torch.no_grad():
            outputs = tuple(self.compute(*self.sources))
        if len(outputs) != len(self.destinations) or any(
            (value.shape, value.dtype, value.device) != (destination.shape, destination.dtype, destination.device)
            for value, destination in zip(outputs, self.destinations, strict=True)
        ):
            raise ValueError(f"computed image geometry changed: {self.name}")
        return outputs

    @cached_property
    def source_pointers(self):
        return tuple(_storage_key(source) for source in self.sources)

    def refresh(self, ok):
        # Apply stream, after the matrix batches; the sticky error keeps every old value.
        for destination, value in zip(self.destinations, self.values(), strict=True):
            bits = _bits(destination)
            torch.where(ok, _bits(value), bits, out=bits)

    def verify(self):
        for destination, value in zip(self.destinations, self.values(), strict=True):
            if not torch.equal(_bits(value), _bits(destination)):
                raise ValueError(
                    f"computed image {self.name} does not reproduce its live consumer; "
                    "the engine must serve the immutable checkpoint"
                )


def _tensor_identity(tensor):
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _require_unquantized(name, module):
    quant_name = type(
        getattr(module, "quant_method", None)
    ).__name__  # config-access-exempt: Plain norms and SGLang linear modules expose different quantization metadata.
    if quant_name not in {
        "NoneType",
        "UnquantizedLinearMethod",
        "UnquantizedEmbeddingMethod",
    }:
        raise ValueError(f"unsupported dense delta quantization {quant_name}: {name}")


def dense_target(name, meta, module, target, slices):
    _require_unquantized(name, module)
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

            # Exact classes only: fused subclasses (merged column, QKV) shard each part separately,
            # so one contiguous rank slice would address the wrong bytes; their mappings bind shards.
            if type(module) is RowParallelLinear:
                allowed_dim = getattr(
                    target, "input_dim", None
                )  # config-access-exempt: SGLang parameter loaders attach the row-parallel sharding axis.
            elif type(module) is ColumnParallelLinear:
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


def _output_dim(param):
    dim = getattr(
        param, "output_dim", None
    )  # config-access-exempt: SGLang parameter loaders attach the column-parallel sharding axis.
    if dim is None:
        raise ValueError("column-parallel projection parameter has no output_dim")
    return dim


def _loader_view(param, dim, start, rows, slices):
    # MergedColumnParallelLinear.weight_loader asserts the live rows and the narrowed checkpoint rows have equal shapes.
    view = param.narrow(dim, start, rows)
    if tuple(view.shape) != tuple(stop - begin for begin, stop in slices):
        raise ValueError("projection geometry differs from its loader")
    return view


def _merged_partition(module):
    from sglang.srt.distributed.utils import get_group_rank_size
    from sglang.srt.layers.linear import MergedColumnParallelLinear

    # Subclasses and MergedColumnParallelRepeatedLinear own different loaders.
    if type(module) is not MergedColumnParallelLinear or module.use_presharded_weights:
        raise ValueError(f"unclassified merged column projection: {type(module).__name__}")
    rank, size = get_group_rank_size(module.tp_group)
    sizes = list(module.output_sizes)
    if any(extent <= 0 or extent % size for extent in sizes):
        raise ValueError("merged column shards must be nonempty and divide their TP group")
    return rank, size, sizes


def merged_column_target(param, module, shard, canonical_shape):
    """Mirror ``MergedColumnParallelLinear.weight_loader`` for one int shard; returns ``(live view, slices)``.

    Shard ``s`` fills local rows from ``sum(output_sizes[:s]) // tp`` with canonical rows
    ``[rank * width, (rank + 1) * width)`` of its own checkpoint tensor, ``width = output_sizes[s] // tp``.
    """
    rank, size, sizes = _merged_partition(module)
    if type(shard) is not int or not 0 <= shard < len(sizes):
        raise ValueError(f"unsupported merged column shard: {shard!r}")
    dim = _output_dim(param)
    if (
        len(canonical_shape) != param.ndim
        or param.shape[dim] != sum(sizes) // size
        or canonical_shape[dim] != sizes[shard]
    ):
        raise ValueError("merged column projection geometry differs from its loader")
    width = sizes[shard] // size
    slices = _full_slices(canonical_shape)
    slices[dim] = [rank * width, (rank + 1) * width]
    return _loader_view(param, dim, sum(sizes[:shard]) // size, width, slices), slices


def _byte_extent(tensor):
    # Half-open byte range of its storage that a strided view can address.
    start = tensor.storage_offset() * tensor.element_size()
    if tensor.numel() == 0:
        return start, start
    span = sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True))
    return start, start + (span + 1) * tensor.element_size()


class AliasGroups:
    """Canonical tensors a loader concatenates into one live allocation; a mapping's ``aliases``.

    ``declare`` verifies that every live view of a binding lies inside its group's fused buffer
    and is disjoint from the other members, so any two members of one group may share storage.
    """

    def __init__(self):
        self._groups = {}  # canonical name -> group
        self._extents = {}  # group -> [(canonical name, byte start, byte stop)]

    def declare(self, name, group, binding, buffer):
        if name in self._groups:
            raise ValueError(f"fused view declared twice: {name}")
        low, high = _byte_extent(buffer)
        members = self._extents.setdefault(group, [])
        views = [view for segment in binding.segments for view in segment.destinations] or binding.storage
        for view in views:
            start, stop = _byte_extent(view)
            if _storage_key(view) != _storage_key(buffer) or start < low or stop > high:
                raise ValueError(f"{name} is not a view of its fused buffer {group}")
            for other, other_start, other_stop in members:
                if other != name and start < other_stop and other_start < stop:
                    raise ValueError(f"fused views overlap in {group}: {other} and {name}")
        members.extend((name, *_byte_extent(view)) for view in views)
        self._groups[name] = group

    def __call__(self, a, b):
        group = self._groups.get(a)
        return group is not None and group == self._groups.get(b)


def router_bias_consumer(name, topk, bias):
    """Pin the correction bias that TopK and the fused routers read on every call.

    They read the Parameter itself, so its XOR reaches routing; a copy would never see it.
    """
    if topk.topk_config.correction_bias is not bias:
        raise ValueError(f"routing does not read the bound correction bias: {name}")
    return ConsumerSnapshot(lambda: (topk.topk_config, topk.topk_config.correction_bias))


def _shadow_unavailable(name):
    return f"canonical shadow {name} is unavailable after version 0; restart the engine from the HF base"


@dataclass
class CanonicalShadow:
    """Receiver-owned exact rows of one canonical tensor."""

    dtype: str
    shape: tuple[int, ...]  # the canonical tensor's shape
    rows: tuple[int, int]
    tensor: torch.Tensor
    filled: bool = False
    allocation_ready: torch.cuda.Event | None = None


class CanonicalShadows:
    """Receiver-owned exact canonical rows of tensors the loader serves only through arithmetic.

    The DeltaSession owns ``store``, which survives clear_gpu_delta_state, so a filled shadow always
    holds the committed version: every applied update XORs it like any other binding target. Only a
    version-0 admission (``create``) adds or fills a shadow, from the immutable checkpoint.
    """

    def __init__(self, store, create):
        self.store = store
        self.create = create

    def get(self, name, meta, rows, device):
        """Rows ``[start, stop)`` of ``name`` on ``device``, the destination tensor's own (indexed) device.

        Allocated once per session; every later admission must name the same device.
        """
        _require_canonical_dtype(meta)
        shape, rows, device = tuple(meta["shape"]), tuple(rows), torch.device(device)
        if not shape or not 0 <= rows[0] < rows[1] <= shape[0]:
            raise ValueError(f"canonical shadow rows {list(rows)} are outside {name}{list(shape)}")
        if device.type == "cuda" and device.index is None:
            # An allocation resolves "cuda" to the current device; re-admission must compare that exact device.
            raise ValueError(f"canonical shadow {name} takes its destination's indexed device, not {device}")
        shadow = self.store.get(name)
        if shadow is None:
            if not self.create:
                raise ValueError(_shadow_unavailable(name))
            tensor = torch.empty((rows[1] - rows[0], *shape[1:]), dtype=_TORCH_DTYPES[meta["dtype"]], device=device)
            shadow = self.store[name] = CanonicalShadow(meta["dtype"], shape, rows, tensor)
            if device.type == "cuda":
                # The allocator may reuse storage with pending writes on this stream. Preparation runs
                # on another thread/stream, so retain this allocation's readiness until its first fill.
                shadow.allocation_ready = torch.cuda.Event()
                shadow.allocation_ready.record(torch.cuda.current_stream(device))
        elif (shadow.dtype, shadow.shape, shadow.rows, shadow.tensor.device) != (meta["dtype"], shape, rows, device):
            raise ValueError(f"canonical shadow geometry changed: {name}")
        return shadow.tensor

    def fill(self, checkpoint, names):
        """Copy the immutable checkpoint's rows into every unfilled shadow, at the first prepare."""
        for name in names:
            shadow = self.store[name]
            if shadow.filled:
                continue
            if not self.create:
                raise ValueError(_shadow_unavailable(name))
            meta = checkpoint.inventory[name]
            if (meta["dtype"], tuple(meta["shape"])) != (shadow.dtype, shadow.shape):
                raise ValueError(f"canonical shadow geometry changed: {name}")
            data = torch.from_numpy(checkpoint.read_rows(name, *shadow.rows).copy())
            target = shadow.tensor.view(torch.uint8).reshape(-1)
            if data.numel() != target.numel():
                raise ValueError(f"canonical shadow geometry changed: {name}")
            if shadow.allocation_ready is not None:
                torch.cuda.current_stream(shadow.tensor.device).wait_event(shadow.allocation_ready)
            target.copy_(data)
            shadow.filled = True
            shadow.allocation_ready = None


class ParameterBindings:
    """Shared routed-expert and ordinary dense binding context at admission."""

    def __init__(self, model, stacked_params=(), shadows=None):
        self.modules = dict(model.named_modules())
        self.params = dict(model.named_parameters(remove_duplicate=False))
        self.gate_up_params = [(source, shard) for target, source, shard in stacked_params if target == "gate_up_proj"]
        self.excluded = {}
        self.moe_layers = {}  # prefix -> (layer, its identified ExpertLayout)
        self.shadows = shadows
        self.shadowed = {}  # canonical name -> its shadow; the layout fills them at the first prepare

    def module(self, name):
        """The served model's module ``name``; a canonical tensor that names no module is unclassified."""
        module = self.modules.get(name)
        if module is None:
            raise ValueError(f"canonical tensor has no runtime module: {name}")
        return module

    def shadow(self, name, meta, rows, device):
        """Canonical rows a lossy loader cast is computed from, on ``destination.device``.

        Bind them with ``bind_view(..., None, ...)``; a computed image must read them.
        """
        if self.shadows is None:
            raise _DtypeMismatch("lossy loader cast needs a canonical shadow")
        tensor = self.shadowed[name] = self.shadows.get(name, meta, rows, device)
        return tensor

    def bind_view(self, name, meta, module, view, slices):
        """Bind canonical ``slices`` to one live view the mapping resolved from its loader.

        Views are loader shards (``merged_column_target``), head-group slices or fused rows of a
        parameter that ``module`` owns.
        """
        _require_canonical_dtype(meta)
        _require_unquantized(name, module)
        _require_slices(name, meta, slices)
        return _direct_binding(name, meta, view, [list(axis) for axis in slices])

    def bind(self, name, meta, target_name, target=None):
        _require_canonical_dtype(meta)
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
                self.excluded[name] = STATIC_CALIBRATION if suffix == "input_scale" else REMOTE_EXPERT
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
        return _direct_binding(name, meta, target, slices, live_name=None if target_name == name else target_name)
