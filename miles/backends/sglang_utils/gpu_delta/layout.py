"""Immutable canonical layout admission and live-consumer identity checks (CUDA receiver only)."""

from __future__ import annotations

from contextlib import contextmanager
from itertools import chain

import torch

from miles.backends.sglang_utils.gpu_delta.bindings import (
    ComputedImage,
    ConsumerSnapshot,
    DerivedImage,
    _dtype_name,
    _tensor_identity,
)
from miles.backends.sglang_utils.gpu_delta.byte_views import _DtypeMismatch, _storage_key
from miles.backends.sglang_utils.gpu_delta.derived import plan_updates
from miles.backends.sglang_utils.gpu_delta.kernel_backends import admits_moe_dispatcher
from miles.backends.sglang_utils.gpu_delta.models import model_mapping
from miles.utils.gpu_delta import profiling
from miles.utils.gpu_delta.publication import canonical_encoding, static_tensor_reason


class GpuDeltaLayout:
    """Frozen canonical bindings with generic backend and consumer admission."""

    def __init__(self, model, checkpoint, shadows):
        """Bind ``checkpoint`` (a ``CanonicalCheckpoint``) to the live ``model``.

        ``shadows`` is the session's ``CanonicalShadows`` for tensors the loader serves only through
        arithmetic.
        """
        from miles.utils.gpu_delta.host_worker import _natural_key

        if not checkpoint.inventory:
            raise ValueError("canonical checkpoint metadata is unavailable")
        policy = checkpoint.policy
        # Normalize canonical wire metadata without changing checkpoint inventory.
        self.inventory = {
            name: {"encoding": canonical_encoding(meta["shape"]), **meta}
            for name, meta in checkpoint.inventory.items()
        }
        mapping = model_mapping(model, shadows)
        self.bindings = []
        self.excluded = mapping.parameters.excluded
        errors = []
        for name, meta in self.inventory.items():
            # The trainer's plan applies the same policy, so no mapping may override it.
            reason = static_tensor_reason(name, policy)
            if reason is not None:
                self.excluded[name] = reason
                continue
            try:
                binding = mapping.bind(name, meta)
                if binding is not None and binding.format is None:
                    _check_direct_dtype(binding)
            except _DtypeMismatch as error:
                # Report every mismatch together, including tensors whose delta stays empty.
                errors.append(
                    f"{name}: checkpoint={meta.get('checkpoint_dtype', meta['dtype'])}, "
                    f"canonical={meta['dtype']}; {error}"
                )
                continue
            if binding is not None:
                # The plan publishes canonical_encoding's choice; any other would skip this tensor's mask at apply.
                if binding.encoding != meta["encoding"]:
                    raise ValueError(
                        f"binding encoding {binding.encoding} differs from the canonical plan's "
                        f"{meta['encoding']}: {name}"
                    )
                self.bindings.append(binding)
        if errors:
            raise ValueError(
                "GPU-delta XOR apply/cancel requires matching checkpoint and served dtypes "
                "or an explicit lossless representation (including BF16->FP32 scalar/vector widening). "
                f"Found {len(errors)} mismatched tensors while planning the full checkpoint "
                "(including unchanged tensors); no delta has been applied:\n" + "\n".join(errors)
            )
        self.bindings.sort(key=lambda b: _natural_key(b.name))
        self._parameter_roots = [
            (module, name, tensor, _tensor_identity(tensor))
            for module in mapping.parameters.modules.values()
            for name, tensor in module._parameters.items()
            if tensor is not None
        ]
        self._consumers = list(mapping.consumers)
        # Mappings pin their consumers while binding; finish() may not add any.
        self._mapping_consumers = tuple(mapping.consumers)
        for layer, layout in mapping.parameters.moe_layers.values():
            self._consumers.append(ConsumerSnapshot(lambda layer=layer, layout=layout: layout.consumer_objects(layer)))
        self._prepare_identity_checks()
        self._mapping = mapping
        # Only shadows read tensor bytes; keep the header locations until they are filled.
        self._checkpoint = checkpoint if mapping.parameters.shadowed else None

    def finish(self):
        """Finish rank-local consumers on the first prepare, not during describe."""
        mapping = self._mapping
        if mapping is None:
            return
        self.check_identity()
        parameters = mapping.parameters
        if parameters.moe_layers:
            from sglang.srt.runtime_context import get_exec

            _require_fixed_moe_topology(get_exec().moe)
        if parameters.shadowed:
            # A version-0 admission fills its shadows off the scheduler thread, before any update reads them.
            with _side_stream(parameters.shadowed.values()):
                parameters.shadows.fill(self._checkpoint, parameters.shadowed)
        updates = []
        for layer, layout in mapping.parameters.moe_layers.values():
            updates.extend(layout.expert_derived_updates(layer))
        # Raw segments write typed pieces, so derived sources patch those pieces, not their parameter.
        self.derived_updates = plan_updates(
            updates,
            [
                target
                for binding in self.bindings
                if binding.is_raw
                for view in binding.raw_bindings()
                for target in view.storage
            ],
        )
        self.derived_targets = frozenset(
            id(target) for _, sources in self.derived_updates for _, targets in sources for target in targets
        )
        mapping.finish()
        if tuple(mapping.consumers) != self._mapping_consumers:
            raise ValueError("model mappings pin consumers while binding at describe, never in finish()")
        self.derived, self.computed = _split_images(mapping.derived)
        self._admit_images(updates, parameters.shadowed)
        self._reject_overlaps(mapping.aliases)
        self._mapping = self._checkpoint = None

    def _prepare_identity_checks(self):
        # Deduplicate geometry once, not in the per-update validation loop.
        # Separate views sharing storage still require separate geometry checks.
        checks = chain(
            ((tensor, expected, "parameter identity") for _, _, tensor, expected in self._parameter_roots),
            (
                (consumer.objects[index], expected, "consumer storage")
                for consumer in self._consumers
                for index, expected in consumer.tensors
            ),
        )
        unique = {}
        for tensor, expected, kind in checks:
            unique.setdefault(id(tensor), (tensor, expected, kind))
        self._tensor_checks = tuple(unique.values())

    @profiling.annotate("gpu_delta.check_identity")
    def check_identity(self):
        for module, name, tensor, _ in self._parameter_roots:
            if module._parameters.get(name) is not tensor:
                raise RuntimeError("GPU delta parameter identity changed; readmission required")
        for consumer in self._consumers:
            consumer.check_objects()
        for tensor, expected, kind in self._tensor_checks:
            if _tensor_identity(tensor) != expected:
                raise RuntimeError(f"GPU delta {kind} changed; readmission required")

    def _admit_images(self, updates, shadowed):
        """Images read canonical storage only and write pinned consumers; computed images reproduce theirs.

        Every canonical shadow (``shadowed``: name -> tensor) must feed a computed image: an update
        XORs the shadow, and only a computed image serves its value, from CUDA-resident storage.
        """
        read = {pointer for image in self.computed for pointer in image.source_pointers}
        for name, tensor in shadowed.items():
            if _storage_key(tensor) not in read:
                raise ValueError(f"canonical shadow {name} feeds no computed image")
        bound = {pointer for binding in self.bindings for pointer in binding.storage_pointers}
        written = {_storage_key(image.destination) for image in self.derived}
        written.update(_storage_key(tensor) for image in self.computed for tensor in image.destinations)
        written.update(_storage_key(tensor) for update in updates for tensor in update.destinations)
        for image in (*self.derived, *self.computed):
            if any(pointer in written for pointer in image.source_pointers):
                raise ValueError(f"derived image {image.name} reads another derived destination")
        # The binding table refreshes an image only when a binding on one of its sources is active.
        for kind, images in (("derived image", self.derived), ("computed image", self.computed)):
            for image in images:
                if any(pointer not in bound for pointer in image.source_pointers):
                    raise ValueError(f"{kind} {image.name} reads storage no canonical binding owns")
        # A replaced, pinned consumer fails the identity check; an unpinned one would go stale silently.
        pinned = {_storage_key(tensor) for _, _, tensor, _ in self._parameter_roots}
        pinned.update(
            _storage_key(consumer.objects[index]) for consumer in self._consumers for index, _ in consumer.tensors
        )
        for image in self.derived:
            if _storage_key(image.destination) not in pinned:
                raise ValueError(f"derived image {image.name} writes an unpinned consumer")
        for image in self.computed:
            if any(_storage_key(tensor) in bound for tensor in image.destinations):
                raise ValueError(f"computed image {image.name} overlaps a canonical binding")
            if any(_storage_key(tensor) not in pinned for tensor in image.destinations):
                raise ValueError(f"computed image {image.name} writes an unpinned consumer")
            if any(tensor.device.type != "cuda" for tensor in (*image.sources, *image.destinations)):
                raise ValueError(f"computed images require CUDA-resident storage: {image.name}")
        tensors = [tensor for image in self.computed for tensor in (*image.sources, *image.destinations)]
        with _side_stream(tensors):
            for image in self.computed:
                image.verify()

    def _reject_overlaps(self, model_aliases):
        # Gate/up intentionally share one interleaved weight/scale allocation;
        # their maps are disjoint. Other shared storage needs an explicit alias
        # adapter so a tied parameter is never XORed twice.
        owners = {}
        for binding in self.bindings:
            for tensor in binding.storage:
                key = _storage_key(tensor)
                prior = owners.get(key)
                if prior and prior != binding.name:
                    pair = (prior, binding.name)
                    if not all(".experts." in name for name in pair) and not (
                        _gate_up_alias(*pair) or model_aliases(*pair)
                    ):
                        raise ValueError(f"unclassified live weight alias: {pair}")
                owners[key] = binding.name


def _split_images(images):
    copies, computed = [], []
    for image in images:
        if type(image) is DerivedImage:
            copies.append(image)
        elif type(image) is ComputedImage:
            computed.append(image)
        else:
            raise ValueError(f"unsupported derived image: {type(image).__name__}")
    return copies, computed


@contextmanager
def _side_stream(tensors):
    """Run admission-time copies and checks off the serving streams, then wait for them."""
    devices = {tensor.device for tensor in tensors if tensor.device.type == "cuda"}
    if not devices:
        yield
        return
    if len(devices) != 1:
        raise ValueError(f"GPU delta admission work spans several CUDA devices: {sorted(map(str, devices))}")
    device = devices.pop()
    stream = torch.cuda.Stream(device=device)
    # Admission may just have built consumer views or images on this thread's current stream.
    # Order their reads without blocking that producer or synchronizing the whole device.
    stream.wait_stream(torch.cuda.current_stream(device))
    try:
        with torch.cuda.stream(stream):
            yield
    finally:
        # A rejected admission still drains its copies before its tensors can be reused.
        stream.synchronize()


def _check_direct_dtype(binding):
    allowed = {binding.dtype}
    if binding.dtype == "BF16" and len(binding.shape) <= 1:
        # prepare_raw_xor widens BF16 scalar/vector masks into FP32 storage losslessly.
        allowed.add("F32")
    served = list(dict.fromkeys(_dtype_name(target.dtype) for target in binding.storage))
    if any(dtype not in allowed for dtype in served):
        raise _DtypeMismatch(f"served={served}, allowed_served={sorted(allowed)}, shape={list(binding.shape)}")


def _gate_up_alias(a, b):
    return a.replace("gate_proj", "up_proj") == b or b.replace("gate_proj", "up_proj") == a


def _require_fixed_moe_topology(moe):
    # The canonical expert map is valid only while physical expert ownership is
    # the trivial EP partition. EPLB/custom/elastic maps can change ownership
    # without changing tensor pointers, so pointer checks alone cannot admit it.
    if (
        moe.enable_eplb
        or moe.elastic_ep_backend is not None
        or moe.init_expert_location != "trivial"
        or moe.ep_num_redundant_experts != 0
        or not admits_moe_dispatcher(moe.moe_runner_backend, moe.moe_a2a_backend)
    ):
        raise ValueError(
            "direct GPU deltas require fixed trivial EP, no redundant experts or EPLB, "
            "and a registered MoE runner with its admitted dispatcher"
        )
