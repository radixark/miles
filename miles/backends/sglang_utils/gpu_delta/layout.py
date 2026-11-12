"""Immutable canonical layout admission and live-consumer identity checks."""

from __future__ import annotations

from itertools import chain

from miles.backends.sglang_utils.gpu_delta.bindings import ConsumerSnapshot, _dtype_name, _tensor_identity
from miles.backends.sglang_utils.gpu_delta.derived import nvfp4_derived_update, plan_updates
from miles.backends.sglang_utils.gpu_delta.flashinfer import _DtypeMismatch, consumer_objects
from miles.backends.sglang_utils.gpu_delta.models import model_mapping
from miles.utils.gpu_delta import profiling


class GpuDeltaLayout:
    """Frozen canonical bindings with generic backend and consumer admission."""

    def __init__(self, model, inventory):
        from miles.utils.gpu_delta.host_worker import _natural_key

        if not inventory:
            raise ValueError("canonical checkpoint metadata is unavailable")
        # Normalize canonical wire metadata without changing checkpoint inventory.
        self.inventory = {
            name: {"encoding": "xor_raw_bytes" if len(meta["shape"]) <= 1 else "xor_bytes", **meta}
            for name, meta in inventory.items()
        }
        mapping = model_mapping(model)
        self.bindings = []
        self.excluded = mapping.parameters.excluded
        errors = []
        for name, meta in self.inventory.items():
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
        for layer, layout in mapping.parameters.moe_layers.values():
            self._consumers.append(
                ConsumerSnapshot(lambda layer=layer, layout=layout: consumer_objects(layer, layout))
            )
        self._prepare_identity_checks()
        self._mapping = mapping

    def finish(self):
        """Finish rank-local consumers on the first prepare, not during describe."""
        mapping = self._mapping
        if mapping is None:
            return
        self.check_identity()
        if mapping.parameters.moe_layers:
            from sglang.srt.runtime_context import get_exec

            _require_fixed_moe_topology(get_exec().moe)
        self.derived = []
        updates = []
        for layer, layout in mapping.parameters.moe_layers.values():
            if layout.format == "nvfp4":
                updates.append(nvfp4_derived_update(layer, layout))
            elif layout.format in {"bf16", "mxfp4", "mxfp8", "fp8-block128-ue8m0"}:
                # These formats carry no per-tensor scale, so no consumer value derives from one.
                continue
            else:
                raise ValueError(f"unsupported GPU delta expert format: {layout.format}")
        self.derived_updates = plan_updates(
            updates,
            [target for binding in self.bindings if binding.is_raw for target in binding.storage],
        )
        self.derived_targets = frozenset(
            id(target) for _, sources in self.derived_updates for _, targets in sources for target in targets
        )
        mapping.finish()
        self.derived.extend(mapping.derived)
        self._reject_overlaps(mapping.aliases)
        self._mapping = None

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

    def _reject_overlaps(self, model_aliases):
        # Gate/up intentionally share one interleaved weight/scale allocation;
        # their maps are disjoint. Other shared storage needs an explicit alias
        # adapter so a tied parameter is never XORed twice.
        owners = {}
        for binding in self.bindings:
            for tensor in binding.storage:
                key = tensor.device, tensor.untyped_storage().data_ptr()
                prior = owners.get(key)
                if prior and prior != binding.name:
                    pair = (prior, binding.name)
                    if not all(".experts." in name for name in pair) and not (
                        _gate_up_alias(*pair) or model_aliases(*pair)
                    ):
                        raise ValueError(f"unclassified live weight alias: {pair}")
                owners[key] = binding.name


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
        or (moe.moe_runner_backend, moe.moe_a2a_backend)
        not in {
            ("flashinfer_trtllm", "none"),
            ("flashinfer_trtllm_routed", "none"),
            ("flashinfer_mxfp4", "none"),
            ("flashinfer_cutedsl", "none"),
            ("flashinfer_megamoe", "flashinfer_megamoe"),
        }
    ):
        raise ValueError(
            "direct GPU deltas require fixed trivial EP, no redundant experts or EPLB, "
            "and a native FlashInfer runner with its matching dispatcher"
        )
