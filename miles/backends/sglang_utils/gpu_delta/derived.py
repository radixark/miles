"""Prepare small derived consumer values without modifying live model state.

Weight and block-scale permutations belong to byte bindings. These updates only
handle NVFP4's complete scalar calibration values, never decoded XOR masks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from types import SimpleNamespace

import torch


@dataclass(frozen=True)
class PreparedCopy:
    destination: torch.Tensor
    source: torch.Tensor


@dataclass(frozen=True)
class DerivedUpdate:
    """An update whose inputs can include several independently changed scales.

    ``overrides`` contains complete tensors keyed by ``id(original_source)``.
    The caller assembles any partial canonical updates before preparation and
    applies the returned copies only after successful DE/apply completion.
    """

    sources: tuple[torch.Tensor, ...]
    destinations: tuple[torch.Tensor, ...]
    compute: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]
    checks: Callable | None = None

    @cached_property
    def compute_device(self):
        return next((source.device for source in self.sources if source.is_cuda), self.sources[0].device)

    @cached_property
    def mixed_sources(self):
        return any(source.device != self.compute_device for source in self.sources)

    def _prepare(self, overrides):
        # Device predicates are returned unchecked: prepare_updates validates
        # every update's predicates once, before returning any prepared copies.
        values = tuple(overrides.get(id(source), source) for source in self.sources)
        with torch.no_grad():
            if self.mixed_sources:
                values = tuple(value.to(self.compute_device) for value in values)
            outputs = tuple(self.compute(values))
            if len(outputs) != len(self.destinations):
                raise ValueError("GPU delta derived output count changed")
            result, snapshots = [], {}
            for destination, value in zip(self.destinations, outputs, strict=False):
                _same_geometry(destination, value)
                # Retain complete target values independently of live buffers;
                # later source copies must not alter a prepared update.
                if id(value) not in snapshots:
                    snapshots[id(value)] = value.detach().clone()
                result.append(PreparedCopy(destination, snapshots[id(value)]))
            checks = self.checks(outputs) if self.checks is not None else ()
        return tuple(result), checks


def _validate_checks(checks):
    if not checks:
        return
    # Batch each device's predicates; CPU-resident calibration is valid too.
    by_device = {}
    for predicate, message in checks:
        by_device.setdefault(predicate.device, []).append((predicate, message))
    for group in by_device.values():
        passed = torch.stack([predicate for predicate, _ in group]).cpu().tolist()
        for ok, (_, message) in zip(passed, group, strict=False):
            if not ok:
                raise ValueError(message)


def _same_geometry(destination, value):
    # A derived result may be computed on CUDA for a CPU consumer (or vice
    # versa). The paused commit owns the cross-device copy.
    if not isinstance(value, torch.Tensor) or destination.shape != value.shape or destination.dtype != value.dtype:
        raise ValueError("GPU delta derived destination geometry changed")


def _storage_key(tensor):
    return tensor.device, tensor.untyped_storage().data_ptr()


def _element_offsets(tensor):
    return [
        tensor.storage_offset() + sum(i * stride for i, stride in zip(index, tensor.stride(), strict=False))
        for index in product(*(range(size) for size in tensor.shape))
    ]


def plan_updates(updates, targets):
    """Validate and cache scalar patch geometry once at admission.

    Each plan is ``(update, ((source, targets), ...))``.
    """
    by_storage = {}
    for target in targets:
        by_storage.setdefault(_storage_key(target), []).append(target)
    plans = []
    for update in updates:
        sources = []
        for original in update.sources:
            patches = by_storage.get(_storage_key(original), ())
            if not patches:
                continue
            if not original.is_contiguous():
                raise ValueError("GPU delta derived source must be contiguous")
            start, stop = (
                original.storage_offset(),
                original.storage_offset() + original.numel(),
            )
            written = set()
            for target in patches:
                if target.dtype != original.dtype:
                    raise ValueError("GPU delta scalar view dtype mismatch")
                offsets = set(_element_offsets(target))
                if any(offset < start or offset >= stop for offset in offsets):
                    raise ValueError("GPU delta scalar view is outside its source")
                if len(offsets) != target.numel() or offsets & written:
                    raise ValueError("GPU delta scalar updates overlap")
                written.update(offsets)
            sources.append((original, tuple(patches)))
        plans.append((update, tuple(sources)))
    return tuple(plans)


def prepare_updates(plans, raw_values):
    """Patch cached scalar views, then validate device predicates once per update.

    ``raw_values`` maps ``id(target)`` of a raw binding target to its predicted value.
    """
    prepared, checks = [], []
    for update, sources in plans:
        overrides = {}
        for original, targets in sources:
            changed = [target for target in targets if id(target) in raw_values]
            if not changed:
                continue
            # Admission proves the patches are disjoint and within this source.
            # Read live values only when some elements remain unchanged.
            complete = (
                torch.empty_like(original)
                if sum(target.numel() for target in changed) == original.numel()
                else original.detach().clone()
            )
            destinations, values = [], []
            for target in changed:
                offset = target.storage_offset() - original.storage_offset()
                destinations.append(complete.as_strided(target.shape, target.stride(), offset))
                values.append(raw_values[id(target)])
            torch._foreach_copy_(destinations, values)
            overrides[id(original)] = complete
        if overrides:
            copies, predicates = update._prepare(overrides)
            prepared.extend(copies)
            checks.extend(predicates)
    _validate_checks(checks)
    return tuple(prepared)


def _normal_scale_helpers():
    # Keep dependencies lazy: byte-layout planning need not load inference
    # kernels merely to describe an update.
    from sglang.srt.layers.quantization.modelopt_quant import _compute_gemm1_alphas, _input_scale_to_local_experts

    return _compute_gemm1_alphas, _input_scale_to_local_experts


def _cutedsl_scale_helper():
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import resolve_cutedsl_standard_scales

    return resolve_cutedsl_standard_scales


def _trtllm_scale_helper():
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import _compute_g1_scale_c

    return _compute_g1_scale_c


def nvfp4_derived_update(layer, layout):
    """Describe the selected FlashInfer consumer's small NVFP4 scale updates.

    This mirrors the ordinary loader's scale policy and reuses its pure helpers.
    It neither invokes weight postprocessing nor replaces captured tensors.
    MegaMOE's input normalization is a captured scalar, so a change to that
    scalar is rejected during preparation, before any live weight is modified.
    """
    # identify_layout read the backend and quant mode from the warmed consumer;
    # finish() rechecks that consumer's identity before building this update.
    backend = layout.backend
    if backend == "trtllm":
        trtllm, cute, mega = True, False, False
    elif backend == "cutedsl":
        trtllm, cute, mega = False, True, False
    elif backend == "megamoe":
        trtllm, cute, mega = False, False, True
    else:
        raise ValueError(f"unsupported NVFP4 derived consumer: {backend}")
    if layout.quant_mode == "w4a4":
        w4a16 = False
    elif layout.quant_mode == "w4a16" and not trtllm:
        w4a16 = True
    else:
        raise ValueError(f"unsupported NVFP4 derived quant mode for {backend}: {layout.quant_mode!r}")
    per_token = bool(layer.quant_config.use_per_token_activation)
    neutral_input = w4a16 or per_token
    config = layer.moe_runner_config
    gated, activation = config.is_gated, config.activation
    local_experts, experts, ep_rank = (
        layer.num_local_experts,
        layer.num_experts,
        layer.moe_ep_rank,
    )
    scalar_input13 = layer.w13_input_scale_quant.ndim == 0
    source_names = ["w13_weight_scale_2", "w2_weight_scale_2"]
    if not neutral_input:
        source_names.extend(("w13_input_scale", "w2_input_scale"))
    sources = tuple(
        getattr(layer, name) for name in source_names
    )  # config-access-exempt: Resolve admitted named calibration buffers on the inference layer.
    destinations = [layer.g1_alphas, layer.g1_alphas_up, layer.g2_alphas]

    # W4A16 already has neutral activation constants. Retain its direct-copy
    # refresh rather than recomputing those constants every update.
    if not w4a16:
        destinations.extend((layer.w13_input_scale_quant, layer.w2_input_scale_quant))
    if trtllm:
        destinations.append(layer.g1_scale_c)
    extra_names = []
    if trtllm and gated:
        if (
            getattr(layer, "gemm1_clamp_limit", None) is not None
        ):  # config-access-exempt: The selected TRTLLM inference path may retain a clamp image.
            extra_names.append("gemm1_clamp_limit")
        if (
            getattr(layer, "gemm1_beta", None) is not None
        ):  # config-access-exempt: The selected TRTLLM inference path may retain a beta image.
            extra_names.append("gemm1_beta")
        destinations.extend(
            getattr(layer, name) for name in extra_names
        )  # config-access-exempt: Resolve admitted named calibration buffers on the inference layer.
    if cute:
        scales = layer._cutedsl_scales
        input_scale = layer._cutedsl_input_scale
        if not isinstance(scales, tuple) or len(scales) != 3:
            raise ValueError("CuTe DSL scale metadata must be warmed before admission")
        destinations.extend((scales[0], scales[2]) if w4a16 else (*scales, input_scale))
    mega_scalar = None
    if mega and not w4a16:
        if per_token:
            raise ValueError("FlashInfer MegaMOE does not support per-token NVFP4")
        consumer = layer._flashinfer_megamoe_layer
        cached = layer._flashinfer_megamoe_input_norm_const
        if consumer is None or cached is None:
            raise ValueError("FlashInfer MegaMOE must be warmed before delta admission")
        mega_scalar = float(consumer._megakernel_config.input_norm_const)
        if float(cached) != mega_scalar:
            raise ValueError("FlashInfer MegaMOE input normalization cache disagrees")
    for tensor in (*sources, *destinations):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise ValueError("NVFP4 derived calibration requires FP32 scalar tensors")

    # Resolve pure inference helpers and configuration once at admission.
    if not w4a16:
        gemm1_alphas, local_input = _normal_scale_helpers()
    if cute and not w4a16:
        cutedsl_values = _cutedsl_scale_helper()
        fixed_input_scale = layer._cutedsl_input_scale
    if trtllm:
        trtllm_scale_c = _trtllm_scale_helper()
    clamp = config.gemm1_clamp_limit or config.swiglu_limit
    if "gemm1_clamp_limit" in extra_names and (clamp is None or activation == "situ"):
        raise ValueError("unsupported TRTLLM activation clamp metadata")
    tied_clamp = "gemm1_clamp_limit" in extra_names
    tied_clamp_error = (
        "GPU delta requires tied TRTLLM gate/up scales with activation clamping; "
        "unequal scales require block-scale requantization"
    )
    if tied_clamp:
        # The ordinary loader otherwise rescales one half of the checkpoint
        # block scales. Those bytes cannot be updated by canonical XOR masks.
        weight13 = layer.w13_weight_scale_2
        _validate_checks((((weight13[:, 0] == weight13[:, 1]).all(), tied_clamp_error),))
    situ_beta = config.gemm1_clamp_limit
    extra_outputs = []
    for name in extra_names:
        if name == "gemm1_clamp_limit":
            extra_outputs.append(lambda shadow: (clamp / shadow.g1_alphas).to(torch.float32))
        elif name == "gemm1_beta":
            # SGLang's get_activation_type admits silu, gelu and situ for gated TRTLLM experts.
            if activation == "situ":
                extra_outputs.append(lambda shadow: torch.full_like(shadow.g1_alphas, situ_beta))
            elif activation in {"silu", "gelu"}:
                extra_outputs.append(lambda shadow: (1 / shadow.g1_alphas_up).to(torch.float32))
            else:
                raise ValueError(f"unsupported TRTLLM gemm1_beta activation: {activation}")
        else:
            raise ValueError(f"unsupported TRTLLM GEMM1 image: {name}")
    neutral13, neutral2 = layer.w13_input_scale_quant, layer.w2_input_scale_quant
    if per_token and not w4a16:
        compute_device = next((source.device for source in sources if source.is_cuda), sources[0].device)
        neutral13, neutral2 = neutral13.to(compute_device), neutral2.to(compute_device)
        if cute:
            fixed_input_scale = fixed_input_scale.to(compute_device)

    def gate_up(weight13):
        if gated and weight13.ndim == 2 and weight13.shape[1] >= 2:
            return weight13[:, 0], weight13[:, 1]
        gate = weight13.reshape(weight13.shape[0])
        return gate, gate

    def compute(values):
        weight13, weight2 = values[:2]
        if w4a16:
            gate, up = gate_up(weight13)
            return (gate, up, weight2, gate, weight2) if cute else (gate, up, weight2)

        shadow = SimpleNamespace(
            num_local_experts=local_experts,
            num_experts=experts,
            moe_ep_rank=ep_rank,
            w13_weight_scale_2=weight13,
            w2_weight_scale_2=weight2,
        )
        if neutral_input:
            # Per-token consumers already have neutral activation constants.
            # Reuse those captured tensors rather than allocating ones and
            # recomputing their reciprocals for every update.
            shadow.g1_alphas, shadow.g1_alphas_up = gate_up(weight13)
            shadow.g2_alphas = weight2
            shadow.w13_input_scale_quant = neutral13
            shadow.w2_input_scale_quant = neutral2
        else:
            raw13, raw2 = values[2:]
            input13 = raw13.max().to(torch.float32)
            if cute or mega:
                input2 = local_input(raw2, local_experts, experts, ep_rank)
                if cute:
                    input13 = input13.expand(local_experts)
                    if scalar_input13:
                        input13 = input13[0]
            else:
                input2 = raw2.max().to(torch.float32)
            shadow.g1_alphas, shadow.g1_alphas_up = gemm1_alphas(weight13, input13, gated)
            shadow.g2_alphas = (input2 * weight2).to(torch.float32)
            shadow.w13_input_scale_quant = (1 / input13).to(torch.float32)
            shadow.w2_input_scale_quant = (1 / input2).to(torch.float32)
        outputs = [
            shadow.g1_alphas,
            shadow.g1_alphas_up,
            shadow.g2_alphas,
            shadow.w13_input_scale_quant,
            shadow.w2_input_scale_quant,
        ]
        if trtllm:
            outputs.append(
                trtllm_scale_c(
                    shadow.w2_input_scale_quant,
                    shadow.g1_alphas,
                    shadow.g1_alphas_up,
                    gated,
                    activation,
                )
            )
        outputs.extend(output(shadow) for output in extra_outputs)
        if cute:
            w1, fc2_input, w2, input_scale = cutedsl_values(shadow)
            if per_token:
                input_scale = fixed_input_scale
            outputs.extend((w1, fc2_input, w2, input_scale))
        return tuple(outputs)

    def checks(outputs):
        predicates = []
        if mega and not w4a16:
            predicates.append(
                (
                    # Match the former Python-float comparison exactly, including a
                    # captured scalar that is not representable as an FP32 value.
                    outputs[3].max().to(torch.float64) == mega_scalar,
                    "FlashInfer MegaMOE input normalization changed; CUDA graph recapture is required",
                )
            )
        if mega and gated:
            predicates.append(
                (
                    torch.isclose(outputs[0], outputs[1]).all(),
                    "FlashInfer MegaMOE requires matching gate/up FC1 alphas",
                )
            )
        if tied_clamp:
            predicates.append(((outputs[0] == outputs[1]).all(), tied_clamp_error))
        return tuple(predicates)

    return DerivedUpdate(sources, tuple(destinations), compute, checks if mega or tied_clamp else None)
