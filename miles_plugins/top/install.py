"""Patches for ops with no seam. Each one named, guarded, and asserted to have applied."""

from __future__ import annotations

import torch

_INSTALLED: dict[str, bool] = {}


def install_backward_without_batch_invariance() -> bool:
    """Run Megatron's backward step with the batch-invariant overrides off.

    Backward needs ordinary accuracy only, and its GEMM shapes are ones the strict
    serving matmul rejects. The schedules call backward_step by module name.
    """
    if _INSTALLED.get("backward_without_batch_invariance"):
        return True
    from megatron.core.pipeline_parallel import schedules
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode

    original = getattr(
        schedules, "backward_step", None
    )  # config-access-exempt: module variants differ in whether they define backward_step
    if original is None:
        raise RuntimeError("[top] backward patch: schedules.backward_step missing; megatron changed shape")

    def backward_step(*args, **kwargs):
        with set_batch_invariant_mode(False):
            return original(*args, **kwargs)

    schedules.backward_step = backward_step
    _INSTALLED["backward_without_batch_invariance"] = True
    return True


def install_serving_bi_overloads() -> bool:
    """Bind the aten overloads SGLang's BI mode replaces and Megatron's does not to Megatron's BI switch.

    SGLang code run in the trainer (the DS4 compressor's out_dtype GEMM, F.rms_norm) must dispatch as it does
    in serving; on with Megatron's overrides, off with them in backward and references."""
    if _INSTALLED.get("serving_bi_overloads"):
        return True
    from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import _mm_dtype_compat, _rms_norm_aten_compat

    state = {"lib": None}
    enable, disable = bik.enable_batch_invariant_mode, bik.disable_batch_invariant_mode

    def register():
        if state["lib"] is None:
            lib = torch.library.Library("aten", "IMPL")
            lib.impl("aten::mm.dtype", _mm_dtype_compat, "CUDA")
            lib.impl("aten::rms_norm", _rms_norm_aten_compat, "CUDA")
            state["lib"] = lib

    def enable_batch_invariant_mode(*args, **kwargs):
        enable(*args, **kwargs)
        register()

    def disable_batch_invariant_mode():
        if state["lib"] is not None:
            state["lib"]._destroy()
            state["lib"] = None
        disable()

    # set_batch_invariant_mode calls these through the module globals.
    bik.enable_batch_invariant_mode, bik.disable_batch_invariant_mode = (
        enable_batch_invariant_mode,
        disable_batch_invariant_mode,
    )
    if bik.is_batch_invariant_mode_enabled():
        register()
    _INSTALLED["serving_bi_overloads"] = True
    return True


def install_batch_invariant_recompute() -> bool:
    """Activation recompute replays the forward inside backward_step, where the BI overrides are off; replay it
    with them on, so the recomputed activations (and DS4 routing and C4 selections) are the scored ones."""
    if _INSTALLED.get("batch_invariant_recompute"):
        return True
    from megatron.core import tensor_parallel
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode

    original = tensor_parallel.checkpoint

    def checkpoint(function, distribute_saved_activations, *args):
        def replay(*inputs):
            with set_batch_invariant_mode(True):
                return function(*inputs)

        return original(replay, distribute_saved_activations, *args)

    # transformer_block calls it through the tensor_parallel package.
    tensor_parallel.checkpoint = checkpoint
    _INSTALLED["batch_invariant_recompute"] = True
    return True


def refuse_unpatched_backward(config) -> None:
    """Backward paths that do not go through schedules.backward_step would run with the
    serving overrides on, and activation recompute must replay the forward with them on."""
    # install_batch_invariant_recompute wraps tensor_parallel.checkpoint, which full recompute uses unless Megatron
    # FP8/FP4 routes it through te_checkpoint; selective recompute checkpoints modules through other paths.
    granularity = getattr(
        config, "recompute_granularity", None
    )  # config-access-exempt: TransformerConfig versions and model families differ in optional recompute_granularity
    if granularity not in (None, "full"):
        raise NotImplementedError("[top] selective activation recompute is not replayed under batch invariance")
    if granularity and (
        getattr(config, "fp8", None) or getattr(config, "fp4", None)
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional fp8
        raise NotImplementedError("[top] full recompute with Megatron FP8/FP4 replays through te_checkpoint")
    if getattr(
        config, "overlap_moe_expert_parallel_comm", False
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional overlap_moe_expert_parallel_comm
        raise NotImplementedError("[top] combined 1F1B backward bypasses the backward patch")
    if getattr(config, "cuda_graph_impl", "none") != "none" or getattr(
        config, "external_cuda_graph", False
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional external_cuda_graph
        raise NotImplementedError("[top] CUDA-graph backward bypasses the backward patch")


def installed() -> dict[str, bool]:
    return dict(_INSTALLED)


def install_fused_rope() -> bool:
    """Bind the function SelfAttention actually calls, without replacing its forward."""
    from megatron.core.models.common.embeddings import rope_utils
    from megatron.core.transformer import attention

    from miles_plugins.top.rope import apply_fused_rope

    current = getattr(
        attention, "apply_rotary_pos_emb", None
    )  # config-access-exempt: module variants differ in whether they define apply_rotary_pos_emb
    if current is not apply_fused_rope:
        if current is None or current is not rope_utils.apply_rotary_pos_emb:
            raise RuntimeError("[top] fused-RoPE patch: unexpected SelfAttention call site")
        attention.apply_rotary_pos_emb = apply_fused_rope
    if attention.apply_rotary_pos_emb is not apply_fused_rope:
        raise RuntimeError("[top] fused-RoPE patch did not bind SelfAttention")
    _INSTALLED["fused_rope"] = True
    return True
