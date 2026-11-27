"""Delegate BF16 expert GEMMs, padding each expert segment to the required 128 rows."""

from __future__ import annotations

import torch

from miles_plugins.top.kernels.activation import SiluAndMulFunction

ALIGN = 128


def _pad_plan(tokens_per_expert, device):
    """Positions of the real rows inside the 128-aligned padded buffer, plus its m_indices.

    Sizes come from the host counts (DeepEP reports them on the host), so building the plan does not sync the GPU;
    counts given on the device are copied to the host once.
    """
    counts_cpu = torch.as_tensor(tokens_per_expert).to(device="cpu", dtype=torch.int64)
    padded_cpu = (counts_cpu + ALIGN - 1) // ALIGN * ALIGN
    total, padded_total = int(counts_cpu.sum()), int(padded_cpu.sum())
    counts = counts_cpu.pin_memory().to(device, non_blocking=True)
    padded = padded_cpu.pin_memory().to(device, non_blocking=True)
    real_starts = torch.cumsum(counts, 0) - counts
    padded_starts = torch.cumsum(padded, 0) - padded
    experts = torch.arange(counts.numel(), device=device)
    row_expert = torch.repeat_interleave(experts, counts, output_size=total)
    intra = torch.arange(total, device=device) - real_starts[row_expert]
    real_pos = padded_starts[row_expert] + intra
    m_indices = torch.repeat_interleave(experts.to(torch.int32), padded, output_size=padded_total)
    return (real_pos, m_indices, padded_total)


class _DeepGemmGroupedLinear(torch.autograd.Function):
    """Delegated forward over the 128-padded segments; backward is BF16 grouped GEMMs over
    the unpadded rows, with each expert's weight gradient returned to its own parameter."""

    @staticmethod
    def forward(ctx, a, real_pos, m_indices, padded_total, offsets, fusion, *weights):
        from sglang.srt.layers.deep_gemm_wrapper.entrypoint import grouped_gemm_nt_bf16_contig

        ctx.save_for_backward(a, offsets)
        ctx.weights = weights
        ctx.fusion = fusion
        if padded_total == 0:
            return a.new_empty((0, weights[0].shape[0]))
        w = torch.stack(weights)
        a_pad = torch.zeros(padded_total, a.shape[1], device=a.device, dtype=a.dtype)
        a_pad[real_pos] = a
        d_pad = torch.empty(padded_total, w.shape[1], device=a.device, dtype=a.dtype)
        grouped_gemm_nt_bf16_contig(a_pad, w, d_pad, m_indices)
        return d_pad.index_select(0, real_pos)

    @staticmethod
    def backward(ctx, grad_out):
        a, offsets = ctx.saved_tensors
        weights = ctx.weights
        dy = grad_out.contiguous()
        grad_a = torch._grouped_mm(dy, torch.stack(weights), offs=offsets)
        dw = torch._grouped_mm(dy.t(), a, offs=offsets)
        grads = []
        for i, weight in enumerate(weights):
            if not ctx.fusion:
                grads.append(dw[i])
                continue
            # Same accumulator and hook protocol as the dense TOP linears.
            from transformer_engine.pytorch.module.base import get_dummy_wgrad

            main_grad = getattr(
                weight, "main_grad", None
            )  # config-access-exempt: Megatron attaches main_grad to parameters only under some gradient-buffer modes
            if main_grad is None or main_grad.dtype != torch.float32:
                raise RuntimeError("[top] expert gradient fusion requires FP32 weight.main_grad")
            main_grad.add_(dw[i])
            grad = None
            if hasattr(
                weight, "grad_added_to_main_grad"
            ):  # config-access-exempt: Megatron attaches grad_added_to_main_grad to parameters only under some gradient-buffer modes
                grad = get_dummy_wgrad(
                    list(weight.shape), weight.dtype, zero=getattr(weight, "zero_out_wgrad", False)
                )  # config-access-exempt: Megatron attaches zero_out_wgrad to parameters only under some gradient-buffer modes
                weight.grad_added_to_main_grad = True
            grads.append(grad)
        return grad_a, None, None, None, None, None, *grads


def deepgemm_expert_rows(
    dispatched_input, w1, w2, tokens_per_expert, *, gradient_accumulation_fusion=False, swiglu_limit=None
):
    """Return unweighted rows; routing weights belong in the owner-side combine. With ``swiglu_limit`` (GLM-5.3-Flash)
    gate and up are clamped in BF16 before the SiLU, as serving's TOP experts do (moe.top_clamp_swiglu_input).

    w1 and w2 are per-expert weight sequences, so each parameter receives its own gradient."""
    if dispatched_input.dtype != torch.bfloat16 or any(w.dtype != torch.bfloat16 for w in (*w1, *w2)):
        raise NotImplementedError("[top] expert GEMMs require BF16 inputs and weights")
    device = dispatched_input.device
    # Host counts (a device tensor would sync once here).
    counts_cpu = torch.as_tensor(tokens_per_expert).to(device="cpu", dtype=torch.int64)
    if len(counts_cpu) != len(w1) or len(counts_cpu) != len(w2) or int(counts_cpu.sum()) != dispatched_input.shape[0]:
        raise ValueError("[top] expert counts do not match the weights and dispatched rows")
    real_pos, m_indices, padded_total = _pad_plan(counts_cpu, device)
    offsets = torch.cumsum(counts_cpu, 0).to(torch.int32).pin_memory().to(device, non_blocking=True)
    args = (real_pos, m_indices, padded_total, offsets, gradient_accumulation_fusion)
    gate_up = _DeepGemmGroupedLinear.apply(dispatched_input.contiguous(), *args, *w1)
    if swiglu_limit is not None:
        gate, up = gate_up.chunk(2, dim=-1)
        gate_up = torch.cat((gate.clamp(max=swiglu_limit), up.clamp(min=-swiglu_limit, max=swiglu_limit)), dim=-1)
    act = SiluAndMulFunction.apply(gate_up)
    return _DeepGemmGroupedLinear.apply(act.contiguous(), *args, *w2)
