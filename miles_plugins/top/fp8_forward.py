"""FP8 parameter adapters delegating arithmetic to SGLang.

Checkpoint payloads and scales stay paired. The adapters hold BF16 masters (fp8_masters)
behind serving's processed FP8. Their backward differentiates BF16 GEMMs at the served
weights (the processed bytes' dequantization, rebuilt in backward), with the gradient
straight through to the masters.
"""

import torch

from miles_plugins.top.kernels.deepgemm_experts import _pad_plan


def prepare_weight(weight, scale):
    """Use the serving loader's exact Blackwell weight requantization."""
    from sglang.srt.layers.quantization.fp8_utils import requant_weight_ue8m0

    if weight.dtype != torch.float8_e4m3fn or scale.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Expected checkpoint E4M3 payload and FP32 or BF16 block scales")
    # Serving loads scales into FP32 parameters; widening BF16 scales is exact.
    return requant_weight_ue8m0(weight.contiguous(), scale.float().contiguous(), [128, 128])


def synced_weight(weight, scale):
    """What serving holds after a weight sync of a UE8M0 pair: Miles sends the packed scales and serving
    keeps the bytes as sent. Requantizing them like a checkpoint (prepare_weight) is not idempotent."""
    from sglang.srt.layers.quantization.fp8_utils import transform_scale_ue8m0

    return weight, transform_scale_ue8m0(scale, mn=weight.shape[-2])


def dequantize_processed(payload, scale):
    """The BF16 weight a processed pair holds, exactly: E4M3 ``payload`` [..., n, k] with ``scale`` either serving's
    packed UE8M0 words, expanded per row ([..., n, ceil(k / 512)] int32, prepare_weight / synced_weight), or FP32
    128x128 block scales [..., ceil(n / 128), ceil(k / 128)]. On CUDA one fused kernel (kernels.fp8_dequant) writes
    the bytes of the eager decode below without its FP32 temporaries."""
    *lead, n, k = payload.shape
    groups = (k + 127) // 128
    if scale.dtype == torch.int32:
        if scale.shape[-2] != n or scale.shape[-1] * 4 < groups:
            raise ValueError(f"[top] packed UE8M0 scales {tuple(scale.shape)} are not per-row for a {n}x{k} weight")
    elif scale.dtype != torch.float32:
        raise ValueError(f"[top] no served value for {scale.dtype} block scales")
    elif tuple(scale.shape[-2:]) != ((n + 127) // 128, groups):
        raise ValueError(f"[top] FP32 scales {tuple(scale.shape)} are not 128x128 blocks of a {n}x{k} weight")
    flat_payload, flat_scale = payload.reshape(-1, n, k), scale.reshape(-1, *scale.shape[-2:])
    if flat_scale.shape[0] != flat_payload.shape[0]:
        raise ValueError(f"[top] {flat_scale.shape[0]} scale matrices for {flat_payload.shape[0]} weights")
    if not (payload.is_cuda and payload.dtype == torch.float8_e4m3fn):
        return _dequantize_eager(flat_payload, flat_scale).view(payload.shape)
    from miles_plugins.top.kernels.fp8_dequant import dequantize_blocks

    if flat_payload.stride(-1) != 1:
        flat_payload = flat_payload.contiguous()
    return dequantize_blocks(flat_payload, flat_scale).view(payload.shape)


def _dequantize_eager(payload, scale):
    """dequantize_processed for validated [m, n, k] / [m, ...] operands, one matrix (expert) at a time, so the FP32
    temporaries stay one matrix large."""
    m, n, k = payload.shape
    groups = (k + 127) // 128
    if scale.dtype == torch.int32:
        words = scale[..., ::128, :]
        exponent = (words.unsqueeze(-1) >> torch.tensor([0, 8, 16, 24], device=scale.device, dtype=torch.int32)) & 255
        exponent = exponent.flatten(-2)[..., :groups]
        # Hopper's unpack_ue8m0: exponent 0 is 2^-127, 255 is NaN.
        bits = torch.where(exponent == 0, 0x00400000, torch.where(exponent == 255, 0x7FC00000, exponent << 23))
        scale = bits.to(torch.int32).view(torch.float32)
    out = torch.empty((m, n, k), dtype=torch.bfloat16, device=payload.device)
    for i in range(m):
        # Power-of-two scales times E4M3 values are exact in BF16.
        if n % 128 == 0 and k % 128 == 0:
            blocks = payload[i].view(n // 128, 128, groups, 128).float() * scale[i][:, None, :, None]
            out[i] = blocks.view(n, k).to(torch.bfloat16)
        else:
            block = scale[i].repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
            out[i] = (payload[i].float() * block).to(torch.bfloat16)
    return out


def expert_rows(x, packed_weights, tokens_per_expert, swiglu_limit=None):
    """Serving's FP8 experts over Megatron's packed rows; the forward of trainable_expert_rows."""
    from sglang.srt.true_on_policy.fp8.fp8_expert_forward import expert_forward

    if torch.is_grad_enabled() and x.requires_grad:
        raise NotImplementedError("[top] expert_rows has no backward; use trainable_expert_rows")
    w13, s13, w2, s2 = packed_weights
    positions, indices, total = _pad_plan(tokens_per_expert, x.device)
    rows = x.shape[0]
    if total == 0:
        return x.new_empty((0, w2.shape[1]))
    # Padded slot -> real row, -1 for alignment padding (left zero, as serving pads).
    source = torch.full((total,), -1, dtype=torch.int64, device=x.device)
    source[positions] = torch.arange(rows, device=x.device)
    # Rows past `rows` are per-slot scratch for padding outputs, so every write target is distinct.
    out = x.new_empty((rows + _EXPERT_FWD_ROWS, w2.shape[1]))
    for lo in range(0, total, _EXPERT_FWD_ROWS):
        src = source[lo : lo + _EXPERT_FWD_ROWS]
        live = src >= 0
        chunk = torch.where(live.unsqueeze(-1), x.index_select(0, src.clamp(min=0)), 0)
        values = expert_forward(
            chunk, (w13, s13), (w2, s2), m_indices=indices[lo : lo + src.numel()], swiglu_limit=swiglu_limit
        )
        out.index_copy_(0, torch.where(live, src, rows + torch.arange(src.numel(), device=x.device)), values)
    return out[:rows]


def _clamped_swiglu_value(gate_up, swiglu_limit):
    from miles_plugins.top.kernels.clamped_swiglu import swiglu_forward

    if gate_up.ndim == 2 and gate_up.is_cuda and gate_up.dtype == torch.bfloat16:
        return swiglu_forward(gate_up.contiguous(), swiglu_limit)
    gate, up = gate_up.chunk(2, dim=-1)
    if swiglu_limit is not None:
        gate, up = gate.clamp(max=swiglu_limit), up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return (torch.nn.functional.silu(gate.float()) * up.float()).to(gate_up.dtype)


# Padded rows per serving-kernel forward chunk; a multiple of the 128-row segment alignment, so each chunk is a
# valid contiguous layout and a row's bytes do not depend on it (batch-invariant DeepGEMM). Arbitrary.
_EXPERT_FWD_ROWS = 32768
# Expert rows per reference-backward chunk. Arbitrary; one 8.7k-token/GPU Base micro-batch
# put ~430k rows on a single EP32 rank's experts.
_EXPERT_BWD_ROWS = 65536


class _ClampedSwiGLU(torch.autograd.Function):
    """Saves only the BF16 gate/up; autograd through the FP32 math keeps four rows x I FP32 tensors,
    which a hot expert's rows turn into the trainer's peak. The backward is one fused pass (kernels/clamped_swiglu).
    """

    @staticmethod
    def forward(ctx, gate_up, swiglu_limit):
        ctx.swiglu_limit = swiglu_limit
        ctx.save_for_backward(gate_up)
        return _clamped_swiglu_value(gate_up, swiglu_limit)

    @staticmethod
    def backward(ctx, grad):
        from miles_plugins.top.kernels.clamped_swiglu import swiglu_backward

        (gate_up,) = ctx.saved_tensors
        return swiglu_backward(gate_up.contiguous(), grad, ctx.swiglu_limit), None


def clamped_swiglu(gate_up, swiglu_limit):
    """SGLang's SwiGLU on BF16 gate/up halves (DeepSeek-V4 clamps both first), FP32 math."""
    if torch.is_grad_enabled() and gate_up.requires_grad:
        return _ClampedSwiGLU.apply(gate_up, swiglu_limit)
    return _clamped_swiglu_value(gate_up, swiglu_limit)


class TrainableFP8Linear(torch.nn.Module):
    """FP8 forward on serving's kernel; backward at the served weight, straight through to ``weight`` (the master)."""

    def __init__(self, payload, scale):
        super().__init__()
        from miles_plugins.top.fp8_masters import FP8Master

        self._fp8 = FP8Master(
            payload, scale, prepare_weight, lambda pair: dequantize_processed(*pair), synced=synced_weight
        )
        self.weight = self._fp8.master
        self.is_top_fp8 = True

    def processed(self):
        return self._fp8.processed()

    def served(self):
        return self._fp8.served()

    def forward(self, x, weight=None, runtime_gather_output=None):
        from sglang.srt.true_on_policy.fp8.fp8_linear import linear

        from miles_plugins.top.kernels.surrogate import linear_surrogate

        if weight is not None or runtime_gather_output:
            raise NotImplementedError("[top] trainable FP8 linear admits no external weight or gather")
        w, s = self.processed()
        return (
            linear_surrogate(lambda x, m: linear(x, w, s), x, self.weight, name="fp8_linear", value_fn=self.served),
            None,
        )

    def reference(self, x):
        """``x @ served.T`` in BF16 for a gradient reference, with the weight gradient going to the master."""
        from miles_plugins.top.kernels.surrogate import linear_surrogate

        return linear_surrogate(
            lambda x, m: x @ self.served().t(), x, self.weight, name="fp8_linear_reference", value_fn=self.served
        )


class _StraightThrough(torch.autograd.Function):
    """Forward returns ``visible`` bit for bit (no ``x + 0`` sign-of-zero change); backward is identity onto ``master``."""

    @staticmethod
    def forward(ctx, visible, master):
        return visible

    @staticmethod
    def backward(ctx, grad):
        return None, grad


class AbsorbedFP8Weight(torch.nn.Module):
    """A block-FP8 weight read as BF16 (MLA kv_b, absorbed into q and o): ``weight`` is the dequantized FP8 form.

    Serving builds w_kc/w_vc from the dequantized FP8 it loaded: the checkpoint at install, the weight-sync
    quantizer's output after an update. Reading the BF16 master instead diverges after the first optimizer step.
    The parameter (what the optimizer and weight sync see) is the master; backward is identity onto it.
    """

    def __init__(self, payload, scale):
        super().__init__()
        from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

        from miles_plugins.top.fp8_masters import FP8Master

        self._fp8 = FP8Master(
            payload, scale, lambda w, s: block_quant_dequant(w, s, [128, 128], torch.bfloat16), lambda w: w.detach()
        )
        # Registered directly: the ``weight`` property below shadows the attribute.
        self._parameters["weight"] = self._fp8.master

    @property
    def weight(self):
        return self._fp8.jacobian_weight()


def trainable_expert_rows(x, experts, tokens_per_expert, swiglu_limit=None):
    """FP8 grouped experts on serving's kernel; backward through BF16 grouped GEMMs at the served weights, straight
    through to the per-expert masters (``experts`` is an ExpertFP8Masters)."""
    from miles_plugins.top.kernels.surrogate import row_surrogate

    # Host counts plan the rows (no sync); the device copy offsets the reference GEMMs.
    counts_cpu = torch.as_tensor(tokens_per_expert).to(device="cpu", dtype=torch.int64)
    counts = counts_cpu.pin_memory().to(x.device, non_blocking=True)
    offsets = torch.cumsum(counts, 0).to(torch.int32)
    packed = experts.processed()
    n = len(experts.w13)
    # The stacked served weights, decoded once at the first backward chunk and freed with the autograd graph.
    served = []

    def served_values():
        if not served:
            served.extend(experts.served())
        return served

    def visible(x, *_):
        return expert_rows(x, packed, counts_cpu, swiglu_limit=swiglu_limit)

    def chunk_reference(lo, hi, x, *masters):
        v13, v2 = served_values()
        w13 = _StraightThrough.apply(v13, torch.stack(masters[:n]))
        w2 = _StraightThrough.apply(v2, torch.stack(masters[n:]))
        offs = (offsets - lo).clamp(0, hi - lo).to(torch.int32)
        gate_up = torch._grouped_mm(x, w13.transpose(1, 2), offs=offs)
        return torch._grouped_mm(clamped_swiglu(gate_up, swiglu_limit), w2.transpose(1, 2), offs=offs)

    def chunk_backward(lo, hi, x, grad, *masters):
        # chunk_reference's gradients in closed form: its down-projection output is never needed.
        w13, w2 = served_values()
        offs = (offsets - lo).clamp(0, hi - lo).to(torch.int32)
        grad = grad.contiguous()
        with torch.enable_grad():
            gate_up = torch._grouped_mm(x, w13.transpose(1, 2), offs=offs).requires_grad_(True)
            h = clamped_swiglu(gate_up, swiglu_limit)
        d_h = torch._grouped_mm(grad, w2, offs=offs)
        d_w2 = torch._grouped_mm(grad.t(), h.detach(), offs=offs)
        (d_gate_up,) = torch.autograd.grad(h, gate_up, d_h)
        d_w13 = torch._grouped_mm(d_gate_up.t(), x, offs=offs)
        d_x = torch._grouped_mm(d_gate_up, w13, offs=offs)
        return d_x, [*d_w13.unbind(0), *d_w2.unbind(0)]

    return row_surrogate(
        visible,
        chunk_reference,
        x.contiguous(),
        *experts.w13,
        *experts.w2,
        chunk_rows=_EXPERT_BWD_ROWS,
        name="fp8_experts",
        chunks=_expert_chunks(counts_cpu.tolist(), n, _EXPERT_BWD_ROWS),
        chunk_backward_fn=chunk_backward,
    )


def _expert_chunks(counts, n, limit):
    """Backward chunks of whole experts up to ``limit`` rows; an expert above ``limit`` is split alone.
    Each entry is (lo, hi, {w13 index, w2 index} of the experts it covers)."""
    chunks, lo, rows, touched = [], 0, 0, set()
    start = 0
    for e, c in enumerate(counts):
        if c > limit:
            if rows:
                chunks.append((lo, start, touched))
            for piece in range(start, start + c, limit):
                chunks.append((piece, min(piece + limit, start + c), {e, n + e}))
            lo, rows, touched = start + c, 0, set()
        elif c:
            if rows + c > limit:
                chunks.append((lo, start, touched))
                lo, rows, touched = start, 0, set()
            rows += c
            touched |= {e, n + e}
        start += c
    if rows:
        chunks.append((lo, start, touched))
    return chunks


def trainable_clamped_swiglu_mlp(x, fc1, fc2, swiglu_limit):
    """DeepSeek-V4 shared expert: serving's FP8 clamped MLP; backward at the served weights, straight through to the
    BF16 masters."""
    from sglang.srt.true_on_policy.fp8.fp8_linear import clamped_swiglu_mlp

    from miles_plugins.top.kernels.surrogate import surrogate

    p1, p2 = fc1.processed(), fc2.processed()
    return surrogate(
        lambda x, w1, w2: clamped_swiglu_mlp(x, p1, p2, swiglu_limit),
        lambda x, w1, w2: clamped_swiglu(x @ _StraightThrough.apply(fc1.served(), w1).t(), swiglu_limit)
        @ _StraightThrough.apply(fc2.served(), w2).t(),
        x,
        fc1.weight,
        fc2.weight,
        name="fp8_shared_expert",
    )
