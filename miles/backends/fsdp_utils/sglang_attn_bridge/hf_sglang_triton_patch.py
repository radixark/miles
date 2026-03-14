"""Minimal patch: replace HF attention kernel with SGLang Triton extend_attention_fwd_unified.

Instead of monkey-patching the entire attention forward (qkv proj, norm, rope, o_proj),
we register a custom attention backend via HF's ALL_ATTENTION_FUNCTIONS. This way HF
keeps its own qkv_proj, norm, rope, o_proj logic — we ONLY replace the core attention
computation (Q@K^T softmax V) with extend_attention_fwd_unified.

The extend_attention_fwd_unified kernel uses per-request indptrs, which naturally
gives batch-invariant results (each request is computed independently).

The TritonAttnFunction autograd wrapper ensures:
  - Forward: uses SGLang Triton kernel (bitwise-aligned with inference)
  - Backward: computes correct gradients so qkv_proj receives gradient flow
"""

import torch


class TritonAttnFunction(torch.autograd.Function):
    """Autograd wrapper: Triton forward + naive PyTorch backward.

    This gives us true on-policy (fwd matches inference exactly) while
    maintaining full gradient flow through q/k/v projections.
    The backward uses standard PyTorch ops as a reference implementation;
    it can later be replaced with a Triton backward kernel for performance.
    """

    @staticmethod
    def forward(ctx, q, k, v, B, S):
        """
        Args:
            q: [total, num_heads, D], bf16, contiguous
            k: [total, num_kv_heads, D], bf16, contiguous
            v: [total, num_kv_heads, D], bf16, contiguous
            B: batch size (int)
            S: sequence length (int)
        Returns:
            o: [total, num_heads, D], bf16
        """
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd_unified,
        )

        total = B * S
        num_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        D = q.shape[2]
        device = q.device

        o = torch.empty_like(q)
        qo_indptr = torch.arange(0, B + 1, device=device, dtype=torch.int32) * S
        kv_indptr = qo_indptr.clone()
        kv_indices = torch.arange(total, device=device, dtype=torch.int64)
        prefix_lens = torch.zeros(B, device=device, dtype=torch.int32)

        extend_attention_fwd_unified(
            q, o, k, v,
            qo_indptr, kv_indptr, kv_indices, prefix_lens,
            max_len_extend=S, is_causal=True,
        )

        ctx.save_for_backward(q, k, v, o)
        ctx.B = B
        ctx.S = S
        return o

    @staticmethod
    def backward(ctx, grad_output):
        """Naive PyTorch backward for standard causal self-attention.

        Math:
            P = softmax(Q @ K^T / sqrt(d), causal_mask)
            O = P @ V

            dV = P^T @ dO
            dP = dO @ V^T
            dS = P * (dP - (dO * O).sum(-1, keepdim=True))  # softmax bwd
            dQ = dS @ K / sqrt(d)
            dK = dS^T @ Q / sqrt(d)
        """
        q, k, v, o = ctx.saved_tensors
        B, S = ctx.B, ctx.S
        num_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        D = q.shape[2]
        kv_group_num = num_heads // num_kv_heads

        # Reshape to [B, heads, S, D] and upcast to float32 for numerical stability
        q_4d = q.view(B, S, num_heads, D).permute(0, 2, 1, 3).float()
        k_4d = k.view(B, S, num_kv_heads, D).permute(0, 2, 1, 3).float()
        v_4d = v.view(B, S, num_kv_heads, D).permute(0, 2, 1, 3).float()
        do_4d = grad_output.view(B, S, num_heads, D).permute(0, 2, 1, 3).float()
        o_4d = o.view(B, S, num_heads, D).permute(0, 2, 1, 3).float()

        # Expand KV heads for GQA: [B, num_kv_heads, S, D] -> [B, num_heads, S, D]
        if kv_group_num > 1:
            k_4d = k_4d.repeat_interleave(kv_group_num, dim=1)
            v_4d = v_4d.repeat_interleave(kv_group_num, dim=1)

        scale = 1.0 / (D ** 0.5)

        # Recompute attention weights (memory-efficient: don't save P in forward)
        scores = torch.matmul(q_4d, k_4d.transpose(-2, -1)) * scale
        causal_mask = torch.triu(
            torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1
        )
        scores.masked_fill_(causal_mask, float('-inf'))
        p = torch.softmax(scores, dim=-1)  # [B, H, S, S]

        # dV = P^T @ dO
        dv_4d = torch.matmul(p.transpose(-2, -1), do_4d)

        # dP = dO @ V^T
        dp = torch.matmul(do_4d, v_4d.transpose(-2, -1))

        # Softmax backward: dS = P * (dP - sum(dO * O, dim=-1, keepdim=True))
        sum_do_o = (do_4d * o_4d).sum(-1, keepdim=True)
        ds = p * (dp - sum_do_o) * scale

        # dQ = dS @ K, dK = dS^T @ Q
        dq_4d = torch.matmul(ds, k_4d)
        dk_4d = torch.matmul(ds.transpose(-2, -1), q_4d)

        # Reduce GQA gradients: [B, num_heads, S, D] -> [B, num_kv_heads, S, D]
        if kv_group_num > 1:
            dk_4d = dk_4d.view(B, num_kv_heads, kv_group_num, S, D).sum(dim=2)
            dv_4d = dv_4d.view(B, num_kv_heads, kv_group_num, S, D).sum(dim=2)

        # Reshape back to [total, heads, D] and cast back to input dtype
        dq = dq_4d.permute(0, 2, 1, 3).contiguous().view(B * S, num_heads, D).to(q.dtype)
        dk = dk_4d.permute(0, 2, 1, 3).contiguous().view(B * S, num_kv_heads, D).to(k.dtype)
        dv = dv_4d.permute(0, 2, 1, 3).contiguous().view(B * S, num_kv_heads, D).to(v.dtype)

        return dq, dk, dv, None, None  # None for B, S


def _sglang_triton_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Drop-in replacement for HF's attention_interface.

    Input:  query [B, num_heads, S, D], key/value [B, num_kv_heads, S, D]
    Output: attn_output [B, S, num_heads, D], None
    """
    del attention_mask, scaling, dropout, kwargs

    B, num_heads, S, D = query.shape
    num_kv_heads = key.shape[1]
    total = B * S

    q = query.transpose(1, 2).contiguous().view(total, num_heads, D)
    k = key.transpose(1, 2).contiguous().view(total, num_kv_heads, D)
    v = value.transpose(1, 2).contiguous().view(total, num_kv_heads, D)

    # Force kernel inputs to bf16 right before extend_unified to match
    # the SGLang inference-side Triton backend behavior.
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)

    o = TritonAttnFunction.apply(q, k, v, B, S)

    attn_output = o.view(B, S, num_heads, D)
    return attn_output, None


def apply_sglang_triton_attention_patch(model):
    """Register SGLang Triton as attention backend and activate it on the model."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["triton"] = _sglang_triton_attention
    model.config._attn_implementation = "triton"

    patched = sum(
        1
        for _, m in model.named_modules()
        if hasattr(m, "q_proj") and hasattr(m, "o_proj")
    )
    return patched
