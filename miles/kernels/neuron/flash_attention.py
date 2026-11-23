"""Packed-sequence causal attention on the nkilib flash kernels (`attention_cte` / `attention_bwd`), for Neuron.

A token's lowest visible key is its segment start (the kernels' sequence-packing bound), its highest the causal mask;
no [T, T] score matrix is built.
"""

import torch
from nkilib.core.attention.attention_bwd import attention_bwd
from nkilib.core.attention.attention_cte import attention_cte
from torch_neuronx import nki_op
from torch_neuronx.nki_hop import wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

# kernel-call glue adapted from aws-neuron/torchtitan-neuron release_2.33.0 models/gpt_oss/nki_attention.py
# (BSD-3, Copyright Amazon.com, Inc. or its affiliates)

wrapped_flash_fwd = wrap_nki(attention_cte)
wrapped_flash_bwd = wrap_nki(attention_bwd)
LNC = (int(get_logical_neuron_cores()),)


def _prescaled(q_HTD: torch.Tensor, scale: float) -> torch.Tensor:
    # the packing path takes scale 1.0: scale q in fp32, round once
    return (q_HTD.float() * scale).to(q_HTD.dtype).contiguous()


@nki_op("miles_nki_attn::packed_causal_fwd")
def packed_causal_attention(
    q_HTD: torch.Tensor, k_KTD: torch.Tensor, v_KTD: torch.Tensor, segment_start_T: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    num_heads, num_tokens, head_dim = q_HTD.shape
    # key range [segment start, T); the causal mask cuts it at the query
    bound_min = segment_start_T.to(torch.float32).reshape(1, num_tokens, 1).expand(num_heads, -1, -1).contiguous()
    bound_max = torch.full_like(bound_min, float(num_tokens))
    out_HTD, neg_max, sum_recip = wrapped_flash_fwd[LNC](
        _prescaled(q_HTD, scale),
        k_KTD.contiguous(),
        v_KTD.contiguous(),
        scale=1.0,
        causal_mask=True,
        tp_q=True,
        tp_k=True,
        cache_softmax=True,
        bound_min=bound_min,
        bound_max=bound_max,
    )
    lse = -(neg_max + torch.log(sum_recip))
    return out_HTD.reshape(num_heads, num_tokens, head_dim), lse.to(torch.float32).reshape(
        1, num_heads, *lse.shape[-2:]
    )


@nki_op("miles_nki_attn::packed_causal_bwd")
def _packed_causal_attention_bwd(
    q_HTD: torch.Tensor,
    k_KTD: torch.Tensor,
    v_KTD: torch.Tensor,
    out_HTD: torch.Tensor,
    grad_out_HTD: torch.Tensor,
    lse: torch.Tensor,
    segment_start_T: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = q_HTD.shape[1]
    bound_min = segment_start_T.to(torch.float32).reshape(1, num_tokens)
    bound_max = torch.full_like(bound_min, float(num_tokens))

    # the backward kernel takes [batch, heads, dim, tokens]
    def to_bhds(x):
        return x.permute(0, 2, 1).unsqueeze(0).contiguous()

    grad_q, grad_k, grad_v = wrapped_flash_bwd[LNC](
        to_bhds(_prescaled(q_HTD, scale)),
        to_bhds(k_KTD),
        to_bhds(v_KTD),
        to_bhds(out_HTD),
        to_bhds(grad_out_HTD),
        lse,
        bound_min=bound_min,
        bound_max=bound_max,
        use_causal_mask=True,
        mixed_precision=True,
        softmax_scale=1.0,
    )[:3]
    # the kernels saw q * scale
    grad_q = grad_q * scale
    return tuple(g.squeeze(0).permute(0, 2, 1).contiguous() for g in (grad_q, grad_k, grad_v))


def _setup_context(ctx, inputs, output):
    q_HTD, k_KTD, v_KTD, segment_start_T, scale = inputs
    out_HTD, lse = output
    ctx.save_for_backward(q_HTD, k_KTD, v_KTD, out_HTD, lse, segment_start_T)
    ctx.scale = scale


def _backward(ctx, grad_out_HTD, _grad_lse):
    q_HTD, k_KTD, v_KTD, out_HTD, lse, segment_start_T = ctx.saved_tensors
    grad_q, grad_k, grad_v = _packed_causal_attention_bwd(
        q_HTD, k_KTD, v_KTD, out_HTD, grad_out_HTD, lse, segment_start_T, ctx.scale
    )
    return grad_q, grad_k, grad_v, None, None


packed_causal_attention.register_autograd(_backward, setup_context=_setup_context)
