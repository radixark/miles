"""True on-policy training on TPU: the trainer's forward runs the rollout engine's own batch-invariant ops.

sglang-torchtpu's true on-policy mode (SGLANG_TPU_ENABLE_TRUE_ON_POLICY=1) serves every linear, RMSNorm, the attention
and the returned log-probs with programs whose bytes for a row do not depend on the batch
(sglang_tpu.rl.true_on_policy).
Here each block of a dense Qwen3 or Qwen3.5-family model (Qwen3.5, Qwen3.6, Qwen3.8), the final norm and the lm_head
call the same ops in the engine's order, so the trainer's log-probs equal the rollout's bit for bit. Engine habits:
- the residual add happens inside the next norm's kernel (Qwen3 normalizes the unrounded fp32 sum, Qwen3.5 rounds it
  to bf16 first); blocks pass [hidden | residual] (2 x dim) to each other instead of their bf16 sum;
- RoPE is sglang's RotaryEmbedding, with its cos/sin table built on the device as the engine builds it at load;
- every parameter is read in the serving dtype, bf16, including the ones a checkpoint stores in fp32.
The backward is the gradient of the same math (plain torch, splash attention for the attention, the trainer's own
kernel for the gated delta net); it does not need to be bitwise.

TPU only: it imports sglang_tpu, torch_tpu and jax.
"""

import functools

import torch
import torch.nn.functional as F
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang_tpu.rl import true_on_policy as ops
from torchtitan.models.common.attention.attention import GQAttention, QKVLinear
from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.common.moe import MoE
from torchtitan.models.qwen3_5.model import Qwen35Attention

from miles.backends.torchtitan_utils import packed_sequences
from miles.kernels.tpu.flash_attention import _backward_op as _splash_backward
from miles.kernels.tpu.flash_attention import _forward_op as _splash_forward

# Sequence slots of the packed attention, a static count: a micro-batch's samples and its padding tail must fit
# (check_microbatch). Arbitrary; each slot adds a page-table row and one scratch KV page to every attention call.
MAX_SEQUENCES = 128


def install(model: torch.nn.Module) -> None:
    """Route a dense Qwen3 or Qwen3.5-family forward through the engine's ops (before its blocks are compiled)."""
    blocks = list(model.get_submodule("layers").children())
    _check_supported(blocks)
    # Qwen3.5's norms scale by 1 + weight and round the residual sum first (the engine's GemmaRMSNorm)
    gemma = any(isinstance(block.attention, (Qwen35Attention, GDN)) for block in blocks)
    rope = _engine_rope(next(block.attention for block in blocks if not isinstance(block.attention, GDN)))
    for block in blocks:
        block.forward = functools.partial(_block_forward, block, _attend(block.attention, rope), gemma)
    model.norm.forward = functools.partial(_final_norm_forward, model.norm, gemma)
    model.lm_head.forward = functools.partial(_lm_head_forward, model.lm_head)


def log_probs(logits: torch.Tensor, targets: torch.Tensor, temperature: float) -> torch.Tensor:
    """The engine's returned log-prob: log_softmax(bf16 logits / temperature)[target], from the same kernel."""
    rows = logits.shape[0]
    t = torch.full((rows,), temperature, dtype=torch.float32, device=logits.device)
    return ops.logprob(logits, t, targets.to(torch.int32))


def check_microbatch(positions: torch.Tensor) -> None:
    """Every packed sequence gets its own KV pages: refuse more than MAX_SEQUENCES."""
    sequences = packed_sequences.count_sequences(positions)
    if sequences > MAX_SEQUENCES:
        raise ValueError(f"true on-policy attention packs at most {MAX_SEQUENCES} sequences, got {sequences}")


def _check_supported(blocks) -> None:
    for block in blocks:
        attention = block.attention
        if isinstance(block.feed_forward, MoE):
            raise ValueError("true on-policy training on TPU covers dense feed-forward blocks, not MoE")
        if isinstance(attention, GDN):
            continue
        if isinstance(attention, GQAttention):
            fused_qkv = isinstance(attention.qkv_linear, QKVLinear)
            if not fused_qkv or attention.q_norm is None or attention.rope is None:
                raise ValueError("true on-policy training on TPU covers Qwen3 attention (fused QKV, q/k norm, RoPE)")
        elif not isinstance(attention, Qwen35Attention):
            raise ValueError(f"true on-policy training on TPU does not cover {type(attention).__name__}")
        if attention.scaling is not None and abs(attention.scaling - attention.head_dim**-0.5) > 1e-9:
            raise ValueError("the engine's attention scales by head_dim**-0.5")


def _engine_rope(attention) -> RotaryEmbedding:
    rope_config = attention.rope.config
    if rope_config.scaling != "none":
        raise ValueError("true on-policy training on TPU covers unscaled RoPE")
    head_dim = attention.head_dim
    with torch.device("tpu"):  # the engine builds its table on the device, at model load
        if isinstance(attention, Qwen35Attention):
            # the engine's multimodal RoPE; on text positions it rotates the first rotary_dim channels
            return get_rope(
                head_size=head_dim,
                rotary_dim=head_dim,
                max_position=rope_config.max_context_length,
                base=rope_config.theta,
                is_neox_style=True,
                rope_scaling={
                    "rope_type": "default",
                    "mrope_section": rope_config.mrope_section,
                    "mrope_interleaved": True,
                },
                partial_rotary_factor=attention.rotary_dim / head_dim,
                dtype=torch.bfloat16,
            )
        return RotaryEmbedding(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=rope_config.max_context_length,
            base=rope_config.theta,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )


def _attend(attention, rope):
    """The attention sublayer as `(x, segment_ids, positions) -> output`, for this block's layer type."""
    if isinstance(attention, GDN):
        return functools.partial(_gated_delta_net, attention)
    if isinstance(attention, Qwen35Attention):
        return functools.partial(_gated_attention, attention, rope)
    return functools.partial(_attention, attention, rope)


def _eps(norm) -> float:
    if norm.eps is None:
        raise ValueError("true on-policy RMSNorm needs an explicit eps, as the engine's")
    return norm.eps


def _block_forward(
    block, attend, gemma, x, attention_metadata, positions=None, *, padding_mask=None, aux_loss_denominator=None
):
    norm, ffn_norm = block.attention_norm, block.ffn_norm
    dim = norm.weight.shape[0]
    flat = x.reshape(-1, x.shape[-1])
    if flat.shape[-1] == dim:  # the first block: the embedding is both the residual and the norm's input
        residual = flat
        h = ops.rms_norm(flat, None, norm.weight, _eps(norm), gemma=gemma)
    else:
        h, residual = ops.rms_norm(*flat.split(dim, dim=-1), norm.weight, _eps(norm), gemma=gemma)
    attn = attend(h, attention_metadata, positions)
    h, residual = ops.rms_norm(attn, residual, ffn_norm.weight, _eps(ffn_norm), gemma=gemma)
    ffn = _feed_forward(block.feed_forward, h)
    return torch.cat([ffn, residual], dim=-1).reshape(*x.shape[:-1], 2 * dim)


def _attention(attention, rope, x, segment_ids, positions):
    tokens, head_dim = x.shape[0], attention.head_dim
    qkv_linear = attention.qkv_linear
    qkv = ops.matmul(x, qkv_linear.wqkv.weight).view(tokens, -1, qkv_linear.r_dim, head_dim)
    q, k, v = (t.reshape(tokens, -1) for t in torch.split(qkv, [qkv_linear.heads_per_kv, 1, 1], dim=-2))
    q = ops.rms_norm(q.reshape(-1, head_dim), None, attention.q_norm.weight, _eps(attention.q_norm)).view(q.shape)
    k = ops.rms_norm(k.reshape(-1, head_dim), None, attention.k_norm.weight, _eps(attention.k_norm)).view(k.shape)
    q, k = rope.forward_native(positions, q, k)
    out = _PackedAttention.apply(
        q.view(tokens, -1, head_dim), k.view(tokens, -1, head_dim), v.view(tokens, -1, head_dim), segment_ids
    )
    return ops.matmul(out.reshape(tokens, -1), attention.wo.weight)


def _gated_attention(attention, rope, x, segment_ids, positions):
    """Qwen3.5 attention: per head the Q projection is [query | output gate], and RoPE covers rotary_dim channels."""
    tokens, head_dim = x.shape[0], attention.head_dim
    q, gate = ops.matmul(x, attention.wq.weight).view(tokens, -1, 2 * head_dim).chunk(2, dim=-1)
    k = ops.matmul(x, attention.wk.weight)
    v = ops.matmul(x, attention.wv.weight)
    q_norm, k_norm = attention.q_norm, attention.k_norm
    q = ops.rms_norm(q.reshape(-1, head_dim), None, q_norm.weight, _eps(q_norm), gemma=True).view(tokens, -1)
    k = ops.rms_norm(k.reshape(-1, head_dim), None, k_norm.weight, _eps(k_norm), gemma=True).view(tokens, -1)
    q, k = rope.forward_native(positions, q, k)
    out = _PackedAttention.apply(
        q.view(tokens, -1, head_dim), k.view(tokens, -1, head_dim), v.view(tokens, -1, head_dim), segment_ids
    )
    out = out.reshape(tokens, -1) * torch.sigmoid(gate.reshape(tokens, -1))
    return ops.matmul(out, attention.wo.weight)


def _gated_delta_net(gdn, x, segment_ids, positions):
    tokens, key_dim, value_dim = x.shape[0], gdn.key_head_dim, gdn.value_head_dim
    q, k, v, z, a, b = (
        ops.matmul(x, proj.weight)
        for proj in (gdn.in_proj_q, gdn.in_proj_k, gdn.in_proj_v, gdn.in_proj_z, gdn.in_proj_a, gdn.in_proj_b)
    )
    conv = (gdn.conv_q.weight, gdn.conv_k.weight, gdn.conv_v.weight)
    # The engine's op is its per-token recurrence, forward only; the trainer's kernel solves the same math in chunks
    # and gives other bytes, so it only carries the gradient.
    own = gdn.inner_attention(
        q, k, v, a, b, *conv, gdn.A_log, gdn.dt_bias,
        key_head_dim=key_dim, value_head_dim=value_dim, attention_metadata=segment_ids,
    )  # fmt: skip
    engine = ops.packed_gated_delta_net(
        torch.cat([q, k, v], dim=-1).detach(),
        b.detach(),
        a.detach(),
        torch.cat(conv).detach(),
        gdn.A_log.detach().to(torch.bfloat16),
        gdn.dt_bias.detach().to(torch.bfloat16),
        packed_sequences.sequence_starts(segment_ids, MAX_SEQUENCES),
        q.shape[-1] // key_dim,
        v.shape[-1] // value_dim,
        key_dim,
        value_dim,
    )
    core = engine + (own - own.detach()).to(engine.dtype)  # the engine's value, the trainer kernel's gradient
    gated = ops.gated_rms_norm(
        core.reshape(-1, value_dim), z.reshape(-1, value_dim), gdn.norm.weight.to(torch.bfloat16), _eps(gdn.norm)
    )
    return ops.matmul(gated.reshape(tokens, -1), gdn.out_proj.weight)


def _feed_forward(feed_forward, x):
    w13 = feed_forward.w13.weight  # [2, F, D]: gate, then up
    gate, up = ops.matmul(x, w13.reshape(-1, w13.shape[-1])).view(x.shape[0], 2, -1).unbind(-2)
    return ops.matmul(F.silu(gate) * up, feed_forward.w2.weight)


def _final_norm_forward(norm, gemma, x):
    dim = norm.weight.shape[0]
    flat = x.reshape(-1, x.shape[-1])
    out, _ = ops.rms_norm(*flat.split(dim, dim=-1), norm.weight, _eps(norm), gemma=gemma)
    return out.reshape(*x.shape[:-1], dim)


def _lm_head_forward(lm_head, x):
    out = ops.matmul(x.reshape(-1, x.shape[-1]), lm_head.weight)
    return out.reshape(*x.shape[:-1], out.shape[-1])


class _PackedAttention(torch.autograd.Function):
    """Forward: the engine's attention. Backward: splash attention's gradient at the same inputs."""

    @staticmethod
    def forward(ctx, q, k, v, segment_ids):
        ctx.save_for_backward(q, k, v, segment_ids)
        starts = packed_sequences.sequence_starts(segment_ids, MAX_SEQUENCES)
        return ops.packed_causal_attention(q, k, v, starts, q.shape[-1] ** -0.5)  # _check_supported pins this scale

    @staticmethod
    def backward(ctx, grad):
        q, k, v, segment_ids = ctx.saved_tensors
        # splash takes [batch, heads, tokens, dim] and recomputes its own output and logsumexp
        q4, k4, v4, g4 = (t.transpose(0, 1).unsqueeze(0).contiguous() for t in (q, k, v, grad))
        seg = segment_ids.unsqueeze(0)
        out, lse = _splash_forward(q4, k4, v4, seg)
        dq, dk, dv = _splash_backward(q4, k4, v4, out, lse, g4, seg)
        return (*(t.squeeze(0).transpose(0, 1) for t in (dq, dk, dv)), None)
