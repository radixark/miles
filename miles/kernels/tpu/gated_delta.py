"""Qwen3.5's gated delta-net on packed documents in plain torch, for platforms without attn_gym's CUDA kernels.

Documents are told apart by `document_ids`, one id per token, which keeps the static shape torch_tpu compiles;
the CUDA kernels take `cu_seqlens` instead, whose length varies.
"""

import torch
import torch.nn.functional as F
from torch._higher_order_ops.scan import scan, scan_op
from torch._higher_order_ops.utils import redirect_to_mode
from torch.utils.checkpoint import _CachedTorchDispatchMode, _CachingTorchDispatchMode

# torch registers these selective-checkpoint rules for flex_attention and invoke_subgraph but not for scan, so a
# remat region around this layer fails on the scan; with them the checkpoint policy saves or recomputes it whole
for _mode in (_CachingTorchDispatchMode, _CachedTorchDispatchMode):
    if _mode not in scan_op.python_key_table:
        redirect_to_mode(scan_op, _mode)


def document_short_conv(x_TC: torch.Tensor, weight_C1W: torch.Tensor, doc_ids_T: torch.Tensor) -> torch.Tensor:
    """attn_gym's varlen `causal_conv1d` with silu: a depthwise causal conv that stops at each document start.

    Shift-and-multiply rather than F.conv1d, whose compiled backward XLA rejects (b/574632209).
    """
    seq_len, kernel_size = x_TC.shape[0], weight_C1W.shape[-1]
    weight_CW = weight_C1W[:, 0].float()
    out_TC = x_TC.float() * weight_CW[:, -1]
    for shift in range(1, kernel_size):
        shifted_TC = F.pad(x_TC.float(), (0, 0, shift, 0))[:seq_len]
        same_doc_T = doc_ids_T == F.pad(doc_ids_T, (shift, 0), value=-1)[:seq_len]
        out_TC = out_TC + shifted_TC * same_doc_T[:, None] * weight_CW[:, -1 - shift]
    return F.silu(out_TC).to(x_TC.dtype)


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    # attn_gym's rsqrt form, not F.normalize's clamp
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + 1e-6)


_INVERSE_BLOCK = 16


def _substitute(n: torch.Tensor) -> torch.Tensor:
    """inv(I - n) for a strictly lower n [..., b, b], by forward substitution."""
    size = n.shape[-1]
    rows = [n[..., :1, :]]
    for i in range(1, size):
        row = n[..., i : i + 1, :]
        rows.append(row + row[..., :i] @ torch.cat(rows, dim=-2))
    return torch.cat(rows, dim=-2) + torch.eye(size, device=n.device, dtype=n.dtype)


def _diagonal_blocks(x: torch.Tensor, size: int) -> torch.Tensor:
    """The [size, size] blocks on the diagonal of x [..., C, C], as [..., C / size, size, size]."""
    blocks = x.unflatten(-2, (-1, size)).unflatten(-1, (-1, size))  # [..., m, size, m, size]
    return torch.diagonal(blocks, dim1=-4, dim2=-2).movedim(-1, -3)


class _UnitLowerInverse(torch.autograd.Function):
    """inv(I - N) for a strictly lower N [..., C, C], saving only the result.

    Substitution on 16-row diagonal blocks, then doubling: for a 2x2 block split, the lower-left of the inverse is
    X22 N21 X11. Autograd through a row-by-row substitution keeps every partial row (O(C^3) per chunk) and runs C
    dependent steps; here backward is one product, dN = tril(X^T dX X^T, -1).
    """

    @staticmethod
    def forward(ctx, n: torch.Tensor) -> torch.Tensor:
        size = min(_INVERSE_BLOCK, n.shape[-1])
        x = _substitute(_diagonal_blocks(n, size))  # [..., C / size, size, size]
        while size < n.shape[-1]:
            x11, x22 = x[..., 0::2, :, :], x[..., 1::2, :, :]
            x21 = x22 @ _diagonal_blocks(n, 2 * size)[..., size:, :size] @ x11
            x = torch.cat([torch.cat([x11, torch.zeros_like(x11)], dim=-1), torch.cat([x21, x22], dim=-1)], dim=-2)
            size *= 2
        x = x.squeeze(-3)
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, dx: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return (x.mT @ dx @ x.mT).tril(-1)


def document_gated_delta(
    q_THK: torch.Tensor,
    k_THK: torch.Tensor,
    v_THV: torch.Tensor,
    g_TH: torch.Tensor,
    beta_TH: torch.Tensor,
    doc_ids_T: torch.Tensor,
    *,
    chunk_size: int = 128,
) -> torch.Tensor:
    """attn_gym's `chunk_gdn` (gated delta rule on l2-normalized q, k; scale K^-0.5) with the state reset per document.

    The chunked form of the recurrence, in fp32. Within a chunk, decays exp(G_i - G_j) are only taken for j <= i
    of one document, so they never exceed 1; the state crosses chunks in a scan. On one v7x core at Qwen3.5-4B
    shapes (T=8192), 128-token chunks with the blocked inverse ran fwd+bwd 3.8x faster than row substitution,
    at 35% less HBM.
    """
    seq_len, num_heads, key_dim = v_THV.shape[0], v_THV.shape[1], q_THK.shape[-1]
    # each query/key head serves a run of consecutive value heads
    q_THK = q_THK.repeat_interleave(num_heads // q_THK.shape[1], dim=1)
    k_THK = k_THK.repeat_interleave(num_heads // k_THK.shape[1], dim=1)
    pad = (-seq_len) % chunk_size

    def chunked(x: torch.Tensor) -> torch.Tensor:
        # [T, H, ...] -> [H, N, C, ...] in fp32
        x = F.pad(x.float(), (0, 0) * (x.dim() - 1) + (0, pad))
        return x.unflatten(0, (-1, chunk_size)).movedim(2, 0)

    q = chunked(_l2norm(q_THK.float())) * key_dim**-0.5
    k = chunked(_l2norm(k_THK.float()))
    v = chunked(v_THV)
    beta = chunked(beta_TH)
    g = chunked(g_TH).cumsum(-1)
    doc = F.pad(doc_ids_T, (0, pad), value=-1).unflatten(0, (-1, chunk_size))  # [N, C]
    prev_doc = F.pad(doc[:-1, -1], (1, 0), value=-2)  # [N], the document of the token before each chunk

    causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).tril()
    in_doc_causal = (doc[:, :, None] == doc[:, None, :]) & causal
    decay = torch.where(in_doc_causal, g[..., :, None] - g[..., None, :], -torch.inf).exp()

    # (I + A)^-1, A[i, j] = beta_i <k_i, k_j> exp(G_i - G_j) for j < i in one document
    transform = _UnitLowerInverse.apply(-(k @ k.mT) * decay.tril(-1) * beta[..., None])
    u = transform @ (v * beta[..., None])
    w = transform @ (k * (beta * g.exp())[..., None])
    a_qk = (q @ k.mT) * decay

    # tokens continuing the carried state's document read it; the chunk's last document writes it
    reads_state = (doc == prev_doc[:, None]).float()[..., None]  # [N, C, 1]
    writes_state = (doc == doc[:, -1:]).float()[..., None]
    state_survives = (doc[:, -1] == prev_doc).float()  # [N]
    w_from_state = w * reads_state
    q_from_state = q * g.exp()[..., None] * reads_state
    k_to_last_T = (k * (g[..., -1:] - g).exp()[..., None] * writes_state).mT
    state_decay = g[..., -1].exp() * state_survives  # [H, N]

    out_HNCV = _ChunkRecurrence.apply(u, w_from_state, q_from_state, a_qk, k_to_last_T, state_decay)
    return out_HNCV.flatten(1, 2).movedim(0, 1)[:seq_len].to(v_THV.dtype)


class _ChunkRecurrence(torch.autograd.Function):
    """The state carried across chunks, as one loop each way.

    Per chunk n with incoming state S: v = u - w S, out = q S + a v, and the next state is d S + k v.
    The forward is a `scan`, avoiding an unrolled autograd graph whose saved buffers grow with the chunk count.
    The explicit static-loop backward keeps AOT from differentiating a scan, which compiled activation checkpointing
    cannot partition, and avoids incorrect gradients from TorchTPU's compiled reverse scan. Saves the incoming
    state of every chunk, [H, N, K, V] in fp32.
    """

    @staticmethod
    def forward(ctx, u, w, q, a_qk, k_to_last_T, decay):
        def step(state, chunk):
            u_n, w_n, q_n, a_qk_n, k_to_last_n, decay_n = chunk
            v_new = u_n - w_n @ state
            next_state = state * decay_n[:, None, None] + k_to_last_n @ v_new
            return next_state, (q_n @ state + a_qk_n @ v_new, state.clone())  # scan outputs may not alias the carry

        initial = q.new_zeros(q.shape[0], q.shape[-1], u.shape[-1])
        _, (out, states) = scan(step, initial, (u, w, q, a_qk, k_to_last_T, decay), dim=1)
        ctx.save_for_backward(u, w, q, a_qk, k_to_last_T, decay, states)
        return out

    @staticmethod
    def backward(ctx, d_out):
        def step(d_next_state, chunk):
            u_n, w_n, q_n, a_qk_n, k_to_last_n, decay_n, state, d_out_n = chunk
            v_new = u_n - w_n @ state
            d_v = a_qk_n.mT @ d_out_n + k_to_last_n.mT @ d_next_state
            d_state = d_next_state * decay_n[:, None, None] + q_n.mT @ d_out_n - w_n.mT @ d_v
            grads = (
                d_v,
                -d_v @ state.mT,
                d_out_n @ state.mT,
                d_out_n @ v_new.mT,
                d_next_state @ v_new.mT,
                (d_next_state * state).sum((-2, -1)),
            )
            return d_state, grads

        saved = ctx.saved_tensors  # unpack once: activation checkpointing refuses a second unpack
        states = saved[-1]
        d_state = d_out.new_zeros(states.shape[0], *states.shape[2:])
        grads = [[] for _ in range(6)]
        for index in range(states.shape[1] - 1, -1, -1):
            d_state, step_grads = step(d_state, tuple(x[:, index] for x in (*saved, d_out)))
            for parts, grad in zip(grads, step_grads, strict=True):
                parts.append(grad)
        return tuple(torch.stack(list(reversed(parts)), dim=1) for parts in grads)
