"""Qwen3.5's gated delta-net on packed documents in plain torch, for platforms without attn_gym's CUDA kernels.

Documents are told apart by `document_ids`, one id per token, which keeps the static shape torch_tpu compiles;
the CUDA kernels take `cu_seqlens` instead, whose length varies.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch._higher_order_ops.scan import scan
from torchtitan.models.common.attention import InnerAttention

from miles.backends.torchtitan_utils.sdpa_document_attention import document_ids


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


def document_gated_delta(
    q_THK: torch.Tensor,
    k_THK: torch.Tensor,
    v_THV: torch.Tensor,
    g_TH: torch.Tensor,
    beta_TH: torch.Tensor,
    doc_ids_T: torch.Tensor,
    *,
    chunk_size: int = 32,
) -> torch.Tensor:
    """attn_gym's `chunk_gdn` (gated delta rule on l2-normalized q, k; scale K^-0.5) with the state reset per document.

    The chunked form of the recurrence, in fp32. Within a chunk, decays exp(G_i - G_j) are only taken for j <= i
    of one document, so they never exceed 1; the state crosses chunks in a scan. 32-token chunks had the lowest
    HBM peak on v7x (the forward substitution saves O(chunk^3) per head and chunk).
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

    # (I + A)^-1 by forward substitution, A[i, j] = beta_i <k_i, k_j> exp(G_i - G_j) for j < i in one document
    neg_a = -(k @ k.mT) * decay.tril(-1) * beta[..., None]
    rows = [neg_a[..., :1, :]]
    for i in range(1, chunk_size):
        row = neg_a[..., i : i + 1, :]
        rows.append(row + row[..., :i] @ torch.cat(rows, dim=-2))
    transform = torch.cat(rows, dim=-2) + torch.eye(chunk_size, device=q.device)
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

    def chunk_step(state, chunk):
        u_n, w_n, q_n, a_qk_n, k_to_last_n, decay_n = chunk
        v_new = u_n - w_n @ state
        return state * decay_n[:, None, None] + k_to_last_n @ v_new, q_n @ state + a_qk_n @ v_new

    # one loop in the compiled program; an unrolled Python loop gives each chunk a full-size gradient buffer,
    # so its backward memory grows with T^2
    _, out_HNCV = scan(
        chunk_step,
        q.new_zeros(num_heads, key_dim, v.shape[-1]),
        (u, w_from_state, q_from_state, a_qk, k_to_last_T, state_decay),
        dim=1,
    )
    return out_HNCV.flatten(1, 2).movedim(0, 1)[:seq_len].to(v_THV.dtype)


class DocumentGatedDeltaInnerAttention(InnerAttention):
    """torchtitan's `InnerGDN` (short convs, decay gate, delta rule) on document ids, without attn_gym."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> torch.Tensor:
            del padding_mask, max_num_documents, max_context_length
            return document_ids(positions)

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        a_TH: torch.Tensor,
        b_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_H: torch.Tensor,
        *,
        key_head_dim: int,
        value_head_dim: int,
        attention_metadata: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(attention_metadata, torch.Tensor):
            raise TypeError(f"expected document ids as attention_metadata, got {type(attention_metadata).__name__}")
        num_tokens = query_TC.shape[0]

        def conv_heads(x_TC: torch.Tensor, weight_C1W: torch.Tensor, head_dim: int) -> torch.Tensor:
            return document_short_conv(x_TC, weight_C1W, attention_metadata).reshape(num_tokens, -1, head_dim)

        g_TH = -torch.exp(A_log_H.float()) * F.softplus(a_TH.float() + dt_bias_H)
        return document_gated_delta(
            conv_heads(query_TC, conv_q_weight_C1W, key_head_dim),
            conv_heads(key_TC, conv_k_weight_C1W, key_head_dim),
            conv_heads(value_TC, conv_v_weight_C1W, value_head_dim),
            g_TH,
            torch.sigmoid(b_TH),
            attention_metadata,
        )
