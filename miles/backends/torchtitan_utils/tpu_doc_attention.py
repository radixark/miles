"""Packed-document causal attention for torchtitan on TPU, chosen by --titan-tpu-attention.

torch_tpu runs FlexAttention as one dense L x L mask (torch_tpu/_internal/compile/flex_attention_redirect.py).
`sdpa_blocks` runs query blocks over native SDPA, each seeing only the keys at or before the block's end;
`splash` runs the Pallas splash-attention kernel with documents as segment ids (tpu_splash_attention.py).
"""

import torch
import torch.nn.functional as F
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.decoder import Decoder

_Q_BLOCK = 1024  # measured on v7x: ties 2048 at L=8192, 512 is ~13% slower


def document_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    doc_ids: torch.Tensor,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """q: (B, Nq, L, H); k, v: (B, Nkv, L, H); doc_ids: (B, L). Skips tiles above the diagonal."""
    seq_len = q.shape[2]
    blocks = []
    for start in range(0, seq_len, _Q_BLOCK):
        end = min(start + _Q_BLOCK, seq_len)
        q_pos = torch.arange(start, end, device=q.device)
        kv_pos = torch.arange(end, device=q.device)
        same_doc = doc_ids[:, start:end, None] == doc_ids[:, None, :end]
        mask = (same_doc & (kv_pos[None, :] <= q_pos[:, None])).unsqueeze(1)
        blocks.append(
            F.scaled_dot_product_attention(
                q[:, :, start:end],
                k[:, :, :end],
                v[:, :, :end],
                attn_mask=mask,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        )
    return torch.cat(blocks, dim=2)


def document_ids(positions: torch.Tensor) -> torch.Tensor:
    """Each token's document, as the index of the document's first token: `arange(L) - positions`.

    Every packed document and the padding tail restart positions at 0 and count up by 1, so this names the
    same documents as `(positions == 0).cumsum(-1)` without a cumulative op: libtpu's associative-scan
    emitter (>= 0.0.44) miscompiles cumsum for lengths that are not a multiple of 1024. google-pytorch/torchtitan's
    `segment_ids_from_positions` (torchtitan/models/common/attention.py) is the cumsum form.
    The ids are non-decreasing, which the splash kernel's block skipping relies on.
    """
    arange = torch.arange(positions.shape[-1], device=positions.device, dtype=torch.int32)
    return arange - positions.to(torch.int32)


def install(backend: str) -> None:
    """Route torchtitan's document-masked FlexAttention to the `backend` implementation."""
    if backend == "splash":
        from miles.backends.torchtitan_utils.tpu_splash_attention import splash_document_attention

        def attention(q, k, v, doc_ids, *, scale=None, enable_gqa=False):
            return splash_document_attention(q, k, v, doc_ids, scale=scale)

    else:
        attention = document_causal_attention

    def create_document_ids(self, positions, attn_config):
        return document_ids(positions)

    def forward(
        self,
        q_BLNH,
        k_BLNH,
        v_BLNH,
        *,
        attention_masks,
        score_mod=None,
        scale=None,
        enable_gqa=False,
        out_transform=None,
    ):
        assert score_mod is None and out_transform is None
        out_BNLH = attention(
            q_BLNH.transpose(1, 2),
            k_BLNH.transpose(1, 2),
            v_BLNH.transpose(1, 2),
            attention_masks,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        return out_BNLH.transpose(1, 2).contiguous()

    Decoder._create_flex_attention_mask_for_document = create_document_ids
    FlexAttention.forward = forward
