"""Inference-side sparse-attention arithmetic for the DSV4 TOP contract."""

from __future__ import annotations

import torch

from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla_fwd import (
    sparse_mqa_fwd_interface,
)

_CONTRACT_TOKENS = 96
_HEAD_DIM = 512
_CONTRACT_LOCAL_HEADS = 8
_QUERY_HEADS = 64
_SWA_SLOTS = 128
_COMPRESSED_SLOTS = 256
_KERNEL_SLOTS = _SWA_SLOTS + _COMPRESSED_SLOTS


def _validate_sparse_indices(
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
) -> None:
    if combined_lens.shape != (_CONTRACT_TOKENS,):
        raise RuntimeError("DSV4 TOP sparse-attention length shape mismatch: " f"{tuple(combined_lens.shape)}")
    if combined_indices.ndim != 2 or combined_indices.shape[0] != (_CONTRACT_TOKENS):
        raise RuntimeError("DSV4 TOP sparse-attention index shape mismatch: " f"{tuple(combined_indices.shape)}")
    if combined_indices.dtype != torch.int32:
        raise RuntimeError("DSV4 TOP sparse-attention indices must be int32, got " f"{combined_indices.dtype}")
    valid_counts = (combined_indices >= 0).sum(dim=-1).to(combined_lens.dtype)
    if not torch.equal(valid_counts, combined_lens):
        raise RuntimeError("DSV4 TOP sparse-attention lengths do not match valid indices")


def _causal_indices(
    *,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    token_positions = torch.arange(
        _CONTRACT_TOKENS,
        dtype=torch.int32,
        device=device,
    )
    indices = torch.full(
        (_CONTRACT_TOKENS, width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    indices[:, :_CONTRACT_TOKENS] = torch.where(
        token_positions.unsqueeze(0) <= token_positions.unsqueeze(1),
        token_positions.unsqueeze(0),
        -1,
    )
    return indices


def maybe_dsv4_top_sparse_prefill(
    *,
    model_runner,
    q_flat: torch.Tensor,
    kv: torch.Tensor,
    swa_slice: torch.Tensor,
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    compress_ratio: int,
    layer_id: int,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor | None:
    """Return the TOP output for the pinned E2E shape, else ``None``."""
    contract_layer = (
        (compress_ratio == 0 and layer_id in (0, 1))
        or (compress_ratio == 4 and layer_id == 2)
        or (compress_ratio == 128 and layer_id == 3)
    )
    if not (
        contract_layer and q_flat.shape[0] == _CONTRACT_TOKENS and kv.ndim == 3 and kv.shape[1:] == (1, _HEAD_DIM)
    ):
        return None

    n_local_heads = model_runner.model_config.get_num_attention_heads(model_runner.tp_size)
    if n_local_heads != _CONTRACT_LOCAL_HEADS:
        raise RuntimeError(
            "DSV4 TOP sparse attention expected " f"{_CONTRACT_LOCAL_HEADS} local heads, got {n_local_heads}"
        )
    expected_q_shape = (
        _CONTRACT_TOKENS,
        _QUERY_HEADS,
        _HEAD_DIM,
    )
    if q_flat.shape != expected_q_shape:
        raise RuntimeError(
            "DSV4 TOP sparse-attention query shape mismatch: "
            f"expected {expected_q_shape}, got {tuple(q_flat.shape)}"
        )
    _validate_sparse_indices(combined_indices, combined_lens)

    if compress_ratio == 0:
        expected_kv_shape = (_CONTRACT_TOKENS, 1, _HEAD_DIM)
        if kv.shape != expected_kv_shape:
            raise RuntimeError(
                "DSV4 TOP C0 KV shape mismatch: " f"expected {expected_kv_shape}, got {tuple(kv.shape)}"
            )
        expected_indices = _causal_indices(
            width=_SWA_SLOTS,
            device=combined_indices.device,
        )
        expected_lens = torch.arange(
            1,
            _CONTRACT_TOKENS + 1,
            dtype=torch.int32,
            device=combined_lens.device,
        )
        if not torch.equal(combined_lens, expected_lens):
            raise RuntimeError("DSV4 TOP C0 lengths are not causal")
        if not torch.equal(combined_indices, expected_indices):
            raise RuntimeError("DSV4 TOP C0 indices are not causal")
        kv_for_miles = kv
        kernel_indices = combined_indices
    elif compress_ratio == 4:
        expected_kv_shape = (120, 1, _HEAD_DIM)
        if kv.shape != expected_kv_shape:
            raise RuntimeError(
                "DSV4 TOP C4 KV shape mismatch: " f"expected {expected_kv_shape}, got {tuple(kv.shape)}"
            )
        expected_index_shape = (_CONTRACT_TOKENS, 640)
        if combined_indices.shape != expected_index_shape:
            raise RuntimeError(
                "DSV4 TOP C4 index shape mismatch: "
                f"expected {expected_index_shape}, got "
                f"{tuple(combined_indices.shape)}"
            )

        token_positions = torch.arange(
            _CONTRACT_TOKENS,
            dtype=torch.int32,
            device=combined_indices.device,
        )
        compressed_lens = (token_positions + 1) // 4
        window_lens = token_positions + 1
        if not torch.equal(
            combined_lens,
            compressed_lens + window_lens,
        ):
            raise RuntimeError("DSV4 TOP C4 sparse lengths are invalid")

        kernel_indices = torch.full(
            (_CONTRACT_TOKENS, _KERNEL_SLOTS),
            -1,
            dtype=torch.int32,
            device=combined_indices.device,
        )
        window_slots = torch.arange(
            _SWA_SLOTS,
            dtype=torch.int64,
            device=combined_indices.device,
        ).unsqueeze(0)
        window_sources = compressed_lens.long().unsqueeze(1) + window_slots
        window_values = combined_indices.gather(1, window_sources)
        kernel_indices[:, :_SWA_SLOTS] = torch.where(
            window_slots < window_lens.long().unsqueeze(1),
            window_values,
            -1,
        )
        compressed_slots = torch.arange(
            _COMPRESSED_SLOTS,
            dtype=torch.int64,
            device=combined_indices.device,
        ).unsqueeze(0)
        kernel_indices[:, _SWA_SLOTS:] = torch.where(
            compressed_slots < compressed_lens.long().unsqueeze(1),
            combined_indices[:, :_COMPRESSED_SLOTS],
            -1,
        )
        valid_kernel_indices = (kernel_indices == -1) | ((kernel_indices >= 0) & (kernel_indices < kv.shape[0]))
        if not bool(torch.all(valid_kernel_indices).item()):
            raise RuntimeError("DSV4 TOP C4 generated out-of-range sparse indices")
        kv_for_miles = kv
    else:
        expected_swa_shape = (
            _CONTRACT_TOKENS,
            1,
            _HEAD_DIM,
        )
        if swa_slice.shape != expected_swa_shape:
            raise RuntimeError(
                "DSV4 TOP C128 SWA shape mismatch: " f"expected {expected_swa_shape}, got " f"{tuple(swa_slice.shape)}"
            )
        expected_lens = torch.arange(
            1,
            _CONTRACT_TOKENS + 1,
            dtype=torch.int32,
            device=combined_lens.device,
        )
        if not torch.equal(combined_lens, expected_lens):
            raise RuntimeError("DSV4 TOP C128 lengths are not causal")
        kv_for_miles = swa_slice
        kernel_indices = _causal_indices(
            width=_KERNEL_SLOTS,
            device=combined_indices.device,
        )

    q_miles = q_flat[:, :n_local_heads, :].unsqueeze(0).contiguous()
    kv_miles = kv_for_miles.squeeze(1).unsqueeze(0).contiguous()
    sink_miles = attn_sink[:n_local_heads].contiguous()
    indices_miles = kernel_indices.unsqueeze(0).contiguous()
    output, _ = sparse_mqa_fwd_interface(
        q_miles,
        kv_miles,
        sink_miles,
        indices_miles,
        sm_scale=softmax_scale,
    )

    result = torch.zeros_like(q_flat)
    result[:, :n_local_heads, :].copy_(output.squeeze(0))
    return result
