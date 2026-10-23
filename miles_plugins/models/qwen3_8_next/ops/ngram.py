"""Qwen3.8-Next PLE n-gram hashing and checkpoint reads, free of any training backend.

The Megatron PLE layer and the torchtitan one both hash token ids here, so the two
backends and sglang's rollout agree on which table rows a token reads.
"""

import json
import os
import struct
from dataclasses import dataclass

import torch
from torch import Tensor

from miles.utils.hf_utils.weight_mapping import get_checkpoint_weight_map


def shift_right_ignore_eos(tokens: Tensor, n: int, eos_token_id: int) -> Tensor:
    """Shift right by ``n`` without letting context cross an EOS boundary."""
    if n == 0:
        return tokens
    batch_size, seq_len = tokens.shape
    idx = torch.arange(seq_len, device=tokens.device, dtype=torch.long)

    eos_pos = torch.where(tokens == eos_token_id, idx, torch.full_like(idx, -1))
    prev_eos_inclusive = torch.cummax(eos_pos, dim=1).values
    prev_eos = torch.cat([eos_pos.new_full((batch_size, 1), -1), prev_eos_inclusive[:, :-1]], dim=1)
    pos_in_segment = idx.unsqueeze(0) - (prev_eos + 1)

    src_idx = idx - n
    gathered = tokens.gather(dim=1, index=torch.clamp(src_idx, min=0).unsqueeze(0).expand(batch_size, -1))
    valid = (pos_in_segment >= n) & (src_idx.unsqueeze(0) >= 0)
    return torch.where(valid, gathered, tokens.new_full((), eos_token_id))


def ngram_hash_ids(
    contexts: Tensor,
    layer_multipliers: Tensor,
    head_vocab_sizes: Tensor,
    head_offsets: Tensor,
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
) -> Tensor:
    """Row ids into the flat PLE table, one per hash head."""
    shifted = [contexts]
    for shift in range(1, ngram_size):
        shifted.append(shift_right_ignore_eos(contexts, shift, eos_token_id))

    blocks = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        end = start + heads_per_ngram
        mix = shifted[0] * layer_multipliers[0]
        for pos in range(1, ngram):
            mix = torch.bitwise_xor(mix, shifted[pos] * layer_multipliers[pos])
        ids = torch.remainder(mix[:, -1:].unsqueeze(-1), head_vocab_sizes[start:end].view(1, 1, -1))
        blocks.append((ids + head_offsets[start:end].view(1, 1, -1))[:, 0])
    return torch.cat(blocks, dim=-1)


def build_ngram_contexts(tokens: Tensor, ngram_size: int, eos_token_id: int) -> Tensor:
    """``[T]`` token ids -> ``[T, ngram_size]`` sliding windows, one row per token."""
    assert tokens.dim() == 1, f"expected a 1-D token sequence, got {tuple(tokens.shape)}"
    pad = tokens.new_full((ngram_size - 1,), eos_token_id)
    return torch.cat([pad, tokens]).unfold(0, ngram_size, 1)


def build_ngram_contexts_packed(
    tokens: Tensor, cu_seqlens: Tensor | None, ngram_size: int, eos_token_id: int
) -> Tensor:
    """``build_ngram_contexts`` for a packed (THD) batch."""
    if cu_seqlens is None:
        return build_ngram_contexts(tokens, ngram_size, eos_token_id)
    bounds = cu_seqlens.tolist()
    parts = [
        build_ngram_contexts(tokens[lo:hi], ngram_size, eos_token_id)
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
        if hi > lo
    ]
    return torch.cat(parts, dim=0)


def safetensors_slice(path: str, name: str, header_cache: dict) -> tuple[int, int, list[int]]:
    """Byte range and shape of one tensor, from the safetensors header."""
    if path not in header_cache:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header_cache[path] = (json.loads(f.read(n)), 8 + n)
    header, base = header_cache[path]
    meta = header[name]
    start, end = meta["data_offsets"]
    return base + start, base + end, meta["shape"]


_PLE_HASH_TENSORS = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")


@dataclass(frozen=True)
class PLEShard:
    """Where one ``ngram_embedding.shard_<i>`` lives in the checkpoint."""

    path: str
    start: int
    end: int
    shape: tuple[int, ...]


def _ple_prefix(hf_layer_index: int) -> str:
    return f"model.language_model.layers.{hf_layer_index}.ple.ple_embedding"


def ple_table_shards(hf_checkpoint: str, hf_layer_index: int) -> list[PLEShard]:
    """Every n-gram table shard of a PLE layer, in shard order; the flat table is their row concatenation."""
    index = get_checkpoint_weight_map(hf_checkpoint)
    prefix = f"{_ple_prefix(hf_layer_index)}.ngram_embedding.shard_"
    shard_ids = sorted(int(k[len(prefix) :].split(".")[0]) for k in index if k.startswith(prefix))
    if not shard_ids or shard_ids != list(range(len(shard_ids))):
        raise ValueError(f"{hf_checkpoint} has PLE shards {shard_ids[:5]}... for layer {hf_layer_index}, not 0..N-1")
    header_cache: dict = {}
    shards = []
    for shard_id in shard_ids:
        key = f"{prefix}{shard_id}.weight"
        path = os.path.join(hf_checkpoint, index[key])
        start, end, shape = safetensors_slice(path, key, header_cache)
        shards.append(PLEShard(path=path, start=start, end=end, shape=tuple(shape)))
    return shards


def read_ple_shard_into(shard: PLEShard, dst: Tensor) -> None:
    """Read one shard's bytes straight into ``dst`` (a CPU tensor of the shard's shape)."""
    if tuple(dst.shape) != shard.shape:
        raise ValueError(f"{shard.path}: destination {tuple(dst.shape)} does not match shard {shard.shape}")
    view = memoryview(dst.view(torch.uint8).reshape(-1).numpy())
    with open(shard.path, "rb") as f:
        f.seek(shard.start)
        read = f.readinto(view)
    if read != shard.end - shard.start:
        raise OSError(f"{shard.path} ended after {read} of {shard.end - shard.start} bytes; is the download complete?")


def read_ple_hash_tensors(hf_checkpoint: str, hf_layer_index: int) -> dict[str, Tensor]:
    """The three int64 tensors that parameterise the n-gram hash."""
    index = get_checkpoint_weight_map(hf_checkpoint)
    header_cache: dict = {}
    tensors = {}
    for name in _PLE_HASH_TENSORS:
        key = f"{_ple_prefix(hf_layer_index)}.{name}"
        path = os.path.join(hf_checkpoint, index[key])
        start, end, shape = safetensors_slice(path, key, header_cache)
        with open(path, "rb") as f:
            f.seek(start)
            raw = f.read(end - start)
        tensors[name] = torch.frombuffer(bytearray(raw), dtype=torch.int64).reshape(shape).clone()
    return tensors
