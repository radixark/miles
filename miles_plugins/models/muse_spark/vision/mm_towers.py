"""Wire the frozen Muse Spark vision tower into ``MuseSpark1xGPTModel`` (Inkling ``mm_towers`` pattern).

``pre_process`` ranks build the tower from the HF checkpoint and merge image
embeddings into ``decoder_input``; other pipeline stages swallow the multimodal
kwargs. The tower lives in ``model.__dict__`` (not a registered submodule), so
it is invisible to the optimizer, checkpoints and ``named_parameters``.

Merge order mirrors SGLang's omni path: text rows = ``embedding_norm(embedding)``,
image rows = tower output (already RMS-normed), then everything is multiplied by
``embedding_multiplier``. ``MuseSpark1xGPTModel._preprocess`` skips both steps when it
receives ``decoder_input``, so they are done here.

Image rows are addressed by ``mm_vision_positions`` in the natural packed row
(``data.py`` shifts sample-local positions by the CP-padded sample offsets); this module maps
them to this rank's zigzag CP chunk and sequence-parallel window.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import torch

from miles_plugins.models.muse_spark.vision.navit import MuseSparkVisionConfig, MuseSparkVisionTower
from miles_plugins.models.muse_spark.weight_mapping import rope_pairs_hf_to_kernel

MM_KWARGS = ("mm_vision_patches", "mm_vision_grids", "mm_vision_positions")
_ROTARY_ROWS = re.compile(r"\.attn\.(q|k)_proj\.")
logger = logging.getLogger(__name__)
_MERGE_CALLS = {"n": 0}


def load_tower_state(hf_checkpoint: str | Path, tower: MuseSparkVisionTower) -> dict[str, torch.Tensor]:
    """Read ``tower``'s parameters out of a Muse Spark checkpoint, q/k rows re-interleaved into rotary pairs."""
    from safetensors import safe_open

    root = Path(hf_checkpoint)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(root / "model.safetensors", framework="pt", device="cpu") as reader:
            weight_map = dict.fromkeys(reader.keys(), "model.safetensors")
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for key in tower.state_dict():
        name = tower.checkpoint_name(key)
        by_shard.setdefault(weight_map[name], []).append((name, key))
    state: dict[str, torch.Tensor] = {}
    for shard, pairs in sorted(by_shard.items()):
        with safe_open(root / shard, framework="pt", device="cpu") as reader:
            for name, key in pairs:
                tensor = reader.get_tensor(name)
                if _ROTARY_ROWS.search(key):
                    tensor = rope_pairs_hf_to_kernel(tensor, tower.config.head_dim)
                state[key] = tensor
    return state


def build_vision_tower(
    hf_checkpoint: str | Path, device, dtype, *, attn_impl: str = "sdpa", train: bool = False
) -> MuseSparkVisionTower:
    hf_config = json.loads((Path(hf_checkpoint) / "config.json").read_text())
    tower = MuseSparkVisionTower(MuseSparkVisionConfig.from_hf(hf_config), attn_impl=attn_impl)
    state = {key: value.to(dtype) for key, value in load_tower_state(hf_checkpoint, tower).items()}
    tower.load_state_dict(state, strict=True)
    tower.to(device=device, dtype=dtype)
    tower.requires_grad_(train)
    if not train:
        tower.eval()
    return tower


def natural_to_local_rows(
    positions: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    cp_size: int,
    cp_rank: int,
    sp_size: int = 1,
    sp_rank: int = 0,
    local_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map natural packed row indices to this rank's rows.

    ``cu_seqlens`` are the natural (CP-padded) per-sample boundaries, i.e. Miles'
    ``cu_seqlens_q``; rank ``cp_rank`` holds chunks ``[cp_rank, 2*cp_size-1-cp_rank]`` of every
    sample (``cp_utils.slice_with_cp``). A sequence-parallel window then keeps rows in
    ``[sp_rank*local_len, (sp_rank+1)*local_len)``. Returns ``(local_rows, keep_mask)``.
    """
    positions = positions.to(torch.long)
    if cp_size > 1:
        if cu_seqlens is None:
            raise ValueError("context parallelism needs packed cu_seqlens to place image rows")
        cu = cu_seqlens.to(positions.device, torch.long)
        sample = torch.bucketize(positions, cu[1:], right=True)  # sample index per position
        start = cu[sample]
        length = cu[sample + 1] - start
        chunk = length // (2 * cp_size)
        within = positions - start
        chunk_id = within // chunk
        first, second = cp_rank, 2 * cp_size - 1 - cp_rank
        keep = (chunk_id == first) | (chunk_id == second)
        local_sample_start = start // cp_size
        local = local_sample_start + torch.where(chunk_id == first, within % chunk, chunk + within % chunk)
    else:
        keep = torch.ones_like(positions, dtype=torch.bool)
        local = positions.clone()
    if sp_size > 1:
        if local_len is None:
            raise ValueError("sequence parallelism needs the local sequence length")
        local = local - sp_rank * local_len
        keep = keep & (local >= 0) & (local < local_len)
    return local, keep


def wire_mm_towers(model, hf_checkpoint: str, *, train: bool = False, attn_impl: str = "sdpa") -> None:
    """Wrap a ``MuseSpark1xGPTModel`` forward for image inputs (see module docstring)."""
    import megatron.core.parallel_state as ps

    _orig_forward = model.forward

    if not getattr(model, "pre_process", False):

        def _mm_passthrough(*a, mm_vision_patches=None, mm_vision_grids=None, mm_vision_positions=None, **kw):
            return _orig_forward(*a, **kw)

        model.forward = _mm_passthrough
        return

    device = torch.cuda.current_device()
    tower = build_vision_tower(
        hf_checkpoint, device=device, dtype=model.config.params_dtype, attn_impl=attn_impl, train=train
    )
    model.__dict__["_mm_towers"] = (tower,)
    spark = model.config.spark

    def _mm_forward(*a, mm_vision_patches=None, mm_vision_grids=None, mm_vision_positions=None, **kw):
        input_ids = kw.get("input_ids", a[0] if a else None)
        position_ids = kw.get("position_ids")
        packed = kw.get("packed_seq_params")
        decoder_input = model.embedding(input_ids=input_ids, position_ids=position_ids)
        if spark.normalize_tok_embeddings:
            decoder_input = model.embedding_norm(decoder_input.to(model.config.params_dtype))
        if mm_vision_patches is not None:
            assert (
                mm_vision_grids is not None and mm_vision_positions is not None
            ), "mm_vision_patches needs grids and positions"
            grids = [tuple(int(v) for v in row) for row in mm_vision_grids.tolist()]
            ctx = torch.enable_grad() if train else torch.no_grad()
            tower_start = time.perf_counter()
            with ctx:
                embeds = tower.forward_patches(mm_vision_patches.to(device), grids)
            assert (
                embeds.shape[0] == mm_vision_positions.numel()
            ), f"{embeds.shape[0]} image rows vs {mm_vision_positions.numel()} <|patch|> positions"
            seq_local = decoder_input.shape[0]
            sp_size = ps.get_tensor_model_parallel_world_size() if model.config.sequence_parallel else 1
            local, keep = natural_to_local_rows(
                mm_vision_positions.to(device),
                None if packed is None else packed.cu_seqlens_q,
                ps.get_context_parallel_world_size(),
                ps.get_context_parallel_rank(),
                sp_size,
                ps.get_tensor_model_parallel_rank() if sp_size > 1 else 0,
                seq_local,
            )
            decoder_input = decoder_input.clone()  # keep the embedding graph; rows are written in place below
            if keep.any():
                assert decoder_input.shape[1] == 1, "image merge expects the packed [s, 1, h] layout"
                decoder_input[local[keep], 0] = embeds[keep].to(decoder_input.dtype)
            _MERGE_CALLS["n"] += 1
            if _MERGE_CALLS["n"] <= 4 or _MERGE_CALLS["n"] % 50 == 0:
                logger.info(
                    "muse_spark mm merge #%d: %d images, %d patch rows -> %d image rows, %d on this rank "
                    "(seq_local=%d, cp %d/%d, sp %d/%d, tower %.3fs, decoder_input %s)",
                    _MERGE_CALLS["n"],
                    len(grids),
                    int(mm_vision_patches.shape[0]),
                    int(embeds.shape[0]),
                    int(keep.sum()),
                    seq_local,
                    ps.get_context_parallel_rank(),
                    ps.get_context_parallel_world_size(),
                    ps.get_tensor_model_parallel_rank() if sp_size > 1 else 0,
                    sp_size,
                    time.perf_counter() - tower_start,
                    str(decoder_input.dtype),
                )
        multiplier = spark.embedding_multiplier
        if multiplier != 1.0:
            decoder_input = decoder_input * multiplier
        kw["decoder_input"] = decoder_input
        return _orig_forward(*a, **kw)

    model.forward = _mm_forward
