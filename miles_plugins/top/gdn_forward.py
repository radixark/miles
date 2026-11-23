"""Qwen GDN TOP trainer forward: every sample as one packed document through serving's GDN.

Under TOP, rollout GDN decode follows serving's fixed-tile chunk prefill bit for bit, so each GDN layer runs that
chunk prefill over every sample from position 0 and zero state. One microbatch stays one model call, so
expert-parallel ranks keep the same number of forwards.
"""

import functools
from dataclasses import dataclass, field

import torch
from megatron.core.utils import get_attr_wrapped_model

from miles.backends.megatron_utils.parallel import PackedSeqParamsWithHostCuSeqlens
from miles_plugins.top.host_copy import device_tensor
from miles_plugins.top.lm_head import deferred_logits_processor


@dataclass
class GDNPackedSeqParams(PackedSeqParamsWithHostCuSeqlens):
    """Packed samples (or, under context parallelism, this rank's pieces of them) for the GDN hybrid layers."""

    doc_starts: tuple = field(kw_only=True, default=())  # absolute first position per document (CP pieces)
    # [3, rows] mRoPE positions of this rank's rows, when the run's samples carry them (media or not).
    mrope_positions: torch.Tensor | None = field(kw_only=True, default=None)
    gdn_cp: object = field(kw_only=True, default=None)  # gdn_cp.GDNCPLayout under context parallelism
    decode_rows: tuple = field(kw_only=True, default=())  # packed rows the attention decode split computes


def gdn_forward(
    model, *, batch, pad_multiple: int, grid: int, hf_checkpoint: str, deferred_lm_head: bool = False, **kwargs
):
    """Run the microbatch's samples as packed documents; returns logits in the original packed layout. ``grid`` is
    serving's chunk prefill grid, where context-parallel pieces may resume.

    ``deferred_lm_head``: the last stage returns ``TopDeferredLogits`` (the LM head's input rows) for the row-chunked
    TOP log-prob instead of materialized logits. Under context parallelism (all-gather CP) every rank builds the same
    documents, runs its share (gdn_cp), and returns the rows of its slice of the original stream.

    Samples with images or video (``batch["multimodal_train_inputs"]``, from SGLang's token-space processor): the
    frozen vision tower of ``hf_checkpoint`` encodes the microbatch's media and its rows replace the placeholder
    embeddings this rank holds; full attention rotates by the samples' mRoPE positions."""
    from megatron.core import parallel_state as mpu

    docs = [tokens[:n] for tokens, n in zip(batch["unconcat_tokens"], batch["total_lengths"], strict=True)]
    rows = sum(batch["total_lengths"])
    pad = (pad_multiple - rows % pad_multiple) % pad_multiple
    if pad:
        docs.append(docs[0].new_zeros(pad))
    original_rows = batch["tokens"].shape[1]
    device = docs[0].device
    # (document, first position, end position) of each document this rank runs, in its packed order.
    pieces = [(d, 0, doc.numel()) for d, doc in enumerate(docs)]
    doc_offsets = [0]
    for doc in docs:
        doc_offsets.append(doc_offsets[-1] + doc.numel())
    layout, starts = None, ()
    if mpu.get_context_parallel_world_size() > 1:
        from miles_plugins.top.gdn_cp import RecomputeStash, build_layout

        mine, layout = build_layout(
            lengths=[doc.numel() for doc in docs],
            sample_rows=rows,
            local_original_rows=original_rows,
            device=device,
            group=mpu.get_context_parallel_group(),
            grid=grid,
        )
        pieces = list(mine)
        docs = [docs[d][a:b] for d, a, b in mine]
        starts = tuple(a for _, a, _ in mine)
        layout.stash = RecomputeStash(torch.is_grad_enabled())
        rows = layout.output_rows
    else:
        # The samples are the original stream's rows in order; its padding rows carry no loss, and row 0 serves.
        rows = device_tensor(list(range(rows)) + [0] * (original_rows - rows), device=device)
    doc_lens = [doc.numel() for doc in docs]
    cu_host = [0]
    for length in doc_lens:
        cu_host.append(cu_host[-1] + length)
    cu = device_tensor(cu_host, device=device, dtype=torch.int)
    packed = GDNPackedSeqParams(
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=max(doc_lens),
        max_seqlen_kv=max(doc_lens),
        qkv_format="thd",
        cu_seqlens_host=tuple(cu_host),
        doc_starts=starts,
        mrope_positions=_piece_mrope_positions(batch.get("sample_mrope_position_ids"), pieces, device),
        gdn_cp=layout,
        decode_rows=_split_rows(sum(doc_lens)),
    )
    input_ids = torch.cat(docs)
    media = batch.get("multimodal_train_inputs")
    if media and get_attr_wrapped_model(model, "pre_process"):
        kwargs["decoder_input"], input_ids = _embed_with_media(
            model, input_ids, media, pieces, doc_offsets, hf_checkpoint=hf_checkpoint
        )
    select = _cp_rows_selector(rows, layout)
    handle = None
    if get_attr_wrapped_model(model, "post_process"):
        if deferred_lm_head:
            kwargs["output_processor"] = deferred_logits_processor(rows, select=select)
        else:
            # The LM head is a row-wise batch-invariant GEMM: scoring only the selected rows gives the same bytes.
            output_layer = get_attr_wrapped_model(model, "output_layer")
            handle = output_layer.register_forward_pre_hook(
                lambda module, args, kw: ((select(args[0]), *args[1:]), kw), with_kwargs=True
            )
    try:
        # Earlier pipeline stages hand the hidden states on; the last returns logits in the packed layout.
        return model(
            input_ids=input_ids.unsqueeze(0),
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=packed,
            **kwargs,
        )
    finally:
        if handle is not None:
            handle.remove()


def _split_rows(rows: int) -> tuple:
    """Rows the attention decode split computes: every row, when the split is on (decode_split.bound_split). Serving
    prefills attention through the decode split's fixed key positions as well as decoding through them, so one
    row's attention bytes do not depend on whether serving prefilled or decoded it."""
    from miles_plugins.top.decode_split import bound_split

    return tuple(range(rows)) if bound_split() else ()


def _cp_rows_selector(rows, layout):
    """Rows of the LM head input [S, 1, H] or [S, H]: local rows, or (CP) rows of all ranks' gathered final hidden."""

    def select(hidden):
        squeeze = hidden.dim() == 3
        flat = hidden[:, 0] if squeeze else hidden
        if layout is not None:
            from miles_plugins.top.gdn_cp import all_gather_rows

            flat = all_gather_rows(flat, layout)
        out = flat.index_select(0, rows)
        return out[:, None] if squeeze else out

    return select


# The tower's inputs and row placement as get_batch packs them: (patches, grids, natural packed row of each tower row).
_MEDIA_KEYS = (
    ("pixel_values", "image_grid_thw", "mm_image_positions"),
    ("pixel_values_videos", "video_grid_thw", "mm_video_positions"),
)


def _piece_mrope_positions(sample_positions, pieces, device):
    """[3, rows] positions of this rank's pieces from each sample's [tokens, 3]; None when the batch carries none.
    The padding document (one past the samples) counts from 0 on every axis, as a text document does."""
    if sample_positions is None:
        return None
    parts = []
    for d, a, b in pieces:
        if d < len(sample_positions):
            parts.append(sample_positions[d][a:b].to(device))
        else:
            parts.append(torch.arange(a, b, device=device)[:, None].expand(-1, 3))
    return torch.cat(parts).T.contiguous()


def _embed_with_media(model, input_ids, media, pieces, doc_offsets, *, hf_checkpoint: str):
    """(the embedding of this rank's rows [S, 1, H] with the vision tower's rows written over the placeholders it
    holds, the rows' token ids as serving's language model reads them).

    ``media`` places every tower row by its row in the natural stream (samples back to back), which ``pieces`` map
    to this rank's rows; a row on another context-parallel rank is dropped here and written there. The tower is
    batch-invariant, so every rank encoding the whole microbatch's media gives each row one value.

    Serving replaces a placeholder's id by a per-media hash for its prefix cache and clamps it to the last
    vocabulary id before the language model runs; a layer that hashes token ids (Qwen3.8-Flash-Next's n-gram
    embedding) reads that id, so the trainer's ids carry it at the placeholders."""
    from miles_plugins.top.vision import top_vision_tower

    embedding = get_attr_wrapped_model(model, "embedding")
    decoder_input = embedding(input_ids=input_ids.unsqueeze(0), position_ids=None)
    input_ids = input_ids.clone()
    placeholder_id = _serving_vocab_size(hf_checkpoint) - 1
    local_row = torch.full((doc_offsets[-1],), -1, dtype=torch.long, device=input_ids.device)
    start = 0
    for d, a, b in pieces:
        local_row[doc_offsets[d] + a : doc_offsets[d] + b] = torch.arange(
            start, start + b - a, device=input_ids.device
        )
        start += b - a
    tower = top_vision_tower(hf_checkpoint, device=input_ids.device)
    decoder_input = decoder_input.clone()
    for patches_key, grid_key, positions_key in _MEDIA_KEYS:
        if patches_key not in media:
            continue
        rows = tower(media[patches_key], media[grid_key])
        positions = media[positions_key]
        if rows.shape[0] != positions.numel():
            raise ValueError(f"[top] {rows.shape[0]} vision rows for {positions.numel()} placeholder tokens")
        local = local_row[positions.to(input_ids.device)]
        held = local >= 0
        decoder_input[local[held], 0] = rows[held].to(decoder_input.dtype)
        input_ids[local[held]] = placeholder_id
    return decoder_input, input_ids


@functools.cache
def _serving_vocab_size(hf_checkpoint: str) -> int:
    """The rows of serving's token embedding: the checkpoint's vocabulary, without Megatron's padding."""
    from miles.utils.hf_utils.config import load_hf_config

    config = load_hf_config(hf_checkpoint)
    text = getattr(
        config, "text_config", config
    )  # config-access-exempt: HF configs may wrap the text model's config in text_config
    return text.vocab_size


def validate_gdn_forward(args, batch) -> None:
    if args.backend.mtp_num_layers or args.enable_witness:
        raise ValueError("[top] GDN TOP trains samples without MTP or witness ids")
    media = batch.get("multimodal_train_inputs")
    if media and not args.multimodal_token_space_processing:
        raise ValueError(
            "[top] GDN TOP takes media rows and mRoPE positions from SGLang's token-space processor; pass "
            "--multimodal-token-space-processing and --sglang-enable-token-space-processor"
        )
    if media and any(key not in {k for keys in _MEDIA_KEYS for k in keys} for key in media):
        raise ValueError(f"[top] GDN TOP has no program for the media inputs {sorted(media)}")
    if media and batch.get("sample_mrope_position_ids") is None:
        raise ValueError("[top] samples with media need their mRoPE positions from the token-space processor")
    if args.qkv_format != "thd" or (args.context_parallel_size > 1) != bool(args.allgather_cp):
        raise ValueError("[top] GDN TOP requires THD packing, with --allgather-cp exactly when CP > 1")
