"""DeepSeek-V4.1 images under TOP: the frozen tower's span rows written over the image tokens' embeddings.

SGLang's token-space processor expands each image placeholder to its whole span, ``image_token_id`` at every
position (start, rows, newlines, end), and the trainer rebuilds the same BF16 patches with it. Serving embeds the
text tokens, then writes every image's span rows (``deepseek_v41_vit.image_span``) at its positions; the language
model keeps ``image_token_id`` there, which routes those rows by the VL bias and keeps their streams through Engram.
The trainer does the same on the first pipeline stage and hands Megatron the result as ``decoder_input``.
"""

from __future__ import annotations

import functools

import torch
from megatron.core.utils import get_attr_wrapped_model

_MEDIA_KEYS = ("pixel_values", "image_grid_hw", "mm_image_positions")


@functools.cache
def _checkpoint_model_type(path: str) -> str | None:
    from miles.utils.hf_utils.config import load_hf_config

    return getattr(
        load_hf_config(path), "model_type", None
    )  # config-access-exempt: third-party HF configurations may omit model_type


def is_top_dsv41(args) -> bool:
    """TOP v1 on a DeepSeek-V4.1 checkpoint."""
    from miles.true_on_policy.contracts import uses_true_on_policy_v1

    return uses_true_on_policy_v1(args) and _checkpoint_model_type(args.hf_checkpoint) == "deepseek_v41"


def dsv41_media_forward_kwargs(model, *, tokens: torch.Tensor, media: dict | None, hf_checkpoint: str) -> dict:
    """The forward kwargs a V4.1 micro-batch with images adds: this stage's ``decoder_input`` with the image rows
    written in (first stage only; later stages take the previous stage's streams). Empty for a text micro-batch."""
    if not media:
        return {}
    unknown = set(media) - set(_MEDIA_KEYS)
    if unknown or set(media) != set(_MEDIA_KEYS):
        raise NotImplementedError(
            f"[top] DeepSeek-V4.1 trains images only, from the token-space processor's {_MEDIA_KEYS}; got "
            f"{sorted(media)}"
        )
    if not get_attr_wrapped_model(model, "pre_process"):
        return {}
    return {"decoder_input": _decoder_input(model, tokens=tokens, media=media, hf_checkpoint=hf_checkpoint)}


def _decoder_input(model, *, tokens, media, hf_checkpoint) -> torch.Tensor:
    """[T, 1, H]: the token embedding of the packed micro-batch [1, T], the image spans written at their rows."""
    from miles_plugins.top.dsv41_vision import top_v41_vision_tower

    embedding = get_attr_wrapped_model(model, "embedding")
    decoder_input = embedding(input_ids=tokens, position_ids=None).clone()
    tower = top_v41_vision_tower(hf_checkpoint, device=tokens.device)
    grids = [(int(h), int(w)) for h, w in media["image_grid_hw"].tolist()]
    patches = media["pixel_values"].split([h * w for h, w in grids])
    rows = torch.cat([tower(p, n_vit_h=h, n_vit_w=w) for p, (h, w) in zip(patches, grids, strict=True)])
    positions = media["mm_image_positions"].to(tokens.device)
    if rows.shape[0] != positions.numel():
        raise ValueError(f"[top] {rows.shape[0]} image span rows for {positions.numel()} image tokens")
    decoder_input[positions, 0] = rows.to(decoder_input.dtype)
    return decoder_input
