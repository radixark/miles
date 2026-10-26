"""Wire the frozen Qwen3.5 vision tower into the plugin text GPTModel (Inkling ``mm_towers`` pattern).

``pre_process`` ranks build the tower from the HF checkpoint and write its output rows into ``decoder_input``;
other pipeline stages swallow the multimodal kwargs. The tower lives in ``model.__dict__`` (not a registered
submodule), so the optimizer, checkpoints, weight sync and ``named_parameters`` never see it.

Where each row goes and the mRoPE positions both come from SGLang's token-space processor on the trainer
(``mm_spans`` and ``compute_mrope_positions``), the same computation the rollout engines run:

- ``mm_image_positions`` / ``mm_video_positions``: [num_rows] token index of every tower output row, in the
  natural packed row (``get_batch`` shifts each sample's positions by its offset);
- ``mrope_position_ids``: [num_tokens, 3] mRoPE positions of every packed token (stored token-major so the
  samples concatenate along dim 0).
"""

import torch

from miles_plugins.models.qwen3_5.vision import load_vision_tower

MM_KWARGS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "mm_image_positions",
    "mm_video_positions",
    "mrope_position_ids",
)


def wire_mm_towers(model, hf_checkpoint: str) -> None:
    """Wrap a Qwen3.5 GPTModel's forward to merge images and videos and to take SGLang's mRoPE positions."""
    import megatron.core.parallel_state as ps

    orig_forward = model.forward

    if not model.pre_process:

        def passthrough(*args, mrope_position_ids=None, **kwargs):
            # Every stage rotates with the mRoPE positions; only the first one embeds media.
            for name in MM_KWARGS:
                kwargs.pop(name, None)
            kwargs["position_ids"] = _mrope_position_ids(mrope_position_ids, kwargs["input_ids"])
            return orig_forward(*args, **kwargs)

        model.forward = passthrough
        return

    tower = load_vision_tower(
        hf_checkpoint, device=torch.device("cuda", torch.cuda.current_device()), dtype=model.config.params_dtype
    )
    model.__dict__["_mm_tower"] = tower

    def mm_forward(
        *args,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        mm_image_positions=None,
        mm_video_positions=None,
        mrope_position_ids=None,
        **kwargs,
    ):
        input_ids = kwargs["input_ids"]
        decoder_input = model.embedding(input_ids=input_ids, position_ids=None)
        sequence_parallel = model.config.sequence_parallel and ps.get_tensor_model_parallel_world_size() > 1
        sp_rank = ps.get_tensor_model_parallel_rank() if sequence_parallel else 0
        for patches, grid_thw, positions in (
            (pixel_values, image_grid_thw, mm_image_positions),
            (pixel_values_videos, video_grid_thw, mm_video_positions),
        ):
            if patches is None:
                continue
            rows = tower(patches, grid_thw)
            assert (
                rows.shape[0] == positions.numel()
            ), f"{rows.shape[0]} vision rows vs {positions.numel()} placeholder positions"
            decoder_input = _scatter_rows(decoder_input, rows, positions.to(decoder_input.device), sp_rank=sp_rank)
        kwargs["decoder_input"] = decoder_input
        kwargs["position_ids"] = _mrope_position_ids(mrope_position_ids, input_ids)
        return orig_forward(*args, **kwargs)

    model.forward = mm_forward


def _scatter_rows(decoder_input: torch.Tensor, rows: torch.Tensor, positions: torch.Tensor, *, sp_rank: int):
    """Write ``rows`` at the natural packed ``positions`` that fall in this rank's [s_local, 1, h] window."""
    s_local = decoder_input.shape[0]
    local = positions - sp_rank * s_local
    keep = (local >= 0) & (local < s_local)
    # Clone instead of writing in place, so the embedding's autograd graph stays intact.
    decoder_input = decoder_input.clone()
    decoder_input[local[keep], 0] = rows[keep].to(decoder_input.dtype)
    return decoder_input


def _mrope_position_ids(mrope_position_ids: torch.Tensor | None, input_ids: torch.Tensor) -> torch.Tensor:
    """[3, 1, T] mRoPE positions for the [1, T] packed input, continuing them over the trailing padding."""
    if mrope_position_ids is None:
        raise ValueError(
            "the Qwen3.5 VL model takes its mRoPE positions from SGLang's token-space processor; "
            "set --multimodal-token-space-processing"
        )
    positions = mrope_position_ids.to(input_ids.device).T
    padding = input_ids.shape[1] - positions.shape[1]
    assert padding >= 0, f"{positions.shape[1]} mRoPE positions for {input_ids.shape[1]} tokens"
    if padding:
        start = int(positions.max()) + 1
        tail = torch.arange(start, start + padding, device=positions.device).expand(3, -1)
        positions = torch.cat([positions, tail], dim=1)
    return positions.unsqueeze(1)
