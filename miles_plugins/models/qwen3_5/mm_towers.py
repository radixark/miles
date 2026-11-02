"""Wire the frozen Qwen3.5 vision tower into the plugin text GPTModel (Inkling ``mm_towers`` pattern).

``pre_process`` ranks build the tower from the HF checkpoint and write its output rows into ``decoder_input``;
other pipeline stages swallow the multimodal kwargs. The tower lives in ``model.__dict__`` (not a registered
submodule), so the optimizer, checkpoints, weight sync and ``named_parameters`` never see it.

Where each row goes comes from SGLang's token-space processor on the trainer (``mm_spans``), the same
computation the rollout engines run. ``get_batch`` turns those token indices into ``mm_image_positions`` /
``mm_video_positions``: the row of this rank's packed input for every tower output row, -1 where another
context-parallel rank holds the token. Every CP rank runs the tower on all of its micro-batch's media. The
mRoPE positions reach the model as its ``position_ids``, untouched here.
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
)


def wire_mm_towers(model, hf_checkpoint: str) -> None:
    """Wrap a Qwen3.5 GPTModel's forward to merge the vision tower's image and video rows into its embedding."""
    import megatron.core.parallel_state as ps

    orig_forward = model.forward

    if not model.pre_process:

        def passthrough(*args, **kwargs):
            # Only the first pipeline stage embeds media.
            for name in MM_KWARGS:
                kwargs.pop(name, None)
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
        return orig_forward(*args, **kwargs)

    model.forward = mm_forward


def _scatter_rows(decoder_input: torch.Tensor, rows: torch.Tensor, positions: torch.Tensor, *, sp_rank: int):
    """Write ``rows`` at the packed ``positions`` that fall in this rank's [s_local, 1, h] window; -1 skips a row."""
    s_local = decoder_input.shape[0]
    local = positions - sp_rank * s_local
    keep = (local >= 0) & (local < s_local)
    # Clone instead of writing in place, so the embedding's autograd graph stays intact.
    decoder_input = decoder_input.clone()
    decoder_input[local[keep], 0] = rows[keep].to(decoder_input.dtype)
    return decoder_input
