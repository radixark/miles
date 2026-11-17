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

import logging

import torch

from miles.backends.training_utils.data.context_parallel import scatter_cp_rows
from miles_plugins.models.qwen3_5.vision import load_vision_tower

logger = logging.getLogger(__name__)

MM_KWARGS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "mm_image_positions",
    "mm_video_positions",
)


def check_multimodal_setup(config, sglang) -> None:
    """Refuse RoPE fusion the engine cannot mirror, and warn when tensor-parallel engines shard the vision tower."""
    try:
        from megatron.core.fusions.fused_mrope import is_fused_mrope_available
    except ImportError:
        is_fused_mrope_available = lambda: False  # noqa: E731
    # Without the fused kernel Megatron hands per-token mRoPE frequencies to the THD RoPE kernel, which reads them as
    # a position table per sequence and rotates every packed sample after the first with the first one's positions.
    if config.apply_rope_fusion and (
        config.rotary_interleaved or config.fused_single_qkv_rope or not is_fused_mrope_available()
    ):
        raise ValueError("mRoPE with --apply-rope-fusion needs Megatron's fused mRoPE kernel; pass --no-rope-fusion")
    # The engine shards the vision tower across its tensor-parallel ranks unless the DP encoder is on; the trainer
    # runs it whole. The launcher sizes TP from the group's GPUs unless the group overrides tp_size.
    for model in sglang.models:
        for group in model.server_groups if model.update_weights else ():
            tp_size = group.overrides.get("tp_size", group.num_gpus_per_engine)
            if tp_size > 1 and not (sglang.base_args | group.overrides).get("mm_enable_dp_encoder"):
                logger.warning(
                    "tensor-parallel rollout engines shard the vision tower while the trainer runs it whole, so its "
                    "rows differ by the reduction order; pass --sglang-mm-enable-dp-encoder to align them"
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
            decoder_input = scatter_cp_rows(decoder_input, rows, positions.to(decoder_input.device), sp_rank=sp_rank)
        kwargs["decoder_input"] = decoder_input
        return orig_forward(*args, **kwargs)

    model.forward = mm_forward
