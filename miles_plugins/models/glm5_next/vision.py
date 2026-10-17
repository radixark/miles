"""GLM-5.3 visual tower and Megatron model provider.

The four ``Glm5Next*`` classes subclass their ``GlmOcr*`` counterparts from Transformers to
pick up GLM-5.3's clamped SwiGLU and its patch merger; everything else here -- the model
provider, the image-embedding scatter, the trainable scopes and the checkpoint hooks -- is
Miles' own. No Transformers source is copied into this file, so it carries no upstream
header; ``processor.py`` next door does copy code and does.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from torch.utils.checkpoint import checkpoint
from transformers.models.glm_ocr.modeling_glm_ocr import (
    GlmOcrVisionBlock,
    GlmOcrVisionMlp,
    GlmOcrVisionModel,
    GlmOcrVisionPatchMerger,
)

from miles.utils.hf_utils.config import load_hf_config

# Imported from the argparse module, not defined here: that module must stay importable
# without pulling this one in, since the parser is built for every Megatron run.
from miles_plugins.models.glm5_next.arguments import VISION_TRAINABLE_CHOICES

if TYPE_CHECKING:
    from megatron.core.models.gpt import GPTModel

logger = logging.getLogger(__name__)

_VISUAL_PREFIX = "model.visual."

# Attribute name of the visual tower once it is a real submodule, which is also the
# ``visual.*`` prefix its parameters take in the optimizer and the distributed checkpoint.
VISUAL_MODULE_NAME = "visual"

# The image-feature projection that runs after the ViT blocks. ``patch_embed`` and
# ``blocks`` are the pretrained encoder itself and stay frozen unless "full" is requested.
_PROJECTOR_MODULES = ("post_layernorm", "downsample", "merger")


class Glm5NextVisionMLP(GlmOcrVisionMlp):
    def __init__(self, config) -> None:
        super().__init__(config, bias=config.attention_bias)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states).clamp(max=self.swiglu_limit)
        up = self.up_proj(hidden_states).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm5NextVisionBlock(GlmOcrVisionBlock):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.mlp = Glm5NextVisionMLP(config)


class Glm5NextVisionPatchMerger(GlmOcrVisionPatchMerger):
    def __init__(self, config) -> None:
        super().__init__(
            dim=config.out_hidden_size,
            context_dim=config.projection_intermediate_size,
            hidden_act=config.hidden_act,
            bias=False,
        )
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act1(self.post_projection_norm(self.proj(hidden_states)))
        gate = self.gate_proj(hidden_states).clamp(max=self.swiglu_limit)
        up = self.up_proj(hidden_states).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class _RecomputingVisionBlock(Glm5NextVisionBlock):
    """A ViT block that recomputes its activations in the backward pass.

    Subclassing rather than wrapping keeps the parameter names -- and with them the optimizer
    and checkpoint keys -- byte-identical to the non-recomputing block.

    Transformers' own ``gradient_checkpointing_enable()`` does reach these blocks:
    ``GlmOcrVisionBlock`` inherits ``GradientCheckpointingLayer``, whose ``__call__``
    dispatches to ``_gradient_checkpointing_func``. It gates on ``self.training`` alone,
    though, so it also wraps forwards whose result no backward pass will consume. Gating on
    the autograd state keeps the cost where the gradient is.
    """

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        if not (torch.is_grad_enabled() and hidden_states.requires_grad):
            return super().forward(hidden_states, **kwargs)
        return checkpoint(
            lambda states: super(_RecomputingVisionBlock, self).forward(states, **kwargs),
            hidden_states,
            use_reentrant=False,
        )


class Glm5NextVisionModel(GlmOcrVisionModel):
    """Vision-only portion of the upstream GLM-5.3 model."""

    def __init__(self, config, *, recompute_blocks: bool = False) -> None:
        super().__init__(config)
        # Selecting the block class, rather than calling gradient_checkpointing_enable(),
        # keeps the recompute decision tied to the trainable scope and the autograd state
        # (see _RecomputingVisionBlock). Either way the upstream forward -- which only
        # iterates self.blocks -- stays untouched.
        block_class = _RecomputingVisionBlock if recompute_blocks else Glm5NextVisionBlock
        self.blocks = torch.nn.ModuleList(block_class(config) for _ in range(config.depth))
        self.merger = Glm5NextVisionPatchMerger(config)


@contextmanager
def _default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _visual_weight_files(hf_checkpoint: str) -> dict[str, list[str]]:
    checkpoint = Path(hf_checkpoint)
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as index_file:
            weight_map = json.load(index_file)["weight_map"]
        weights_by_file: dict[str, list[str]] = {}
        for name, filename in weight_map.items():
            if name.startswith(_VISUAL_PREFIX):
                weights_by_file.setdefault(filename, []).append(name)
        return weights_by_file

    single_file = checkpoint / "model.safetensors"
    if not single_file.is_file():
        raise FileNotFoundError(f"No model.safetensors.index.json or model.safetensors found in {hf_checkpoint}")
    with safe_open(single_file, framework="pt") as shard:
        names = [name for name in shard.keys() if name.startswith(_VISUAL_PREFIX)]
    return {single_file.name: names}


def _load_visual_state_dict(hf_checkpoint: str) -> dict[str, torch.Tensor]:
    weights_by_file = _visual_weight_files(hf_checkpoint)
    if not any(weights_by_file.values()):
        raise RuntimeError(f"No {_VISUAL_PREFIX} weights found in {hf_checkpoint}")

    state_dict = {}
    for filename, names in weights_by_file.items():
        with safe_open(Path(hf_checkpoint) / filename, framework="pt") as shard:
            for name in names:
                state_dict[name.removeprefix(_VISUAL_PREFIX)] = shard.get_tensor(name)
    return state_dict


def _trainable_parameter_names(visual: Glm5NextVisionModel, trainable: str) -> set[str]:
    """Names of the visual parameters the requested scope trains."""
    if trainable not in VISION_TRAINABLE_CHOICES:
        raise ValueError(f"--glm5-vision-trainable must be one of {VISION_TRAINABLE_CHOICES}, got {trainable!r}")
    if trainable == "none":
        return set()
    if trainable == "full":
        return {name for name, _ in visual.named_parameters()}
    return {
        f"{module_name}.{name}"
        for module_name in _PROJECTOR_MODULES
        for name, _ in getattr(visual, module_name).named_parameters()
    }


def _apply_trainable_scope(visual: Glm5NextVisionModel, trainable: str, *, sequence_parallel: bool) -> None:
    """Select which visual parameters train, and mark their gradients for TP reduction."""
    trainable_names = _trainable_parameter_names(visual, trainable)
    for name, parameter in visual.named_parameters():
        requires_grad = name in trainable_names
        parameter.requires_grad_(requires_grad)
        if requires_grad:
            # The tower is replicated across the TP domain but each rank's loss only covers
            # its own sequence slice, so every rank holds a partial gradient. Megatron sums
            # these in finalize_model_grads._allreduce_non_tensor_model_parallel_grads, which
            # gates on config.sequence_parallel and this per-parameter flag.
            parameter.sequence_parallel = sequence_parallel
    # Frozen towers stay in eval mode; a trainable tower must honour train/eval like the
    # language model so dropout and any norm statistics follow the training loop.
    visual.train(trainable != "none")


def build_glm5_next_visual(
    hf_checkpoint: str,
    *,
    device: torch.device | int,
    dtype: torch.dtype,
    trainable: str = "none",
    sequence_parallel: bool = False,
    recompute_blocks: bool = False,
) -> Glm5NextVisionModel:
    """Construct and strictly load the checkpoint's visual tower."""
    hf_config = load_hf_config(hf_checkpoint)
    vision_config = hf_config.vision_config
    vision_config._attn_implementation = "sdpa"
    with torch.device(device), _default_dtype(dtype):
        visual = Glm5NextVisionModel(vision_config, recompute_blocks=recompute_blocks)

    visual.load_state_dict(_load_visual_state_dict(hf_checkpoint), strict=True)
    _apply_trainable_scope(visual, trainable, sequence_parallel=sequence_parallel)
    return visual


def _scatter_image_embeddings(
    model: GPTModel,
    decoder_input: torch.Tensor,
    positions: torch.Tensor,
    image_embeddings: torch.Tensor,
) -> torch.Tensor:
    from megatron.core import parallel_state

    positions = positions.to(decoder_input.device)
    image_embeddings = image_embeddings.to(device=decoder_input.device, dtype=decoder_input.dtype)
    if model.config.sequence_parallel and parallel_state.get_tensor_model_parallel_world_size() > 1:
        sequence_length = decoder_input.shape[0]
        rank = parallel_state.get_tensor_model_parallel_rank()
        local_positions = positions - rank * sequence_length
        selected = (local_positions >= 0) & (local_positions < sequence_length)
        local_positions = local_positions[selected]
        image_embeddings = image_embeddings[selected]
    else:
        local_positions = positions

    decoder_input = decoder_input.clone()
    if local_positions.numel():
        decoder_input[local_positions, 0] = image_embeddings
    elif image_embeddings.requires_grad:
        # A TP sequence slice may contain no image tokens. Keep its zero-valued
        # tower dependency so frozen-language ranks still join activation backward
        # collectives and fire the replicated visual DDP hooks.
        decoder_input = decoder_input + image_embeddings.sum() * 0
    return decoder_input


def _reject_non_image_inputs(kwargs: dict) -> None:
    unsupported_keys = (
        "pixel_values_videos",
        "video_grid_thw",
        "videos",
        "audio",
        "audios",
        "input_features",
        "input_values",
        "audio_values",
    )
    for key in unsupported_keys:
        value = kwargs.pop(key, None)
        if value is not None:
            nonempty = value.numel() > 0 if isinstance(value, torch.Tensor) else bool(value)
            if nonempty:
                raise ValueError(f"GLM-5.3 image SFT does not support video or audio inputs: {key}")


def wire_glm5_next_visual(
    model: GPTModel,
    hf_checkpoint: str,
    *,
    device: torch.device | None = None,
    trainable: str = "none",
    recompute_blocks: bool = False,
    freeze_language: bool = False,
) -> None:
    """Attach image encoding while leaving language parameter names unchanged.

    With ``trainable="none"`` the tower is held outside the module tree, which keeps it out
    of the optimizer and the distributed checkpoint. Any other scope registers it as the
    ``visual`` submodule, so Megatron's parameter walks pick its trainable parameters up for
    the optimizer, the DDP gradient buckets and the checkpoint.
    """
    if freeze_language:
        if trainable == "none":
            raise ValueError("--glm5-freeze-language requires a trainable visual scope")
        # Freeze before registering the tower and before Megatron builds DDP buckets.
        # Do not use no_grad around the LLM: visual gradients must pass through it.
        model.requires_grad_(False)
    original_forward = model.forward
    if not model.pre_process:

        def _passthrough(*args, pixel_values=None, image_grid_thw=None, **kwargs):
            _reject_non_image_inputs(kwargs)
            return original_forward(*args, **kwargs)

        model.forward = _passthrough
        return

    device = device or torch.device("cuda", torch.cuda.current_device())
    hf_config = load_hf_config(hf_checkpoint)
    visual = build_glm5_next_visual(
        hf_checkpoint,
        device=device,
        dtype=model.config.params_dtype,
        trainable=trainable,
        sequence_parallel=model.config.sequence_parallel,
        # Only "full" trains the ViT blocks, so only it stores their activations.
        recompute_blocks=recompute_blocks and trainable == "full",
    )
    train_visual = trainable != "none"
    if train_visual:
        # A registered submodule puts visual.* into named_parameters(), which is what the
        # optimizer, DDP and sharded_state_dict all walk.
        model.add_module(VISUAL_MODULE_NAME, visual)
    else:
        visual.requires_grad_(False).eval()
        # Keep the frozen tower outside the Megatron optimizer and distributed checkpoint.
        model.__dict__["_glm5_next_visual"] = visual
    image_token_id = hf_config.image_token_id

    def _visual_tower() -> Glm5NextVisionModel:
        return getattr(model, VISUAL_MODULE_NAME) if train_visual else model.__dict__["_glm5_next_visual"]

    def _multimodal_forward(*args, pixel_values=None, image_grid_thw=None, **kwargs):
        _reject_non_image_inputs(kwargs)
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if pixel_values is None:
            if image_grid_thw is not None or (input_ids is not None and (input_ids == image_token_id).any()):
                raise ValueError("GLM-5.3 image tokens and image_grid_thw require pixel_values")
            return original_forward(*args, **kwargs)
        if image_grid_thw is None:
            raise ValueError("pixel_values requires image_grid_thw")
        if kwargs.get("decoder_input") is not None:
            raise ValueError("GLM-5.3 image SFT cannot combine pixel_values with precomputed decoder_input")

        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("GLM-5.3 VLM training expects packed input_ids with shape [1, sequence]")
        position_ids = kwargs.get("position_ids", args[1] if len(args) > 1 else None)
        decoder_input = model.embedding(input_ids=input_ids, position_ids=position_ids)

        tower = _visual_tower()
        # The grad context must cover only the tower forward: _scatter_image_embeddings clones
        # decoder_input, and cloning under no_grad would detach the embedding graph.
        with torch.enable_grad() if train_visual else torch.no_grad():
            image_embeddings = tower(
                pixel_values.to(device=device, dtype=tower.dtype),
                grid_thw=image_grid_thw.to(device=device),
            ).pooler_output
        image_positions = torch.nonzero(input_ids.reshape(-1) == image_token_id, as_tuple=False).flatten()
        if image_positions.numel() != image_embeddings.shape[0]:
            raise ValueError(
                f"GLM-5.3 image token/feature mismatch: {image_positions.numel()} token(s), "
                f"{image_embeddings.shape[0]} feature row(s)"
            )
        kwargs["decoder_input"] = _scatter_image_embeddings(
            model,
            decoder_input,
            image_positions,
            image_embeddings,
        )
        return original_forward(*args, **kwargs)

    model.forward = _multimodal_forward
    if freeze_language:
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable_names or any(not name.startswith(f"{VISUAL_MODULE_NAME}.") for name in trainable_names):
            raise RuntimeError("Frozen-language image SFT must train only visual.* parameters")
        logger.info(
            "GLM-5.3 language frozen; local visual scope=%s, trainable tensors=%s",
            trainable,
            len(trainable_names),
        )


def _checkpoint_has_visual_weights(load_dir: str | Path) -> bool:
    """Whether a torch_dist checkpoint already stores ``visual.*`` parameters."""
    from torch.distributed.checkpoint import FileSystemReader

    path = Path(load_dir)
    step_file = path / "latest_checkpointed_iteration.txt"
    if step_file.is_file():
        step = step_file.read_text().strip()
        path = path / (step if step == "release" else f"iter_{int(step):07d}")
    if not path.is_dir():
        return False
    return any(
        key.split(".")[0] == VISUAL_MODULE_NAME for key in FileSystemReader(path).read_metadata().state_dict_metadata
    )


@contextmanager
def hide_visual_missing_from_checkpoint(model_chunks, load_dir: str | None):
    """Hide a trainable visual tower while loading a checkpoint that predates it.

    A fresh run loads the language-only ``--ref-load`` conversion, which has no ``visual.*``
    entries. Megatron's default ``--dist-ckpt-strictness=assume_ok_unexpected`` leaves those
    keys for the load strategy to reject, so the tower has to leave the module tree for the
    duration of the load, exactly as ``_hide_critic_value_head_from_hf_load`` does for the
    critic head. The HF checkpoint already supplied these weights at build time, so skipping
    them here keeps them; a checkpoint that does carry them loads normally.
    """
    from megatron.core.utils import unwrap_model

    hidden = []
    if load_dir and not _checkpoint_has_visual_weights(load_dir):
        for chunk in unwrap_model(model_chunks):
            visual = getattr(chunk, VISUAL_MODULE_NAME, None)
            if isinstance(visual, Glm5NextVisionModel):
                delattr(chunk, VISUAL_MODULE_NAME)
                hidden.append((chunk, visual))
        if hidden:
            logger.info(
                "%s has no %s.* weights; keeping the HF-loaded visual tower across this load",
                load_dir,
                VISUAL_MODULE_NAME,
            )
    try:
        yield
    finally:
        for chunk, visual in hidden:
            chunk.add_module(VISUAL_MODULE_NAME, visual)


def has_trainable_glm5_next_visual(model_chunks) -> bool:
    """Whether any model chunk carries the visual tower as a registered submodule."""
    from megatron.core.utils import unwrap_model

    return any(
        isinstance(getattr(chunk, VISUAL_MODULE_NAME, None), Glm5NextVisionModel)
        for chunk in unwrap_model(model_chunks)
    )


def glm5_next_vlm_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage=None,
) -> GPTModel:
    """Build the validated GLM-5.3 text model and attach its visual tower."""
    from megatron.training import get_args

    from miles.backends.megatron_utils.model_provider import get_model_provider_func

    args = get_args()
    if getattr(args, "context_parallel_size", 1) != 1:
        raise ValueError("GLM-5.3 VLM training currently requires --context-parallel-size 1")
    if getattr(args, "qkv_format", "thd") != "thd":
        raise ValueError("GLM-5.3 VLM training currently requires packed --qkv-format thd")

    provider_args = copy(args)
    provider_args.custom_model_provider_path = None
    base_provider = get_model_provider_func(provider_args, role="actor")

    model = base_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
    wire_glm5_next_visual(
        model,
        args.hf_checkpoint,
        trainable=getattr(args, "glm5_vision_trainable", "none"),
        recompute_blocks=getattr(args, "glm5_vision_recompute", False),
        freeze_language=getattr(args, "glm5_freeze_language", False),
    )
    return model
