"""Feed the Step5 decoder with text embeddings and frozen image features.

`pixel_values`, `patch_pixel_values`, and `num_patches` describe images; global
`image_token_positions` map crop-then-overview features into the packed sequence.
Text-only micro-batches omit these inputs.
"""

from dataclasses import dataclass

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from torch import Tensor

from miles.backends.megatron_utils.model_provider import gpt_model_kwargs
from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.models.step5 import get_step5_spec
from miles_plugins.models.step5.vision import Step5VisionConfig, Step5VisionEncoder, load_step5_vision_encoder


@dataclass(frozen=True)
class _FrozenVisionTower:
    """Keep the frozen encoder outside the language model's parameter and checkpoint tree.

    DDP, optimizer, weight sync, and model-wide casts skip it, so it is loaded on its
    GPU in BF16 and stays in evaluation mode.
    """

    encoder: Step5VisionEncoder


class Step5VLModel(GPTModel):
    """A ``GPTModel`` whose input embeddings carry image features at the ``<im_patch>`` positions."""

    def __init__(
        self,
        config: TransformerConfig,
        *,
        vision_encoder: Step5VisionEncoder,
        image_token_id: int,
        **gpt_kwargs,
    ) -> None:
        # the tower runs where tokens are embedded, and its positions index the whole packed sequence
        assert config.pipeline_model_parallel_size == 1, "Step5 VL does not support pipeline parallelism"
        assert config.context_parallel_size == 1, "Step5 VL does not support context parallelism"
        super().__init__(config=config, **gpt_kwargs)
        self._vision = _FrozenVisionTower(vision_encoder)
        self._image_token_id = image_token_id

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None,
        attention_mask: Tensor | None,
        *,
        step5_pixel_values: Tensor | None = None,
        step5_patch_pixel_values: Tensor | None = None,
        step5_num_patches: Tensor | None = None,
        step5_image_positions: Tensor | None = None,
        **gpt_forward_kwargs,
    ) -> Tensor:
        """``GPTModel.forward`` on the word embeddings, with image features at the image positions."""
        embeddings = self.embedding(input_ids=input_ids, position_ids=position_ids)
        if step5_image_positions is not None:
            features = self._vision.encoder.features(step5_pixel_values, step5_patch_pixel_values, step5_num_patches)
            embeddings = self._with_image_features(embeddings, input_ids, features, step5_image_positions)
        return super().forward(input_ids, position_ids, attention_mask, decoder_input=embeddings, **gpt_forward_kwargs)

    def _with_image_features(
        self, embeddings: Tensor, input_ids: Tensor, features: Tensor, positions: Tensor
    ) -> Tensor:
        assert input_ids.shape[0] == 1, "Step5 VL image positions index a THD-packed micro-batch"
        num_features, num_positions = features.shape[0], positions.shape[0]
        assert num_features == num_positions, f"{num_features} image feature(s) for {num_positions} image position(s)"
        on_image_tokens = bool((input_ids[0, positions] == self._image_token_id).all())
        assert on_image_tokens, f"every image position must hold the <im_patch> token {self._image_token_id}"
        # under sequence parallelism each TP rank embeds one contiguous slice of the tokens
        shard_start = self.pg_collection.tp.rank() * embeddings.shape[0] if self.config.sequence_parallel else 0
        return scatter_image_features(embeddings, features, positions, shard_start=shard_start)


def scatter_image_features(embeddings: Tensor, features: Tensor, positions: Tensor, *, shard_start: int) -> Tensor:
    """``embeddings`` ``[s, 1, h]`` with ``features`` ``[n, h]`` written at the global token ``positions`` ``[n]``.

    ``embeddings`` covers tokens ``shard_start`` to ``shard_start + s``; features that land outside
    it belong to another sequence-parallel rank. The features are cast to the embeddings' dtype
    (fp32 under ``--fp32-residual-connection``). Out of place, so gradients reach only the text rows.
    """
    local_positions = positions - shard_start
    kept = (local_positions >= 0) & (local_positions < embeddings.shape[0])
    return embeddings.index_copy(0, local_positions[kept], features[kept].to(embeddings.dtype).unsqueeze(1))


def step5_vl_model_provider(
    pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None
) -> Step5VLModel:
    """``--custom-model-provider-path`` entry: miles' default Step5 model plus the frozen vision tower.

    Mirrors the default provider for the Step5 model flags: the config from args, ``get_step5_spec``
    and the ``GPTModel`` arguments miles derives from args. Pipeline parallelism is 1, so every rank
    embeds tokens and loads the tower.
    """
    args = get_args()
    assert not args.mtp_num_layers and not args.fp8_param_gather, "Step5 VL builds neither MTP layers nor FP8 params"
    config = core_transformer_config_from_args(args)
    hf_config = load_hf_config(args.hf_checkpoint)
    vision_encoder = load_step5_vision_encoder(
        args.hf_checkpoint,
        Step5VisionConfig.from_hf(hf_config),
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    return Step5VLModel(
        vision_encoder=vision_encoder,
        image_token_id=hf_config.image_token_id,
        **gpt_model_kwargs(
            args,
            config=config,
            transformer_layer_spec=get_step5_spec(args, config, vp_stage),
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        ),
    )
