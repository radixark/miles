import functools
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy, resolve_fsdp_mesh
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.base.module import Module
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import replicated_state_sharding_config
from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.flops import get_nparams_and_active_nparams, quadratic_attention_flops_per_token
from torchtitan.models.common.moe import MoE

from miles.backends.torchtitan_utils.image_embeddings import ImageInputs, image_token_counts, write_image_features
from miles.backends.torchtitan_utils.models.glm5_next.layers import (
    DSAAttention,
    Glm5NextRMSNorm,
    HyperConnection,
    KimiDeltaAttention,
    update_residual_streams,
)
from miles.backends.torchtitan_utils.models.glm5_next.packed_sequence import (
    ContextParallelLayout,
    PackedSequence,
    build_packed_sequence,
)
from miles.backends.torchtitan_utils.models.glm5_next.state_dict_adapter import Glm5NextStateDictAdapter
from miles.backends.torchtitan_utils.models.glm5_next.vision import Glm5NextImageEncoder
from miles_plugins.models.glm5_next.vision import has_visual_weights

# the image tensors forward takes; they reach every CP rank whole and stay on the host until encoded
_IMAGE_INPUT_KEYS = ("pixel_values", "image_grid_thw", "image_png", "image_png_lengths")


class Glm5NextBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hc_attn: HyperConnection.Config
        hc_ffn: HyperConnection.Config
        attention_norm: Glm5NextRMSNorm.Config
        ffn_norm: Glm5NextRMSNorm.Config
        feed_forward: DenseFeedForward.Config | MoE.Config
        kda: KimiDeltaAttention.Config | None = None
        attention: DSAAttention.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        assert (config.kda is None) != (config.attention is None)
        self.hc_attn = config.hc_attn.build()
        self.hc_ffn = config.hc_ffn.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.attn = (config.kda or config.attention).build()
        self.feed_forward = config.feed_forward.build()

    def forward(self, x_TND: torch.Tensor, sequence: PackedSequence) -> torch.Tensor:
        aggregated, h_post, h_res = self.hc_attn(x_TND)
        out = self.attn(self.attention_norm(aggregated), sequence)
        x_TND = update_residual_streams(out, x_TND, h_post, h_res)

        aggregated, h_post, h_res = self.hc_ffn(x_TND)
        out = self.feed_forward(self.ffn_norm(aggregated))
        return update_residual_streams(out, x_TND, h_post, h_res)


class Glm5NextModel(Decoder):
    state_dict_adapter_cls = Glm5NextStateDictAdapter

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        num_streams: int
        image_token_id: int | None = None

        @property
        def first_full_attention(self) -> None:
            return None

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            attention_flops = sum(
                quadratic_attention_flops_per_token(
                    num_heads=layer.attention.n_heads,
                    qk_head_dim=layer.attention.qk_head_dim,
                    v_head_dim=layer.attention.v_head_dim,
                    seq_len=seq_len,
                )
                for layer in self.layers
                if layer.attention is not None
            )
            return nparams, 6 * active_nparams + attention_flops

    def __init__(self, config: Config):
        super().__init__(config)
        self.num_streams = config.num_streams
        self.kda_conv_kernel_size = next(
            (layer.kda.q_conv1d.kernel_size for layer in config.layers if layer.kda), None
        )
        self.image_token_id = config.image_token_id

    def attach_vision_tower(self, hf_checkpoint: str, *, device: torch.device, dtype: torch.dtype) -> None:
        """Load the frozen tower on the stage that embeds tokens, if the checkpoint has one. It goes into
        ``__dict__``, not a submodule, so FSDP, the optimizer, the state-dict adapter and DCP checkpoints
        never see it."""
        if self.image_token_id is None or self.tok_embeddings is None or not has_visual_weights(hf_checkpoint):
            return
        self.__dict__["_image_encoder"] = Glm5NextImageEncoder(hf_checkpoint, device=device, dtype=dtype)

    def preprocess_inputs(self, input_dict: dict[str, Any], **kwargs: Any):
        """Decoder's preprocessing without the image tensors: every CP rank needs all images of the
        sequence (it encodes the ones its slice holds), and they carry no SPMD layout."""
        images = {key: input_dict.pop(key) for key in _IMAGE_INPUT_KEYS if key in input_dict}
        inputs, labels, model_kwargs, loss_kwargs = super().preprocess_inputs(input_dict, **kwargs)
        model_kwargs.update(images)
        return inputs, labels, model_kwargs, loss_kwargs

    def _parallelize(self, parallelism_context: ParallelismContext) -> None:
        for module in self.modules():
            if not isinstance(module, Module) or module._sharding_config is not None:
                continue
            states = [name for name, _ in module.named_parameters(recurse=False)]
            states += [name for name, _ in module.named_buffers(recurse=False)]
            if states:
                module._sharding_config = replicated_state_sharding_config(*states)
        super()._parallelize(parallelism_context)

    def _apply_fsdp(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
        fp32_policy = MixedPrecisionPolicy(
            param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False
        )
        reshard_after_forward = get_fsdp_reshard_after_forward_policy(
            parallelism.fsdp_reshard_after_forward, parallelism_context.pp_enabled
        )
        offload_policy = CPUOffloadPolicy() if training.enable_cpu_offload else OffloadPolicy()
        for block in self.layers.values():
            modules = [block.hc_attn, block.hc_ffn]
            if isinstance(block.attn, KimiDeltaAttention):
                modules.append(block.attn.gate)
            for module in modules:
                fully_shard(
                    module,
                    mesh=dp_mesh,
                    dp_mesh_dims=dp_mesh_dims,
                    mp_policy=fp32_policy,
                    offload_policy=offload_policy,
                    reshard_after_forward=reshard_after_forward,
                )
        super()._apply_fsdp(parallelism_context=parallelism_context, training=training, parallelism=parallelism)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_metadata=None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        image_png: torch.Tensor | None = None,
        image_png_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert positions is not None and positions.ndim == 1, "GLM-5.3-Flash trains on one packed sequence"
        sequence = build_packed_sequence(
            positions, spmd_mesh_group(MeshAxisName.CP), conv_kernel_size=self.kda_conv_kernel_size
        )
        if self.tok_embeddings is None:
            x_TND = tokens
        else:
            h_TD = self.tok_embeddings(tokens)
            if image_grid_thw is not None:
                images = ImageInputs(
                    grid_thw=image_grid_thw,
                    pixel_values=pixel_values,
                    image_png=image_png,
                    image_png_lengths=image_png_lengths,
                )
                h_TD = self._write_image_features(h_TD, tokens, images, sequence.cp_layout)
            x_TND = h_TD.unsqueeze(-2).expand(*h_TD.shape[:-1], self.num_streams, h_TD.shape[-1]).contiguous()
        for layer in self.layers.values():
            x_TND = layer(x_TND, sequence)
        if self.norm is None:
            return x_TND
        h_TD = self.norm(x_TND.mean(dim=-2))
        if self._skip_lm_head or self.lm_head is None:
            return h_TD
        return self.lm_head(h_TD)

    def _write_image_features(
        self, h_TD: torch.Tensor, tokens: torch.Tensor, images: ImageInputs, cp_layout: ContextParallelLayout | None
    ) -> torch.Tensor:
        encoder = self.__dict__.get("_image_encoder")
        if encoder is None:
            raise RuntimeError(
                "the batch carries images but no vision tower is attached: the checkpoint has no "
                "model.visual.* weights, or attach_vision_tower was not called"
            )
        return write_image_features(
            h_TD,
            tokens,
            image_token_id=self.image_token_id,
            tokens_per_image=image_token_counts(images.grid_thw, encoder.merge_size),
            encode=functools.partial(encoder.encode, images),
            cp_group=None if cp_layout is None else cp_layout.group,
            local_indices=None if cp_layout is None else cp_layout.local_indices,
        )
