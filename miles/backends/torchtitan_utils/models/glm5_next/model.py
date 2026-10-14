import functools
from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.models.common.token_dispatcher import update_ep_token_dispatcher_config
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module

from miles.backends.torchtitan_utils.image_embeddings import ImageInputs, image_token_counts, write_image_features
from miles.backends.torchtitan_utils.models.glm5_next.layers import (
    ClampedFeedForward,
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
from miles.backends.torchtitan_utils.models.glm5_next.vision import Glm5NextImageEncoder
from miles_plugins.models.glm5_next.vision import has_visual_weights

_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, spmd.PerMeshAxisSpmdType] = {
    "w1_EFD": spmd.S(1),
    "w2_EDF": spmd.S(2),
    "w3_EFD": spmd.S(1),
}


class Glm5NextBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hc_attn: HyperConnection.Config
        hc_ffn: HyperConnection.Config
        attention_norm: Glm5NextRMSNorm.Config
        ffn_norm: Glm5NextRMSNorm.Config
        kda: KimiDeltaAttention.Config | None = None
        attention: DSAAttention.Config | None = None
        feed_forward: ClampedFeedForward.Config | None = None
        moe: MoE.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        assert (config.kda is None) != (config.attention is None)
        assert (config.feed_forward is None) != (config.moe is None)
        self.hc_attn = config.hc_attn.build()
        self.hc_ffn = config.hc_ffn.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.attn = (config.kda or config.attention).build()
        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            self.moe = config.moe.build()
        else:
            self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        x_BLND: torch.Tensor,
        sequence: PackedSequence,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        aggregated, h_post, h_res = self.hc_attn(x_BLND)
        out = self.attn(self.attention_norm(aggregated), sequence)
        x_BLND = update_residual_streams(out, x_BLND, h_post, h_res)

        aggregated, h_post, h_res = self.hc_ffn(x_BLND)
        ffn = self.moe if self.moe_enabled else self.feed_forward
        out = ffn(self.ffn_norm(aggregated))
        return update_residual_streams(out, x_BLND, h_post, h_res)


class Glm5NextModel(Decoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        num_streams: int
        max_position_embeddings: int
        image_token_id: int | None = None

        @property
        def max_seq_len(self) -> int:
            return self.max_position_embeddings

        def update_from_config(self, *, config, **kwargs) -> None:
            parallelism = config.parallelism
            if parallelism.tensor_parallel_degree > 1:
                raise NotImplementedError("GLM-5.3-Flash on torchtitan does not support tensor parallelism yet")
            if config.training.seq_len > self.max_position_embeddings:
                raise ValueError(
                    f"training.seq_len {config.training.seq_len} exceeds max_position_embeddings "
                    f"{self.max_position_embeddings}"
                )

            update_ep_token_dispatcher_config(self, config)
            for layer_cfg in self.layers:
                if layer_cfg.moe is None:
                    continue
                layer_cfg.moe.router._debug_force_load_balance = config.debug.moe_force_load_balance
                set_moe_sharding_config(
                    layer_cfg.moe,
                    enable_ep=parallelism.expert_parallel_degree > 1,
                    enable_sp=False,
                    expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
                )

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            attention = self.first_attention
            return get_moe_model_nparams_and_flops(
                self,
                model,
                attention.n_heads,
                attention.qk_head_dim + attention.v_head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.num_streams = config.num_streams
        self.kda_conv_kernel_size = next(
            (layer.kda.q_conv1d.kernel_size for layer in config.layers if layer.kda), None
        )
        self._cp_mesh: DeviceMesh | None = None
        self._cp_load_balancer: str | None = None
        self._cp_layouts: dict[int, ContextParallelLayout] = {}
        self.image_token_id = config.image_token_id

    def attach_vision_tower(self, hf_checkpoint: str, *, device: torch.device, dtype: torch.dtype) -> None:
        """Load the frozen tower on the stage that embeds tokens, if the checkpoint has one. It goes into
        ``__dict__``, not a submodule, so FSDP, the optimizer, the state-dict adapter and DCP checkpoints
        never see it."""
        if self.image_token_id is None or self.tok_embeddings is None or not has_visual_weights(hf_checkpoint):
            return
        self.__dict__["_image_encoder"] = Glm5NextImageEncoder(hf_checkpoint, device=device, dtype=dtype)

    def enable_context_parallel(self, mesh: DeviceMesh, *, load_balancer: str) -> None:
        self._cp_mesh = mesh
        self._cp_load_balancer = load_balancer
        self._cp_layouts = {}

    def _cp_layout(self, local_len: int, device) -> ContextParallelLayout | None:
        if self._cp_mesh is None:
            return None
        if local_len not in self._cp_layouts:
            self._cp_layouts[local_len] = ContextParallelLayout.build(
                self._cp_mesh,
                load_balancer=self._cp_load_balancer,
                seq_len=local_len * self._cp_mesh.size(),
                device=device,
            )
        return self._cp_layouts[local_len]

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks=None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        image_png: torch.Tensor | None = None,
        image_png_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert positions is not None and positions.shape[0] == 1, "GLM-5.3-Flash trains on one packed sequence"
        sequence = build_packed_sequence(
            positions,
            self._cp_layout(positions.shape[1], positions.device),
            conv_kernel_size=self.kda_conv_kernel_size,
        )
        if self.tok_embeddings is None:
            x_BLND = tokens
        else:
            h_BLD = self.tok_embeddings(tokens)
            if image_grid_thw is not None:
                images = ImageInputs(
                    grid_thw=image_grid_thw,
                    pixel_values=pixel_values,
                    image_png=image_png,
                    image_png_lengths=image_png_lengths,
                )
                h_BLD = self._write_image_features(h_BLD, tokens, images, sequence.cp_layout)
            x_BLND = h_BLD.unsqueeze(-2).expand(*h_BLD.shape[:-1], self.num_streams, h_BLD.shape[-1]).contiguous()
        for layer in self.layers.values():
            x_BLND = layer(x_BLND, sequence, positions)
        if self.norm is None:
            return x_BLND
        h_BLD = self.norm(x_BLND.mean(dim=-2))
        if self._skip_lm_head or self.lm_head is None:
            return h_BLD
        return self.lm_head(h_BLD)

    def _write_image_features(
        self, h_BLD: torch.Tensor, tokens: torch.Tensor, images: ImageInputs, cp_layout: ContextParallelLayout | None
    ) -> torch.Tensor:
        encoder = self.__dict__.get("_image_encoder")
        if encoder is None:
            raise RuntimeError(
                "the batch carries images but no vision tower is attached: the checkpoint has no "
                "model.visual.* weights, or attach_vision_tower was not called"
            )
        return write_image_features(
            h_BLD,
            tokens,
            image_token_id=self.image_token_id,
            tokens_per_image=image_token_counts(images.grid_thw, encoder.merge_size),
            encode=functools.partial(encoder.encode, images),
            cp_group=None if cp_layout is None else cp_layout.group,
            local_indices=None if cp_layout is None else cp_layout.local_indices,
            restore_indices=None if cp_layout is None else cp_layout.restore_indices,
        )
