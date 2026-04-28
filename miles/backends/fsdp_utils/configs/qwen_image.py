"""QwenImage training pipeline config."""

from __future__ import annotations

import torch
from miles.utils.types import CondKwargs

from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config


@register_train_pipeline_config("Qwen/Qwen-Image")
class QwenImageTrainPipelineConfig(TrainPipelineConfig):

    lora_target_modules = [
        "to_q", "to_k", "to_v", "to_out.0",
        "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
        "img_mlp.net.0.proj", "img_mlp.net.2",
        "txt_mlp.net.0.proj", "txt_mlp.net.2",
    ]

    # Last-block text-branch outputs are discarded by parent transformer.forward.
    optimizer_state_allowed_missing = [
        "transformer_blocks.59.attn.to_add_out.",
        "transformer_blocks.59.txt_norm2.",
        "transformer_blocks.59.txt_mlp.",
    ]

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        if cond is None:
            return {}
        kwargs = {}
        if cond.encoder_hidden_states:
            enc = torch.cat(cond.encoder_hidden_states).to(device)
            # Ensure batch dimension: (seq_len, dim) → (1, seq_len, dim)
            if enc.ndim == 2:
                enc = enc.unsqueeze(0)
            kwargs["encoder_hidden_states"] = enc
        if cond.txt_seq_lens:
            kwargs["txt_seq_lens"] = cond.txt_seq_lens
        if cond.img_shapes:
            kwargs["img_shapes"] = cond.img_shapes
        return kwargs

    def cfg_combine(
        self,
        noise_pred_pos: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        guidance_scale: float,
        true_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        """CFG matching sglang-d's Qwen-Image rollout.

        Mirrors ``QwenImagePipelineConfig.postprocess_cfg_noise`` in sglang-d:
          - pick ``true_cfg_scale`` if set, else ``guidance_scale``
          - combine with ``uncond + scale * (cond - uncond)``
          - **rescale to cond_norm only when ``true_cfg_scale > 1.0``**
        """
        scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        combined = noise_pred_neg + scale * (noise_pred_pos - noise_pred_neg)
        if true_cfg_scale is not None and true_cfg_scale > 1.0:
            pos_norm = torch.norm(noise_pred_pos, dim=-1, keepdim=True)
            combined_norm = torch.norm(combined, dim=-1, keepdim=True).clamp_min(1e-12)
            combined = combined * (pos_norm / combined_norm)
        return combined
