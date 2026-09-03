"""Native-LoRA specs for the output head (lm_head) slot of an architecture."""

from __future__ import annotations

import json
import os

import torch.nn as nn

from miles_plugins.lora.modules.linear import attach_adapter_forward
from miles_plugins.lora.modules.moe import LoRAOutputHead
from miles_plugins.lora.spec.base import AttachContext


class InklingLMHeadSpec:
    """Inkling's lm_head projection (muP-scaled, pad-trimmed)."""

    def attach(self, model: nn.Module, args, context: AttachContext) -> int:
        if not getattr(model, "post_process", False) or getattr(model, "output_layer", None) is None:
            return 0
        output_layer = model.output_layer
        mup = getattr(model.config.inkling, "logits_mup_width_multiplier", None)
        adapter = LoRAOutputHead(
            hf_prefix="language_model.lm_head.",
            reference=output_layer.weight,
            context=context,
            vocab_local=output_layer.weight.shape[0],
            mup_width_multiplier=float(mup) if mup else None,
            unpadded_vocab_size=_unpadded_vocab_size(getattr(args, "hf_checkpoint", None)),
        )
        model.lora_lm_head_adapter = adapter
        attach_adapter_forward(output_layer, adapter, context.scale)
        return 1


def _unpadded_vocab_size(hf_checkpoint: str | None) -> int | None:
    """True (unpadded) vocab size from the HF config, or None if absent."""
    if not hf_checkpoint:
        return None
    try:
        with open(os.path.join(hf_checkpoint, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        return (config.get("text_config") or config).get("unpadded_vocab_size")
    except Exception:
        return None
