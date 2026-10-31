"""Native-LoRA specs for the output head (lm_head) slot of an architecture."""

from __future__ import annotations

import json
import os

import torch.nn as nn

from miles.backends.megatron_utils.lora.native.modules.linear import attach_adapter_forward
from miles.backends.megatron_utils.lora.native.modules.lm_head import LoRAOutputHead
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext


class LMHeadSpec:
    """The output head, named by the converter. A tied head's adapter is its own, so the input
    embedding stays unchanged, as in Bridge and Tinker's unembed."""

    def attach(self, model: nn.Module, args, context: AttachContext) -> int:
        if not model.post_process:
            return 0
        output_layer = model.output_layer
        tied = model.share_embeddings_and_output_weights
        weight = model.shared_embedding_or_output_weight() if tied else output_layer.weight
        hf_prefix = context.naming.output_head_prefix(weight)
        if not context.lora.selects(hf_prefix.removesuffix(".")):
            return 0
        adapter = LoRAOutputHead(
            hf_prefix=hf_prefix,
            reference=weight,
            context=context,
            vocab_local=weight.shape[0],
            unpadded_vocab_size=args.backend.vocab_size,
            tied=tied,
        )
        model.lora_lm_head_adapter = adapter
        attach_adapter_forward(output_layer, adapter, context.lora.scale)
        return 1


class InklingLMHeadSpec:
    """Inkling's lm_head projection (muP-scaled, pad-trimmed)."""

    def attach(self, model: nn.Module, args, context: AttachContext) -> int:
        if not model.post_process:
            return 0
        output_layer = model.output_layer
        mup = model.config.inkling.logits_mup_width_multiplier
        adapter = LoRAOutputHead(
            hf_prefix="language_model.lm_head.",
            reference=output_layer.weight,
            context=context,
            vocab_local=output_layer.weight.shape[0],
            mup_width_multiplier=float(mup) if mup else None,
            unpadded_vocab_size=_read_unpadded_vocab_size(args.hf_checkpoint),
        )
        model.lora_lm_head_adapter = adapter
        attach_adapter_forward(output_layer, adapter, context.lora.scale)
        return 1


def _read_unpadded_vocab_size(hf_checkpoint: str) -> int | None:
    """SGLang serves Inkling's lm_head over ``unpadded_vocab_size`` when the config declares one
    (sglang configs/inkling.py); None keeps every row."""
    with open(os.path.join(hf_checkpoint, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    return config.get("text_config", config).get("unpadded_vocab_size")
