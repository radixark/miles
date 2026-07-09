"""Megatron-Bridge weight bridge for MiniMax-M3 (text backbone).

Self-installs on import (``@register_bridge``) so ``AutoBridge`` picks it up for
HF checkpoints whose architecture is ``MiniMaxM3VLForConditionalGeneration``.
Only the **language model** is mapped (text training / RL); vision tower and
projector weights in the VL checkpoint are simply not requested when the
Megatron model is the text-only ``GPTModel`` built by ``get_minimax_m3_spec``.

Grounded in the real HF key layout (``model.safetensors.index.json``):

    language_model.model.embed_tokens.weight
    language_model.lm_head.weight
    language_model.model.norm.weight
    language_model.model.layers.{i}.input_layernorm.weight
    language_model.model.layers.{i}.post_attention_layernorm.weight
    language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    language_model.model.layers.{i}.self_attn.{q,k}_norm.weight            # per-head
    # sparse (MSA) layers only — indexer:
    language_model.model.layers.{i}.self_attn.index_{q,k}_proj.weight
    language_model.model.layers.{i}.self_attn.index_{q,k}_norm.weight
    # dense layers 0..first_dense-1:
    language_model.model.layers.{i}.mlp.{gate,up,down}_proj.weight
    # MoE layers:
    language_model.model.layers.{i}.block_sparse_moe.gate.weight
    language_model.model.layers.{i}.block_sparse_moe.e_score_correction_bias
    language_model.model.layers.{i}.block_sparse_moe.experts.{e}.{w1,w3,w2}.weight
    language_model.model.layers.{i}.block_sparse_moe.shared_experts.{gate,up,down}_proj.weight

Relation to GLM-5: GLM-5's bridge subclasses ``DeepSeekV3Bridge`` (MLA). M3 is
**GQA**, so this subclasses ``MiniMaxM2Bridge`` instead — same MiniMax MoE
lineage (sigmoid router + expert bias, ``w1/w2/w3`` experts, fp8 blockwise) —
and only changes: the ``language_model.`` prefix, per-head (not full-dim) QK
norm, a shared expert, dense layers, and the four indexer tensors.

NOTE: validated structurally against the HF key list and the M2 bridge API; the
exact Megatron param paths must be confirmed on a GPU load (``bridge.load_hf_weights``).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

try:
    import torch

    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        GatedMLPMapping,
        QKVMapping,
    )
    from megatron.bridge.models.minimax_m2.minimax_m2_bridge import MiniMaxM2Bridge

    _P = "language_model.model"  # HF text-backbone prefix

    @MegatronModelBridge.register_bridge(
        source="MiniMaxM3VLForConditionalGeneration",
        target=GPTModel,
        model_type="minimax_m3",
    )
    class MiniMaxM3Bridge(MiniMaxM2Bridge):
        """Bridge for MiniMax-M3 (GQA + MSA + MoE) text backbone.

        Also resolves the text-only ``MiniMaxM3ForCausalLM`` arch if present.
        """

        def provider_bridge(self, hf_pretrained):
            # Start from the M2 GQA-MoE provider (sigmoid router, expert bias,
            # grouped-gemm experts, rope, fp8 dequant), then apply M3 deltas.
            provider = super().provider_bridge(hf_pretrained)
            hf_config = hf_pretrained.config
            text_cfg = getattr(hf_config, "text_config", hf_config)

            # M3 uses *per-head* QK norm (M2 used full-dimension). Drop M2's custom
            # full-dim layer spec so mcore builds standard per-head TENorm, and let
            # the miles ``--spec get_minimax_m3_spec`` own the MSA attention layout.
            provider.qk_layernorm = True
            if hasattr(provider, "transformer_layer_spec"):
                provider.transformer_layer_spec = None  # provided via miles --spec

            # Shared expert (M2 had none).
            n_shared = getattr(text_cfg, "n_shared_experts", 0) or 0
            if n_shared:
                provider.moe_shared_expert_intermediate_size = (
                    n_shared * text_cfg.intermediate_size
                )
            # Router scaling.
            provider.moe_router_topk_scaling_factor = getattr(
                text_cfg, "routed_scaling_factor", 1.0
            )
            # Partial RoPE (rotary_dim / head_dim) and theta.
            rd = getattr(text_cfg, "rotary_dim", None)
            hd = getattr(text_cfg, "head_dim", None)
            if rd and hd:
                provider.rotary_percent = rd / hd
            provider.rotary_base = getattr(text_cfg, "rope_theta", provider.rotary_base)
            return provider

        def mapping_registry(self) -> MegatronMappingRegistry:
            # When the target is the composite VL model, the LM params carry a
            # ``language_model.`` prefix (and vision/projector params exist). The
            # ``--minimax-m3-vl`` flow exports ``MINIMAX_M3_VL=1`` (see
            # build_minimax_m3_vl) so a single arch-keyed bridge serves both.
            vl = os.environ.get("MINIMAX_M3_VL", "") not in ("", "0", "false")
            mp = "language_model." if vl else ""

            def M(name: str) -> str:  # megatron-side path, with optional prefix
                return f"{mp}{name}"

            m = [
                # ---- globals ----
                AutoMapping(megatron_param=M("embedding.word_embeddings.weight"),
                            hf_param=f"{_P}.embed_tokens.weight"),
                AutoMapping(megatron_param=M("output_layer.weight"),
                            hf_param="language_model.lm_head.weight"),
                AutoMapping(megatron_param=M("decoder.final_layernorm.weight"),
                            hf_param=f"{_P}.norm.weight"),
                # ---- per-layer norms ----
                AutoMapping(megatron_param=M("decoder.layers.*.self_attention.linear_qkv.layer_norm_weight"),
                            hf_param=f"{_P}.layers.*.input_layernorm.weight"),
                AutoMapping(megatron_param=M("decoder.layers.*.pre_mlp_layernorm.weight"),
                            hf_param=f"{_P}.layers.*.post_attention_layernorm.weight"),
                # ---- attention ----
                QKVMapping(megatron_param=M("decoder.layers.*.self_attention.linear_qkv.weight"),
                           q=f"{_P}.layers.*.self_attn.q_proj.weight",
                           k=f"{_P}.layers.*.self_attn.k_proj.weight",
                           v=f"{_P}.layers.*.self_attn.v_proj.weight"),
                AutoMapping(megatron_param=M("decoder.layers.*.self_attention.linear_proj.weight"),
                            hf_param=f"{_P}.layers.*.self_attn.o_proj.weight"),
                AutoMapping(megatron_param=M("decoder.layers.*.self_attention.q_layernorm.weight"),
                            hf_param=f"{_P}.layers.*.self_attn.q_norm.weight"),
                AutoMapping(megatron_param=M("decoder.layers.*.self_attention.k_layernorm.weight"),
                            hf_param=f"{_P}.layers.*.self_attn.k_norm.weight"),
                # ---- dense-layer MLP ----
                GatedMLPMapping(megatron_param=M("decoder.layers.*.mlp.linear_fc1.weight"),
                                gate=f"{_P}.layers.*.mlp.gate_proj.weight",
                                up=f"{_P}.layers.*.mlp.up_proj.weight"),
                AutoMapping(megatron_param=M("decoder.layers.*.mlp.linear_fc2.weight"),
                            hf_param=f"{_P}.layers.*.mlp.down_proj.weight"),
                # ---- MoE router + expert bias ----
                AutoMapping(megatron_param=M("decoder.layers.*.mlp.router.weight"),
                            hf_param=f"{_P}.layers.*.block_sparse_moe.gate.weight"),
                AutoMapping(megatron_param=M("decoder.layers.*.mlp.router.expert_bias"),
                            hf_param=f"{_P}.layers.*.block_sparse_moe.e_score_correction_bias"),
                # ---- routed experts (w1=gate, w3=up, w2=down) ----
                GatedMLPMapping(megatron_param=M("decoder.layers.*.mlp.experts.linear_fc1.weight*"),
                                gate=f"{_P}.layers.*.block_sparse_moe.experts.*.w1.weight",
                                up=f"{_P}.layers.*.block_sparse_moe.experts.*.w3.weight"),
                AutoMapping(megatron_param=M("decoder.layers.*.mlp.experts.linear_fc2.weight*"),
                            hf_param=f"{_P}.layers.*.block_sparse_moe.experts.*.w2.weight"),
                # ---- shared expert ----
                GatedMLPMapping(megatron_param=M("decoder.layers.*.mlp.shared_experts.linear_fc1.weight"),
                                gate=f"{_P}.layers.*.block_sparse_moe.shared_experts.gate_proj.weight",
                                up=f"{_P}.layers.*.block_sparse_moe.shared_experts.up_proj.weight"),
                AutoMapping(megatron_param=M("decoder.layers.*.mlp.shared_experts.linear_fc2.weight"),
                            hf_param=f"{_P}.layers.*.block_sparse_moe.shared_experts.down_proj.weight"),
            ]

            # ---- MSA lightning indexer (sparse layers only) ----
            for stem in ("index_q_proj", "index_k_proj", "index_q_norm", "index_k_norm"):
                m.append(AutoMapping(
                    megatron_param=M(f"decoder.layers.*.self_attention.{stem}.weight"),
                    hf_param=f"{_P}.layers.*.self_attn.{stem}.weight"))

            # ---- VL: vision tower + projector, replicated (not TP-sharded) ----
            if vl:
                try:
                    from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
                    m += [
                        ReplicatedMapping(megatron_param="vision_tower.**", hf_param="vision_tower.**"),
                        ReplicatedMapping(megatron_param="multi_modal_projector.**",
                                          hf_param="multi_modal_projector.**"),
                    ]
                except Exception:
                    # vision weights are loaded HF-native by build_minimax_m3_vl as a
                    # fallback; the bridge simply won't touch them.
                    logger.debug("ReplicatedMapping unavailable; vision loaded HF-native instead")

            return MegatronMappingRegistry(*m)

    logger.info("Registered MiniMaxM3Bridge for MiniMaxM3VLForConditionalGeneration")

except ImportError as e:
    logger.debug("megatron.bridge / minimax_m2 base not available; M3 bridge not registered: %s", e)
