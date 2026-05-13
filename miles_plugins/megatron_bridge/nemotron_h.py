"""Miles integration for ``megatron.bridge`` Nemotron-H (Mamba+Attention hybrid).

This plugin is a non-invasive drop-in that:

1. Registers :class:`MilesNemotronHBridge` for ``NemotronHForCausalLM`` (overriding
   the upstream :class:`NemotronHBridge`) so MoE variants (Nano-30B-A3B,
   Super-120B-A12B) round-trip correctly through the HF↔Megatron
   ``mapping_registry`` and get the right ``moe_router_*`` fields on the
   provider.
2. Patches :class:`~megatron.core.transformer.transformer_layer.TransformerLayer`
   so hybrid layers whose ``self_attention`` / ``mlp`` is an ``IdentityOp``
   don't blow up on the 2-tuple unpack in ``_forward_attention`` /
   ``_forward_mlp``.
3. Patches :class:`~megatron.core.models.mamba.MambaModel` ``forward`` to
   transparently swallow the ``loss_mask`` kwarg so miles' generic training
   loop (which was designed around ``GPTModel``) can keep passing it
   unconditionally.

Importing this module is idempotent — safe to import multiple times.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


_NEMOTRONH_MOE_MAPPINGS: dict[str, str] = {
    "decoder.layers.*.mlp.router.weight": "backbone.layers.*.mixer.gate.weight",
    "decoder.layers.*.mlp.router.expert_bias": "backbone.layers.*.mixer.gate.e_score_correction_bias",
    # Routed experts: up-only FFN (nemotron_h uses squared_relu, no gate).
    "decoder.layers.*.mlp.experts.linear_fc1.weight*": "backbone.layers.*.mixer.experts.*.up_proj.weight",
    "decoder.layers.*.mlp.experts.linear_fc2.weight*": "backbone.layers.*.mixer.experts.*.down_proj.weight",
    "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight": "backbone.layers.*.mixer.experts.*.up_proj.weight",
    "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight": "backbone.layers.*.mixer.experts.*.down_proj.weight",
    "decoder.layers.*.mlp.shared_experts.linear_fc1.weight": "backbone.layers.*.mixer.shared_experts.up_proj.weight",
    "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "backbone.layers.*.mixer.shared_experts.down_proj.weight",
}

# HF name → provider name → cast. Miles' generic ``model_provider.py`` does not
# forward ``moe_router_topk_scaling_factor`` by default, which silently runs
# Megatron with scaling=1.0 instead of the HF config's (e.g.) 2.5 — producing
# ~0.28 train-vs-rollout logprob drift for Nano-30B-A3B.
_NEMOTRONH_MOE_ROUTING_FIELDS: tuple[tuple[str, str, type], ...] = (
    ("routed_scaling_factor", "moe_router_topk_scaling_factor", float),
    ("n_group", "moe_router_num_groups", int),
    ("topk_group", "moe_router_group_topk", int),
)


def _build_bridge_subclass():
    """Build the ``MilesNemotronHBridge`` class lazily so importing this module
    does not force megatron.bridge.models.nemotronh to load until first use.
    """
    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.bridge.models.conversion.param_mapping import AutoMapping
    from megatron.bridge.models.nemotronh.nemotron_h_bridge import NemotronHBridge
    from megatron.core.models.mamba import MambaModel

    @MegatronModelBridge.register_bridge(source="NemotronHForCausalLM", target=MambaModel)
    class MilesNemotronHBridge(NemotronHBridge):
        """Nemotron-H bridge with MoE support on top of upstream dense bridge.

        Upstream :class:`NemotronHBridge` only handles the dense Mamba+Attention
        variants. The Nano-30B-A3B / Super-120B-A12B MoE variants add router,
        routed-expert, and shared-expert tensors under ``backbone.layers.*.mixer.*``
        and need ``num_moe_experts`` / ``moe_router_*`` on the provider.
        """

        def provider_bridge(self, hf_pretrained):
            provider = super().provider_bridge(hf_pretrained)
            hf = hf_pretrained.config

            n_exp = int(getattr(hf, "num_experts", None) or getattr(hf, "n_routed_experts", None) or 0)
            if n_exp == 0:
                return provider

            provider.num_moe_experts = n_exp
            provider.moe_router_topk = int(getattr(hf, "num_experts_per_tok", 1))
            provider.moe_router_score_function = "sigmoid"
            provider.moe_router_enable_expert_bias = True
            provider.moe_router_dtype = "fp32"
            provider.moe_grouped_gemm = True
            provider.moe_ffn_hidden_size = int(getattr(hf, "moe_intermediate_size", None) or provider.ffn_hidden_size)

            # hybrid_override_pattern ('MEMEM*...') marks which layers are MoE.
            # Mirror to moe_layer_freq so miles' replay_utils registers the
            # rollout routing replay once per real MoE layer, not per transformer layer.
            pattern = getattr(provider, "hybrid_override_pattern", None) or getattr(
                hf, "hybrid_override_pattern", None
            )
            if pattern:
                provider.moe_layer_freq = [1 if ch == "E" else 0 for ch in pattern][: int(provider.num_layers)]

            shared_size = getattr(hf, "moe_shared_expert_intermediate_size", None)
            if shared_size is not None:
                provider.moe_shared_expert_intermediate_size = int(shared_size)
            elif getattr(hf, "n_shared_experts", 0):
                provider.moe_shared_expert_intermediate_size = provider.moe_ffn_hidden_size * int(hf.n_shared_experts)

            for hf_name, prov_name, cast in _NEMOTRONH_MOE_ROUTING_FIELDS:
                val = getattr(hf, hf_name, None)
                if val is not None:
                    setattr(provider, prov_name, cast(val))
            return provider

        def mapping_registry(self):
            # Append MoE mappings unconditionally. Dense variants do not carry
            # the extra megatron params so these mappings are simply unreferenced.
            registry = super().mapping_registry()
            base = list(registry.mappings if hasattr(registry, "mappings") else registry._mappings)
            extras = [AutoMapping(megatron_param=m, hf_param=h) for m, h in _NEMOTRONH_MOE_MAPPINGS.items()]
            return MegatronMappingRegistry(*base, *extras)

    return MilesNemotronHBridge


def _install_nemotronh_hybrid_layer_shims() -> None:
    """Make ``TransformerLayer`` tolerate ``IdentityOp`` self_attention / mlp.

    nemotron_h is a Mamba+Attention hybrid: non-attention positions use
    ``IdentityOp`` for ``self_attention``; attention-only positions use
    ``IdentityOp`` for ``self.mlp``. Both ``TransformerLayer._forward_attention``
    and ``_forward_mlp`` do a 2-tuple unpack that raises ``ValueError`` on the
    raw tensor ``IdentityOp`` returns. Short-circuit both paths.
    """
    from megatron.core.transformer.transformer_layer import TransformerLayer

    if getattr(TransformerLayer, "_miles_nemotron_hybrid_shim_installed", False):
        return

    _orig_fwd_attn = TransformerLayer._forward_attention
    _orig_fwd_mlp = TransformerLayer._forward_mlp

    def _forward_attention(self, *args, **kwargs):
        if type(self.self_attention).__name__ == "IdentityOp":
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            return hidden_states, None
        return _orig_fwd_attn(self, *args, **kwargs)

    def _forward_mlp(self, hidden_states, *args, **kwargs):
        if type(self.mlp).__name__ == "IdentityOp":
            return hidden_states
        return _orig_fwd_mlp(self, hidden_states, *args, **kwargs)

    TransformerLayer._forward_attention = _forward_attention
    TransformerLayer._forward_mlp = _forward_mlp
    TransformerLayer._miles_nemotron_hybrid_shim_installed = True


def _install_mamba_model_loss_mask_shim() -> None:
    """Make ``MambaModel.forward`` silently accept (and drop) ``loss_mask``.

    Miles' generic training loop was written against ``GPTModel.forward``,
    which has ``loss_mask: Optional[Tensor] = None`` as a keyword-only arg.
    ``MambaModel.forward`` does not accept ``loss_mask``, so passing it raises
    ``TypeError``. The loss itself is still computed downstream from the batch
    ``loss_masks`` field — it is only the forward call that must drop it.
    """
    from megatron.core.models.mamba import MambaModel

    if getattr(MambaModel, "_miles_loss_mask_shim_installed", False):
        return

    _orig_forward = MambaModel.forward

    def forward(self, *args, loss_mask=None, **kwargs):
        return _orig_forward(self, *args, **kwargs)

    MambaModel.forward = forward
    MambaModel._miles_loss_mask_shim_installed = True


def install() -> None:
    """Apply all Nemotron-H shims and register the miles bridge subclass."""
    for fn in (
        _install_nemotronh_hybrid_layer_shims,
        _install_mamba_model_loss_mask_shim,
        _build_bridge_subclass,
    ):
        try:
            fn()
        except Exception as e:  # best-effort; avoid breaking unrelated models
            logger.warning("miles nemotron_h shim %s not applied: %s", fn.__name__, e)


install()
