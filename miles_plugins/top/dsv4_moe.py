"""DeepSeek-V4 MoE: SGLang's sqrtsoftplus and hash gates, clamped FP8 experts and shared expert."""

import torch
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

from miles_plugins.top.moe_ep import TopEPMoELayer
from miles_plugins.top.router import TopBiasedRouter

# DeepSeek-V4 swiglu_limit; serving refuses any other clamp under TOP.
SWIGLU_LIMIT = 10.0


class TopDSV4Router(TopBiasedRouter):
    """FP32-output batch-invariant gate GEMM, then the serving gate kernels.

    Layers 1..moe_n_hash_layers select experts from ``tid2eid[token]`` and weight
    them by renormalized sqrt(softplus(logit)); the rest use the fused
    bias-ranked sqrtsoftplus top-k. Weights stay unscaled FP32; the combine
    applies moe_router_topk_scaling_factor.
    """

    @staticmethod
    def _validate_router(config):
        if config.moe_router_score_function != "sqrtsoftplus" or not config.moe_router_enable_expert_bias:
            raise NotImplementedError("[top] DeepSeek-V4 routing requires sqrtsoftplus with correction bias")
        for name in ("moe_router_num_groups", "moe_router_group_topk"):
            if getattr(config, name, None) not in (
                None,
                1,
            ):  # config-access-exempt: TransformerConfig field selected at runtime from a table of checked names
                raise NotImplementedError("[top] DeepSeek-V4 routing is ungrouped")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_hash_layer = self.layer_number is not None and self.layer_number <= self.config.moe_n_hash_layers
        # Filled by the checkpoint installer for hash layers on every build (resume included); a constant,
        # so not checkpointed (its empty non-hash form also has no valid dist-checkpoint sharding).
        self.register_buffer("tid2eid", torch.zeros(0, self.topk, dtype=torch.int32), persistent=False)

    def _apply(self, fn):
        super()._apply(fn)
        self.tid2eid = self.tid2eid.to(torch.int32)
        return self

    def forward(self, input, padding_mask=None, input_ids=None, packed_seq_params=None):
        with torch.no_grad():
            weights, ids = self._route(input.detach(), input_ids)
        if torch.is_grad_enabled() and (input.requires_grad or self.weight.requires_grad):
            # Surrogate: serving's weights as the value, the gradient of renormalized
            # sqrt(softplus(logit)) at serving's experts. Renormalized weights sum to the
            # routed scale the kernel applied, so it is read from the kernel's output.
            from miles_plugins.top.kernels.surrogate import reference_mode

            flat = input.reshape(-1, input.shape[-1])
            with reference_mode():
                logits = flat.float() @ self.weight.float().t()
                scores = torch.nn.functional.softplus(logits).sqrt().gather(1, ids.long())
                ref = scores / scores.sum(-1, keepdim=True) * weights.sum(-1, keepdim=True)
            weights = weights + (ref - ref.detach())
        self.last_topk_ids, self.last_topk_weights = ids, weights
        probs = torch.zeros(ids.shape[0], self.num_experts, device=ids.device, dtype=weights.dtype)
        probs.scatter_(1, ids.long(), weights)
        routing_map = torch.zeros_like(probs, dtype=torch.bool)
        routing_map.scatter_(1, ids.long(), True)
        return probs, routing_map

    def _route(self, input, input_ids):
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import _mm_dtype_compat

        flat = input.reshape(-1, input.shape[-1]).to(self.weight.dtype)
        # SGLang MoEGate under deterministic inference: torch.mm(x, W.t(), out_dtype=fp32), BI-replaced.
        logits = _mm_dtype_compat(flat, self.weight.detach().t(), torch.float32)
        scale = self.config.moe_router_topk_scaling_factor
        if self.is_hash_layer:
            from sglang.kernels.ops.moe.dsv4 import hash_topk

            if input_ids is None:
                raise RuntimeError("[top] DeepSeek-V4 hash routing needs input_ids")
            weights, ids = hash_topk(
                router_logits=logits,
                input_ids=input_ids.T.reshape(-1),
                tid2eid=self.tid2eid,
                num_fused_shared_experts=0,
                routed_scaling_factor=scale,
                scoring_func="sqrtsoftplus",
            )
        else:
            from sglang.srt.layers.moe.topk import biased_topk_jit_kernel_impl

            weights, ids = biased_topk_jit_kernel_impl(
                hidden_states=flat,
                gating_output=logits,
                correction_bias=self.expert_bias,
                topk=self.topk,
                renormalize=True,
                scoring_func="sqrtsoftplus",
                num_fused_shared_experts=0,
                routed_scaling_factor=scale,
                apply_routed_scaling_factor_on_output=False,
            )
        return weights.float(), ids.to(torch.int32)


class TopDSV4MoELayer(TopEPMoELayer):
    """Slot-order DeepEP combine with routed scaling; experts clamp SwiGLU."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.experts.top_swiglu_limit = SWIGLU_LIMIT


class TopClampedSharedExpert(SharedExpertMLP):
    """The serving shared expert: FP8 gate_up, clamped SwiGLU-to-FP8, FP8 down (full K)."""

    def forward(self, hidden_states):
        from miles_plugins.top.fp8_forward import trainable_clamped_swiglu_mlp

        fc1, fc2 = self.linear_fc1, self.linear_fc2
        if not (
            getattr(fc1, "is_top_fp8", False) and getattr(fc2, "is_top_fp8", False)
        ):  # config-access-exempt: only TOP FP8 modules carry is_top_fp8
            raise NotImplementedError("[top] DeepSeek-V4 shared expert requires installed FP8 weights")
        shape = hidden_states.shape
        out = trainable_clamped_swiglu_mlp(hidden_states.reshape(-1, shape[-1]), fc1, fc2, SWIGLU_LIMIT)
        return out.reshape(shape)
