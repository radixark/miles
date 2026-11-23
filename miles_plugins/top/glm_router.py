"""GLM router through the inherited Megatron dispatch-map interface."""

import torch
from megatron.core.transformer.moe.moe_utils import router_gating_linear
from sglang.srt.true_on_policy.glm_router import route_glm

from miles_plugins.top.router import TopBiasedRouter


class TopGLMRouter(TopBiasedRouter):
    @staticmethod
    def _validate_router(config):
        if config.moe_router_score_function != "sigmoid" or not config.moe_router_enable_expert_bias:
            raise NotImplementedError("[top] GLM routing requires sigmoid with correction bias")
        if config.moe_router_topk <= 1:
            raise NotImplementedError("[top] GLM requires renormalized top-k > 1")
        for name in ("moe_router_num_groups", "moe_router_group_topk"):
            if getattr(config, name, None) not in (
                None,
                1,
            ):  # config-access-exempt: TransformerConfig field selected at runtime from a table of checked names
                raise NotImplementedError("[top] GLM routing requires a single expert group")

    def route(self, hidden_states):
        # Match the BF16 gate output in SGLang's deterministic path, even when
        # the ordinary Megatron router config requests FP32.
        logits = router_gating_linear(
            hidden_states.to(self.config.params_dtype).view(-1, 1, hidden_states.shape[-1]),
            self.weight,
            None,
            self.config.params_dtype,
        ).view(-1, self.num_experts)
        if self.expert_bias.device != hidden_states.device:
            self.expert_bias = self.expert_bias.to(hidden_states.device)
            self.local_tokens_per_expert = self.local_tokens_per_expert.to(hidden_states.device)
        with torch.no_grad():
            weights, ids = route_glm(logits.detach(), self.expert_bias, self.topk)
        if torch.is_grad_enabled() and logits.requires_grad:
            from miles_plugins.top.kernels.surrogate import attach, reference_mode

            # Surrogate: serving's weights; the gradient of renormalized sigmoid scores at its experts.
            with reference_mode():
                scores = torch.sigmoid(logits.float()).gather(1, ids.long())
                ref = scores / (scores.sum(-1, keepdim=True) + 1e-20)
            weights = attach(weights, ref, "glm_router")
        return weights, ids
