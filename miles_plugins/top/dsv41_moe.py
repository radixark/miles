"""DeepSeek-V4.1 MoE: SGLang's fused sqrtsoftplus gate with log1p softplus, DS4's FP8 experts and slot combine.

V4.1 has no hash layers; every layer routes by score. The experts are the release's MXFP4 weights, served cast to
128x128 FP8 (lossless), so the expert program and the DeepEP slot-order combine are DeepSeek-V4's.
"""

import torch

from miles_plugins.top.dsv4_moe import TopDSV4MoELayer, TopDSV4Router
from miles_plugins.top.dsv41_runtime import BLOCK_STATE


class TopDSV41Router(TopDSV4Router):
    """FP32 gate logits on BI tiles, then SGLang's fused bias-ranked top-k with sqrtsoftplus through log1p.

    A checkpoint with a vision tower routes every row as serving's ``vision_topk`` does: the same fused gate, image
    rows (``image_token_id``) ranked by the VL bias, the sum renormalized with serving's epsilon."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.is_hash_layer:
            raise NotImplementedError("[top] DeepSeek-V4.1 has no hash-routed layers (moe_n_hash_layers must be 0)")
        from sglang.srt.true_on_policy.dsv41 import serves_images

        from miles_plugins.top.dsv4_runtime import dsv4_hf_config

        cfg = dsv4_hf_config()
        self.image_token_id = cfg.image_token_id if serves_images(cfg) else None
        # The VL bias, FP32; installed from the checkpoint (DSV41Checkpoint.install_moe) when image_token_id is set.
        self.expert_bias_vl = None

    def _route(self, input, input_ids):
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import _mm_dtype_compat

        flat = input.reshape(-1, input.shape[-1]).to(self.weight.dtype)
        logits = _mm_dtype_compat(flat, self.weight.detach().t(), torch.float32)
        if self.image_token_id is not None:
            return self._route_vision(logits, input_ids)
        from sglang.srt.layers.moe.topk import biased_topk_jit_kernel_impl

        weights, ids = biased_topk_jit_kernel_impl(
            hidden_states=flat,
            gating_output=logits,
            correction_bias=self.expert_bias,
            topk=self.topk,
            renormalize=True,
            scoring_func="sqrtsoftplus",
            num_fused_shared_experts=0,
            routed_scaling_factor=self.config.moe_router_topk_scaling_factor,
            apply_routed_scaling_factor_on_output=False,
            sqrtsoftplus_log1p=True,
        )
        return weights.float(), ids.to(torch.int32)

    def _route_vision(self, logits, input_ids):
        if input_ids is None or self.expert_bias_vl is None:
            raise RuntimeError("[top] DeepSeek-V4.1 vision routing needs input_ids and the installed VL bias")
        return vision_gate(
            logits,
            bias=self.expert_bias,
            bias_vl=self.expert_bias_vl,
            input_ids=input_ids.T.reshape(-1),
            image_token_id=self.image_token_id,
            topk=self.topk,
            routed_scaling_factor=self.config.moe_router_topk_scaling_factor,
        )


def vision_gate(logits, *, bias, bias_vl, input_ids, image_token_id, topk, routed_scaling_factor):
    """sglang.srt.multimodal.dsv41.vl_routing.vision_topk's gate call under TOP's router config (unscaled FP32
    weights, no shared slots): (weights [T, topk] FP32, expert ids [T, topk] int32)."""
    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
    from sglang.srt.layers.moe.topk import _RENORMALIZE_SUM_EPSILON

    weights, ids = moe_fused_gate(
        logits,
        bias,
        topk=topk,
        scoring_func="sqrtsoftplus",
        num_fused_shared_experts=0,
        bias_alt=bias_vl,
        input_ids=input_ids,
        bias_alt_token_id=image_token_id,
        renormalize=topk > 1,
        renormalize_epsilon=_RENORMALIZE_SUM_EPSILON,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=False,
        sqrtsoftplus_log1p=True,
    )
    return weights.float(), ids.to(torch.int32)


class TopDSV41MoELayer(TopDSV4MoELayer):
    """Hands a vision router the packed token ids, which Megatron passes down only for hash routing."""

    def forward(
        self, hidden_states, intermediate_tensors=None, padding_mask=None, input_ids=None, packed_seq_params=None
    ):
        if input_ids is None and self.router.image_token_id is not None:
            # [batch, sequence] beside [sequence, batch, hidden] states, as Megatron's MoE layer slices it.
            input_ids = BLOCK_STATE.input_ids.view(1, -1)
        return super().forward(hidden_states, intermediate_tensors, padding_mask, input_ids, packed_seq_params)
