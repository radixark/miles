"""DeepSeek-V4 MoE variants used by the batch-invariant TOP path."""

from __future__ import annotations

from typing import Optional

import torch
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import (
    apply_router_token_dropping,
    compute_routing_scores_for_aux_loss,
)
from megatron.core.transformer.moe.router import TopKRouter

from miles.utils.replay_base import routing_replay_manager


def _sqrtsoftplus_routing_fp32(
    logits: torch.Tensor,
    *,
    topk: int,
    expert_bias: Optional[torch.Tensor],
    tid2eid: Optional[torch.Tensor],
    input_ids: Optional[torch.Tensor],
    is_mtp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match SGLang's DSV4 router normalization and deterministic scatter."""
    num_tokens = logits.shape[0]
    scores = torch.nn.functional.softplus(logits.float()).sqrt()

    if tid2eid is not None:
        if input_ids is None:
            raise RuntimeError("DSV4 hash routing requires input_ids")
        top_indices = tid2eid[input_ids]
        if not torch.all(top_indices >= 0):
            raise RuntimeError("DSV4 hash routing table contains uninitialized expert ids")
    else:
        if expert_bias is None:
            raise RuntimeError("DSV4 non-hash routing requires expert_bias")

        def compute_topk(
            values: torch.Tensor,
            requested_topk: int,
            _num_groups=None,
            _group_topk=None,
        ):
            return torch.topk(values, k=requested_topk, dim=1)

        if is_mtp:
            topk_fn = compute_topk
        else:
            topk_fn = routing_replay_manager.get_topk_fn(
                compute_topk,
                return_probs=True,
            )
        _, top_indices = topk_fn(scores + expert_bias, topk, None, None)

    selected_scores = torch.gather(scores, dim=1, index=top_indices)
    probs = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
    probs = probs.type_as(logits)

    routing_probs = torch.zeros_like(logits)
    rows = torch.arange(num_tokens, device=logits.device).unsqueeze(1)
    routing_probs.index_put_((rows, top_indices), probs, accumulate=False)

    routing_map = torch.zeros_like(logits, dtype=logits.dtype)
    routing_map.index_put_(
        (rows, top_indices),
        torch.ones_like(probs, dtype=routing_map.dtype),
        accumulate=False,
    )
    return routing_probs, routing_map.bool()


class DeepSeekV4TopKRouter(TopKRouter):
    """DSV4 router whose selected-weight normalization stays in FP32."""

    def routing(
        self,
        logits: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):
        if not self.config.dsv4_mode:
            return super().routing(logits, padding_mask, input_ids)
        if self.score_function != "sqrtsoftplus":
            raise RuntimeError(
                "DSV4 batch-invariant routing currently requires " f"sqrtsoftplus, got {self.score_function!r}"
            )
        if self.config.moe_router_fusion:
            raise RuntimeError("DSV4 batch-invariant routing requires the unfused router")
        if self.config.moe_router_num_groups is not None:
            raise RuntimeError("DSV4 sqrtsoftplus routing does not support groups")
        if self.config.moe_router_group_topk is not None:
            raise RuntimeError("DSV4 sqrtsoftplus routing does not support group_topk")
        if not self._routing_mode_initialized:
            raise RuntimeError("DSV4 routing mode was not initialized")

        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1)
        logits = self.apply_z_loss(logits, padding_mask=padding_mask)

        probs, routing_map = _sqrtsoftplus_routing_fp32(
            logits,
            topk=self.topk,
            expert_bias=self.expert_bias,
            tid2eid=self.tid2eid,
            input_ids=(input_ids.view(-1) if self.tid2eid is not None and input_ids is not None else None),
            is_mtp=self.is_mtp,
        )

        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(
                probs,
                routing_map,
                router_topk=self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                drop_policy=self.config.moe_token_drop_policy,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            )

        if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
            routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
                logits,
                self.topk,
                self.score_function,
                fused=False,
                padding_mask=padding_mask,
            )
            probs = self._apply_aux_loss(
                probs,
                scores_for_aux_loss,
                routing_map_for_aux_loss,
                with_padding_mask=padding_mask is not None,
            )
            probs = self._apply_seq_aux_loss(
                probs,
                scores_for_aux_loss,
                routing_map_for_aux_loss,
                seq_length,
                bsz,
                with_padding_mask=padding_mask is not None,
            )
            probs = self._apply_global_aux_loss(
                probs,
                scores_for_aux_loss,
                routing_map_for_aux_loss,
                with_padding_mask=padding_mask is not None,
            )

        self._apply_expert_bias(routing_map, padding_mask=padding_mask)
        return probs, routing_map


class DeepSeekV4TopTEGroupedMLP(TEGroupedMLP):
    """Apply DSV4 routing weights after FC2, matching SGLang's FP8 path."""

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        if self.config.moe_apply_probs_on_input:
            raise RuntimeError("DSV4 batch-invariant grouped experts require " "moe_apply_probs_on_input=False")

        output, output_bias = super().forward(
            permuted_local_hidden_states,
            tokens_per_expert,
            torch.ones_like(permuted_probs),
        )
        output_dtype = output.dtype
        output = (output * permuted_probs.unsqueeze(-1)).to(output_dtype)
        return output, output_bias


class DeepSeekV4TopMoELayer(MoELayer):
    """Apply DSV4 routed scaling after the EP/TP combine, before shared experts."""

    def postprocess(
        self,
        output: torch.Tensor,
        shared_expert_output: Optional[torch.Tensor],
    ):
        output = self.token_dispatcher.combine_postprocess(output)
        if self.config.moe_latent_size:
            output, _ = self.fc2_latent_proj(output)

        scaling_factor = self.config.moe_router_topk_scaling_factor
        if scaling_factor is None:
            raise RuntimeError("DSV4 routing requires a top-k scaling factor")
        output = (output * scaling_factor).to(output.dtype)

        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output
