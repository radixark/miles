"""Qwen softmax routing with ordered slots and Megatron's dense dispatch map."""

from __future__ import annotations

import torch
from sglang.srt.true_on_policy.moe import route_qwen

from miles.utils.replay_base import routing_replay_manager


class TopTopKRouter(torch.nn.Module):
    def __init__(self, *, config, pg_collection=None, is_mtp_layer=False, layer_number=None):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.topk = config.moe_router_topk
        self.num_experts = config.num_moe_experts
        self._validate_router(config)
        self.weight = torch.nn.Parameter(torch.empty(self.num_experts, config.hidden_size, dtype=torch.float32))
        if config.perform_initialization:
            config.init_method(self.weight)
        self.weight.data = self.weight.data.to(config.params_dtype)
        self.weight.sequence_parallel = config.sequence_parallel
        self.last_topk_ids = None
        self.last_topk_weights = None
        self.routing_replay = None
        # Rollout routing replay (R3) streams follow the main-decoder MoE layers, as Megatron's TopKRouter registers;
        # an MTP router would pop from an empty stream.
        if not is_mtp_layer:
            routing_replay_manager.register_to_module(self, "routing_replay")

    @staticmethod
    def _validate_router(config):
        if config.moe_router_score_function != "softmax" or config.moe_router_enable_expert_bias:
            raise NotImplementedError("[top] Qwen routing requires softmax without expert bias")

    def route(self, hidden_states):
        return route_qwen(hidden_states, self.weight, self.topk)

    def _route_replayed(self, hidden_states):
        """Route with the rollout's top-k when the replay manager replays this call, else live."""
        replayed_ids = (
            None
            if self.routing_replay is None
            else routing_replay_manager.get_replay_indices(hidden_states.shape[0], self.topk)
        )
        if replayed_ids is None:
            return self.route(hidden_states)
        if routing_replay_manager.enable_check_replay_result:
            with torch.no_grad():
                routing_replay_manager.check_replay_indices(self.route(hidden_states)[1], replayed_ids)
        return route_qwen(hidden_states, self.weight, self.topk, replayed_ids)

    def set_layer_number(self, layer_number):
        self.layer_number = layer_number

    def forward(self, input, *args, **kwargs):
        if self.weight.device.type == "cpu":
            self.weight.data = self.weight.data.to(input.device)
        flat = input.reshape(-1, input.shape[-1])
        weights, ids = self._route_replayed(flat)
        # The dense dispatch map loses slot order; retain the ordered router result for combine.
        self.last_topk_ids, self.last_topk_weights = ids, weights
        probs = torch.zeros(flat.shape[0], self.num_experts, device=flat.device, dtype=weights.dtype)
        probs.scatter_(1, ids.long(), weights)
        routing_map = torch.zeros_like(probs, dtype=torch.bool)
        routing_map.scatter_(1, ids.long(), True)
        return probs, routing_map


class TopBiasedRouter(TopTopKRouter):
    """A router with the checkpoint's FP32 correction bias (GLM-5, DeepSeek-V4)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.routing_replay is not None:
            raise NotImplementedError(
                "[top] rollout routing replay covers only the Qwen softmax router; drop --use-rollout-routing-replay"
            )
        self.register_buffer("expert_bias", torch.zeros(self.num_experts, dtype=torch.float32))
        # Serving routes with the checkpoint's bias; Megatron's aux-free bias update must not move it.
        self.frozen_expert_bias = True
        # Megatron zeroes this after every step for any router with an expert bias (frozen or not).
        self.register_buffer(
            "local_tokens_per_expert", torch.zeros(self.num_experts, dtype=torch.float32), persistent=False
        )

    def _apply(self, fn):
        super()._apply(fn)
        self.expert_bias = self.expert_bias.float()
        self.local_tokens_per_expert = self.local_tokens_per_expert.float()
        return self
