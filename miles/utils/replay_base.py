import logging
import os
from enum import Enum

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


class Replay:
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list: list[torch.Tensor] = []

    def record(self, top_indices: torch.Tensor):
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    def pop_forward(self) -> torch.Tensor:
        if self.forward_index >= len(self.top_indices_list):
            shapes = [
                t.shape if isinstance(t, torch.Tensor) else f"non-tensor({type(t)})" for t in self.top_indices_list
            ]
            raise IndexError(
                f"pop_forward out of range: forward_index={self.forward_index}, "
                f"len(top_indices_list)={len(self.top_indices_list)}, shapes={shapes}"
            )
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def pop_backward(self) -> torch.Tensor:
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []

    def clear_forward(self):
        self.forward_index = 0


class BaseReplayManager:
    name: str = ""
    filename: str = ""
    replay_check_min_overlap_ratio = 0.0  # 0.0 = mismatch only on zero overlap
    # True when replayed indices are token/KV positions (indexer): rebase the
    # per-sample 0-based rollout indices onto the packed training sequence.
    replay_indices_are_token_positions = False

    def __init__(self):
        self.replays: list[Replay] = []
        self.current: Replay | None = None
        self.enabled = False
        self.stage = "fallthrough"
        self.enable_logits_recording = False
        self.current_cache_action: RouterLogitsCacheAction | None = None
        self.logits_cache = self._create_empty_logits_cache()
        self.global_token_id_counter = 0

    def create_replay(self) -> Replay:
        replay = Replay(layer_idx=len(self.replays))
        self.replays.append(replay)
        return replay

    def set_current(self, replay: Replay):
        self.current = replay

    def get_current(self) -> Replay | None:
        return self.current

    def clear_all(self):
        for replay in self.replays:
            replay.clear()

    def clear_all_forward(self):
        for replay in self.replays:
            replay.clear_forward()

    @staticmethod
    def _create_empty_logits_cache() -> dict[str, list | dict]:
        return {
            "compute_log_prob": [],
            "training": [],
            "router_weights": {},
            "global_token_ids": [],
            "predictive_bias": [],
        }

    @staticmethod
    def _get_data_parallel_context() -> tuple[object | None, int, int]:
        if not dist.is_initialized():
            return None, 0, 1

        try:
            from megatron.core import parallel_state as mpu

            if hasattr(mpu, "get_data_parallel_group_gloo"):
                group = mpu.get_data_parallel_group_gloo(with_context_parallel=False)
            else:
                group = mpu.get_data_parallel_group(with_context_parallel=False)
            rank = mpu.get_data_parallel_rank(with_context_parallel=False)
            world_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
            return group, rank, world_size
        except Exception:
            return None, dist.get_rank(), dist.get_world_size()

    @staticmethod
    def _compute_auto_global_token_ids(
        num_tokens: int,
        base_offset: int,
        rank_counts: list[int] | None = None,
        dp_rank: int = 0,
    ) -> tuple[torch.Tensor, int]:
        if num_tokens <= 0:
            return torch.empty(0, dtype=torch.long), base_offset

        if rank_counts is None:
            rank_counts = [num_tokens]

        offset = base_offset + sum(rank_counts[:dp_rank])
        next_base_offset = base_offset + sum(rank_counts)
        return torch.arange(offset, offset + num_tokens, dtype=torch.long), next_base_offset

    def set_cache_action(self, cache_action: "RouterLogitsCacheAction") -> None:
        self.current_cache_action = cache_action
        self.enable_logits_recording = True
        self.global_token_id_counter = 0

    def clear_cache_action(self) -> None:
        self.current_cache_action = None
        self.enable_logits_recording = False
        self.global_token_id_counter = 0

    def get_and_clear_logits_cache(self) -> dict[str, list | dict]:
        cache = self.logits_cache
        self.logits_cache = self._create_empty_logits_cache()
        return cache

    def record_logits(self, logits: torch.Tensor, layer_idx: int) -> None:
        if not self.enable_logits_recording or self.current_cache_action is None:
            return

        logits_cpu = logits.detach().cpu().contiguous()
        if self.current_cache_action == RouterLogitsCacheAction.COMPUTE_LOG_PROB:
            self.logits_cache["compute_log_prob"].append((layer_idx, logits_cpu))
        elif self.current_cache_action == RouterLogitsCacheAction.TRAINING:
            self.logits_cache["training"].append((layer_idx, logits_cpu))

    def record_predictive_bias(self, delta_logits: torch.Tensor, layer_idx: int) -> None:
        if not self.enable_logits_recording or self.current_cache_action != RouterLogitsCacheAction.COMPUTE_LOG_PROB:
            return

        if delta_logits.ndim == 2:
            delta_logits = delta_logits.unsqueeze(1)
        self.logits_cache["predictive_bias"].append((layer_idx, delta_logits.detach().cpu().contiguous()))

    def record_global_token_ids(self, global_token_ids: torch.Tensor | None = None) -> None:
        if not self.enable_logits_recording or self.current_cache_action is None:
            return

        if global_token_ids is None:
            phase = self.current_cache_action.value
            phase_cache = self.logits_cache.get(phase, [])
            if not phase_cache:
                return
            _, latest_logits = phase_cache[-1]
            num_tokens = latest_logits.shape[0]
            rank_counts = None
            dp_group, dp_rank, dp_world_size = self._get_data_parallel_context()
            if dist.is_initialized() and dp_world_size > 1:
                local_count = torch.tensor([num_tokens], dtype=torch.long)
                gathered_counts = [torch.zeros_like(local_count) for _ in range(dp_world_size)]
                dist.all_gather(gathered_counts, local_count, group=dp_group)
                rank_counts = [int(count.item()) for count in gathered_counts]
            global_token_ids, self.global_token_id_counter = self._compute_auto_global_token_ids(
                num_tokens=num_tokens,
                base_offset=self.global_token_id_counter,
                rank_counts=rank_counts,
                dp_rank=dp_rank,
            )
        else:
            global_token_ids = global_token_ids.detach().cpu().contiguous().reshape(-1)
        self.logits_cache["global_token_ids"].append(global_token_ids)

    def get_topk_fn(self, old_topk_fn, return_probs, replay: Replay | None = None, layer_number: int | None = None):
        manager = self

        def _get_replay_result(top_indices, scores, topk, *args, **kwargs):
            assert (
                top_indices.shape[0] == scores.shape[0]
            ), f"rank {_get_rank()}: replay n_tokens {top_indices.shape[0]} does not match scores n_tokens {scores.shape[0]}"

            assert (
                top_indices.shape[1] == topk
            ), f"replay topk does not match expected topk, replay topk {top_indices.shape[1]}, topk {topk}"

            if self.enable_check_replay_result:
                self.check_replay_result(old_topk_fn, scores, topk, top_indices, *args, **kwargs)

            # fill padding tokens with arange to avoid invalid reading
            all_invalid = (top_indices == -1).all(dim=-1)
            if all_invalid.any():
                ar = (
                    torch.arange(top_indices.shape[1], device=top_indices.device, dtype=top_indices.dtype)
                    % scores.shape[1]
                )
                top_indices = torch.where(all_invalid.unsqueeze(-1), ar, top_indices)

            if return_probs:
                return scores.gather(1, top_indices), top_indices
            else:
                return top_indices

        def new_topk_fn(scores, topk, *args, **kwargs):
            current_replay = replay if replay is not None else manager.get_current()
            if current_replay is not None:
                manager.record_logits(scores, current_replay.layer_idx)
            elif layer_number is not None and manager.enable_logits_recording and manager.current_cache_action is not None:
                manager.record_logits(scores, layer_number - 1)

            if not manager.enabled:
                return old_topk_fn(scores, topk, *args, **kwargs)

            stage = manager.stage
            if current_replay is None:
                return old_topk_fn(scores, topk, *args, **kwargs)

            if stage == "fallthrough":
                return old_topk_fn(scores, topk, *args, **kwargs)

            elif stage == "record":
                result = old_topk_fn(scores, topk, *args, **kwargs)
                if return_probs:
                    probs, top_indices = result
                else:
                    top_indices = result
                current_replay.record(top_indices)
                return result

            elif stage == "replay_forward":
                return _get_replay_result(current_replay.pop_forward(), scores, topk, *args, **kwargs)

            elif stage == "replay_backward":
                return _get_replay_result(current_replay.pop_backward(), scores, topk, *args, **kwargs)

            else:
                return old_topk_fn(scores, topk, *args, **kwargs)

        return new_topk_fn

    def register_to_module(self, module, attr_name: str):
        replay = self.create_replay()
        setattr(module, attr_name, replay)
        manager = self

        def pre_forward_hook(*args, **kwargs):
            manager.set_current(replay)

        module.register_forward_pre_hook(pre_forward_hook)

    def check_replay_result(self, old_topk_fn, scores, topk, top_indices, *args, **kwargs):
        """
        CI checker for R3. Only enable when enable_check_replay_result=True.
        Per token, measure the overlap between the training engine's recomputed
        topk and the replayed topk (ignoring -1 padding). A token is mismatched
        when overlap < replay_check_min_overlap_ratio of its valid picks (ratio 0.0 =
        only zero overlap). Raise when the mismatched fraction exceeds
        replay_check_max_mismatch_fraction.
        """
        orig_top_indices = old_topk_fn(scores, topk, *args, **kwargs)
        if isinstance(orig_top_indices, tuple):
            _, orig_top_indices = orig_top_indices

        orig_flat = orig_top_indices.view(-1, orig_top_indices.shape[-1])  # [n_tokens, topk]
        replay_flat = top_indices.view(-1, top_indices.shape[-1])
        valid_orig = orig_flat != -1
        valid_replay = replay_flat != -1

        # token-wise set overlap via a membership mask, avoiding the O(n*topk^2)
        # all-pairs tensor; -1 padding is routed to a sentinel column
        n_kv = int(torch.maximum(orig_flat.max(), replay_flat.max()).clamp_min(0)) + 1
        membership = orig_flat.new_zeros((orig_flat.shape[0], n_kv + 1), dtype=torch.bool)
        membership.scatter_(1, torch.where(valid_orig, orig_flat, n_kv).long(), True)
        replay_idx = torch.where(valid_replay, replay_flat, n_kv).long()
        hit = membership.gather(1, replay_idx) & valid_replay  # [n_tokens, topk]

        overlap = hit.sum(dim=1)
        valid_count = valid_replay.sum(dim=1)
        is_padding = valid_count == 0
        if self.replay_check_min_overlap_ratio == 0.0:
            required = torch.ones_like(overlap)  # legacy: mismatch only on zero overlap
        else:
            required = (valid_count * self.replay_check_min_overlap_ratio).ceil().to(overlap.dtype)
        is_mismatch = (overlap < required) & ~is_padding

        mismatch_count = is_mismatch.sum().item()
        if mismatch_count == 0:
            return

        threshold = float(os.environ.get("MILES_TEST_R3_THRESHOLD", self.replay_check_max_mismatch_fraction))
        mismatch_threshold = threshold * orig_flat.shape[0]
        mismatch_indices = is_mismatch.nonzero(as_tuple=False).squeeze(1)
        for idx in mismatch_indices:
            i = idx.item()
            lines = []
            for j in range(max(0, i - 3), min(len(orig_flat), i + 4)):
                marker = " <<<" if j == i else ""
                lines.append(f"  token {j}: orig={orig_flat[j].tolist()}, replay={replay_flat[j].tolist()}{marker}")
            logger.warning(
                f"Replay check (rank {_get_rank()}, stage {self.stage}): "
                f"token {i} overlap {overlap[i].item()}/{valid_count[i].item()}, topk={topk}\n" + "\n".join(lines)
            )

        if mismatch_count > mismatch_threshold:
            raise AssertionError(f"R3 mismatch tokens ({mismatch_count}) > threshold ({mismatch_threshold:.0f})")


def apply_routing_replay_patch() -> None:
    from megatron.core.transformer.moe.moe_utils import (
        apply_router_token_dropping,
        compute_routing_scores_for_aux_loss,
        group_limited_topk,
    )
    from megatron.core.transformer.moe.router import TopKRouter

    if hasattr(TopKRouter, "_miles_routing_replay_patched"):
        return

    def patched_topk_routing_with_score_function(
        *,
        logits: torch.Tensor,
        topk: int,
        use_pre_softmax: bool,
        num_groups: int | None,
        group_topk: int | None,
        scaling_factor: float | None,
        score_function: str,
        expert_bias: torch.Tensor | None,
        fused: bool,
        router_replay: Replay | None,
        layer_number: int | None,
    ):
        del fused
        num_tokens, num_experts = logits.shape

        def _compute_topk(scores, topk, num_groups=None, group_topk=None):
            if group_topk:
                return group_limited_topk(
                    scores=scores,
                    topk=topk,
                    num_tokens=num_tokens,
                    num_experts=num_experts,
                    num_groups=num_groups,
                    group_topk=group_topk,
                )
            return torch.topk(scores, k=topk, dim=1)

        compute_topk = routing_replay_manager.get_topk_fn(
            _compute_topk,
            return_probs=True,
            replay=router_replay,
            layer_number=layer_number,
        )

        if score_function == "softmax":
            if use_pre_softmax:
                scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
                probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
            else:
                scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
                probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
        elif score_function == "sigmoid":
            scores = torch.sigmoid(logits.float()).type_as(logits)
            if expert_bias is not None:
                scores_for_routing = scores + expert_bias
                _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
                scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
            else:
                scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
            probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
        else:
            raise ValueError(f"Invalid score_function: {score_function}")

        if scaling_factor:
            probs = probs * scaling_factor

        if torch.are_deterministic_algorithms_enabled():
            routing_probs = torch.zeros_like(logits)
            rows = torch.arange(num_tokens, device=logits.device).unsqueeze(1)
            routing_probs.index_put_((rows, top_indices), probs, accumulate=False)

            routing_map = torch.zeros_like(logits, dtype=logits.dtype)
            routing_map.index_put_((rows, top_indices), torch.ones_like(probs, dtype=routing_map.dtype), accumulate=False)
            routing_map = routing_map.bool()
        else:
            routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
            routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()

        return routing_probs, routing_map

    def patched_routing(self, logits: torch.Tensor):
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)
        logits = self.apply_z_loss(logits)

        if self.routing_type == "sinkhorn":
            probs, routing_map = self.sinkhorn_load_balancing(logits)
        else:
            probs, routing_map = patched_topk_routing_with_score_function(
                logits=logits,
                topk=self.topk,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.config.moe_router_num_groups,
                group_topk=self.config.moe_router_group_topk,
                scaling_factor=self.config.moe_router_topk_scaling_factor,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
                fused=self.config.moe_router_fusion,
                router_replay=getattr(self, "routing_replay", None),
                layer_number=getattr(self, "layer_number", None),
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
                logits, self.topk, self.score_function, fused=self.config.moe_router_fusion
            )
            probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
            probs = self._apply_seq_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz)
            probs = self._apply_global_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)

        if getattr(self, "enable_expert_bias", False) and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)

        return probs, routing_map

    TopKRouter.routing = patched_routing
    TopKRouter._miles_routing_replay_patched = True
    logger.info("Applied Miles runtime routing replay patch to TopKRouter.routing")


def register_routing_replay_modules(model_chunks) -> int:
    from megatron.core.transformer.moe.router import TopKRouter

    seen_modules: set[int] = set()
    registered = 0
    for model_chunk in model_chunks:
        module = getattr(model_chunk, "module", model_chunk)
        for submodule in module.modules():
            if not isinstance(submodule, TopKRouter):
                continue
            if id(submodule) in seen_modules:
                continue
            seen_modules.add(id(submodule))
            if getattr(submodule, "routing_replay", None) is not None:
                continue
            routing_replay_manager.register_to_module(submodule, "routing_replay")
            registered += 1
    return registered


class RoutingReplayManager(BaseReplayManager):
    name = "routing"
    filename = "routing_replay.pt"
    data_key = "rollout_routed_experts"
    if_sp_region = True
    enable_check_replay_result = False
    replay_check_max_mismatch_fraction = 1e-2


class IndexerReplayManager(BaseReplayManager):
    name = "indexer"
    filename = "indexer_replay.pt"
    data_key = "rollout_indexer_topk"
    if_sp_region = False
    enable_check_replay_result = False
    replay_check_max_mismatch_fraction = 1e-2
    replay_check_min_overlap_ratio = 0.8
    replay_indices_are_token_positions = True


class RouterLogitsCacheAction(str, Enum):
    COMPUTE_LOG_PROB = "compute_log_prob"
    TRAINING = "training"


routing_replay_manager = RoutingReplayManager()
indexer_replay_manager = IndexerReplayManager()
all_replay_managers = [routing_replay_manager, indexer_replay_manager]
