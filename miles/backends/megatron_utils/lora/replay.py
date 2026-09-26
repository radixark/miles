"""Scope captured MoE routes to one multi-LoRA forward/backward work unit."""

from contextlib import contextmanager

from miles.backends.training_utils.replay_data import fill_replay_data
from miles.utils.replay_base import routing_replay_manager


@contextmanager
def rollout_routing_replay(args, model, rollout_data, data_iterator, num_microbatches):
    manager = routing_replay_manager
    replay = manager.data_key in rollout_data
    enabled, stage = manager.enabled, manager.stage
    if replay and (not args.use_rollout_routing_replay or not enabled):
        raise ValueError("routed_experts requires --use-rollout-routing-replay at model initialization")
    if replay and args.moe_router_fusion:
        raise ValueError("rollout routing replay requires moe_router_fusion=False")
    try:
        manager.clear_all()
        # The gateway may interleave replay batches and ordinary SFT/scoring batches.
        manager.enabled = replay
        # The shared forward step temporarily selects replay_forward; checkpoint
        # recomputation runs after it restores this stage and uses its own cursor.
        manager.stage = "replay_backward"
        if replay:
            fill_replay_data(
                args=args,
                models=model,
                data_iterator=data_iterator,
                num_microbatches=num_microbatches,
                rollout_data=rollout_data,
                data_key=manager.data_key,
                replay_list=manager.replays,
                register_replay_list_func=manager.register_replay_list_func,
                if_sp_region=manager.if_sp_region,
            )
        yield
    finally:
        manager.clear_all()
        manager.enabled, manager.stage = enabled, stage
