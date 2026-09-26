from contextlib import contextmanager

from miles.backends.training_utils.replay.replay_data import fill_replay_data, register_replay_list_sequential
from miles.utils.replay_base import routing_replay_manager

FALLTHROUGH = "fallthrough"
RECORD = "record"
REPLAY_FORWARD = "replay_forward"
REPLAY_BACKWARD = "replay_backward"


def uses_rollout_replay(args) -> bool:
    return args.use_rollout_routing_replay


def enable(args) -> bool:
    routing_replay_manager.enabled = args.use_routing_replay
    routing_replay_manager.enable_check_replay_result = routing_replay_manager.enabled and args.ci_test
    routing_replay_manager.register_replay_list_func = register_replay_list_sequential
    return routing_replay_manager.enabled


def fill(args, models, data_iterators, num_microbatches, rollout_data, align=None) -> None:
    if not uses_rollout_replay(args):
        return

    fill_replay_data(
        args=args,
        models=models,
        data_iterator=data_iterators,
        num_microbatches=num_microbatches,
        rollout_data=rollout_data,
        data_key=routing_replay_manager.data_key,
        replay_list=routing_replay_manager.replays,
        register_replay_list_func=routing_replay_manager.register_replay_list_func,
        if_sp_region=routing_replay_manager.if_sp_region,
        indices_are_token_positions=routing_replay_manager.replay_indices_are_token_positions,
    )

    if align is None:
        return
    for replay in routing_replay_manager.replays:
        for i, entry in enumerate(replay.top_indices_list):
            replay.top_indices_list[i] = align(entry, -1)


def log_prob_stage(args) -> str:
    if not routing_replay_manager.enabled:
        return FALLTHROUGH
    return REPLAY_FORWARD if uses_rollout_replay(args) else RECORD


@contextmanager
def stage(name: str):
    previous = routing_replay_manager.stage
    routing_replay_manager.stage = name
    try:
        yield
    finally:
        routing_replay_manager.stage = previous


def rewind() -> None:
    routing_replay_manager.clear_all_forward()


def reset() -> None:
    routing_replay_manager.clear_all()
