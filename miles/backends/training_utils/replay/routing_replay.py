import logging
from contextlib import contextmanager

import torch
import torch.distributed as dist

from miles import platforms
from miles.backends.training_utils.replay.data import fill_replay_data, register_replay_list_sequential
from miles.utils.replay_base import routing_replay_manager

logger = logging.getLogger(__name__)

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
    # a missing key gets fill_replay_data's error
    if platforms.IS_TPU and routing_replay_manager.data_key in rollout_data:
        check_tpu_rollout_routed_experts(
            rollout_data[routing_replay_manager.data_key], response_lengths=rollout_data["response_lengths"]
        )

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
        kept_in_train_dump=routing_replay_manager.kept_in_train_dump,
    )

    if align is None:
        return
    for replay in routing_replay_manager.replays:
        for i, entry in enumerate(replay.top_indices_list):
            replay.top_indices_list[i] = align(entry, -1)


def check_tpu_rollout_routed_experts(routed_experts: list[torch.Tensor], *, response_lengths: list[int]) -> None:
    """Refuse ids from a TPU engine that captures in compiled prefill, or that captured nothing it decoded.

    sglang-torchtpu#1580 reports every prefill row as -1; `[num_tokens - 1, num_layers, topk]` per sample, of
    which the last `response_length - 1` rows were decoded. Checks this rank's shard only.
    """
    for index, rows in enumerate(routed_experts):
        if (rows[:1] >= 0).any():
            raise RuntimeError(
                f"sample {index}: the engine returned routed experts for a prefill row, which on TPU means it "
                "captures inside compiled prefill programs and corrupts them; use an sglang-torchtpu with "
                "sgl-project/sglang-torchtpu#1580"
            )
    rows_without_ids = sum(int((rows < 0).any(dim=-1).any(dim=-1).sum()) for rows in routed_experts)
    total_rows = sum(rows.shape[0] for rows in routed_experts)
    decoded_rows = sum(max(length - 1, 0) for length in response_lengths)
    if decoded_rows and rows_without_ids == total_rows:
        raise RuntimeError(
            f"the engine returned no routed experts for any of {decoded_rows} decoded rows; "
            "is --enable-return-routed-experts set?"
        )
    if dist.get_rank() == 0:
        logger.info(
            f"routing replay (rank 0): {rows_without_ids}/{total_rows} rows carry no engine ids and take the "
            "router's top-k"
        )


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
