"""What the torchtitan end-to-end cases actually ask for.

The cases themselves need GPUs, so the parts that can be wrong without a GPU --
a topology that does not divide, a flag that never reaches the command line --
are checked here instead. Every one of these was a real wasted run at some
point: a knob added to the harness but not passed, or a sequence length that
only failed once a long generation arrived.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

import pytest

from tests.e2e.torchtitan._common import CaseConfig, build_train_args


def _case(**overrides) -> CaseConfig:
    base = dict(
        model_repo="Qwen/Qwen3-0.6B",
        titan_model_name="qwen3",
        titan_model_flavor="0.6B",
        num_gpus=4,
        seq_len=4096,
        max_response_len=2048,
    )
    return CaseConfig(**{**base, **overrides})


def test_the_parallelism_degrees_all_reach_the_command_line():
    """Each degree is copied into torchtitan's own config field, so a missing one
    silently runs a different topology than the case claims."""
    args = build_train_args(_case(num_gpus=8, tp_size=2, pp_size=2, cp_size=2, ep_size=2), wandb_file=__file__)
    for flag, value in (
        ("tensor", 2),
        ("pipeline", 2),
        ("context", 2),
        ("expert", 2),
    ):
        assert f"--titan-{flag}-parallel-degree {value} " in args


def test_a_topology_that_does_not_divide_the_gpus_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        _case(num_gpus=4, tp_size=2, pp_size=2, cp_size=2)


def test_a_sequence_that_leaves_no_room_for_a_prompt_is_rejected():
    """torchtitan sizes its rotary tables from seq_len; a response as long as the
    whole sequence asserts inside the rope kernel mid-run."""
    with pytest.raises(ValueError, match="room for a prompt"):
        _case(seq_len=8192, max_response_len=8192)


def test_fully_async_cannot_colocate():
    with pytest.raises(ValueError, match="cannot colocate"):
        _case(fully_async=True, colocate=True)


def test_disaggregated_cases_pass_the_rollout_pool_instead_of_colocate():
    args = build_train_args(
        _case(colocate=False, rollout_num_gpus=2, fully_async=True), wandb_file=__file__
    )
    assert "--rollout-num-gpus 2 " in args
    assert "--colocate " not in args
    assert "--fully-async " in args
    # retract can deadlock flush_cache under load in fully_async.
    assert "--pause-generation-mode in_place " in args


def test_the_engine_size_follows_the_rollout_pool_not_the_trainer():
    """Colocated engines span the training GPUs; disaggregated ones span only
    their own, and sizing the engine off the trainer would ask sglang for GPUs
    it does not have."""
    assert "--rollout-num-gpus-per-engine 4 " in build_train_args(_case(), wandb_file=__file__)
    assert "--rollout-num-gpus-per-engine 2 " in build_train_args(
        _case(colocate=False, rollout_num_gpus=2), wandb_file=__file__
    )


def test_routing_replay_is_the_rollout_variant():
    """--use-routing-replay records the training engine's own routing and replays
    it; R3 replays the rollout's, which is the one that makes training faithful
    to what generated the tokens."""
    assert "--use-rollout-routing-replay " in build_train_args(_case(use_r3=True), wandb_file=__file__)
    assert "--use-rollout-routing-replay" not in build_train_args(_case(), wandb_file=__file__)


def test_the_transfer_mode_reaches_the_command_line_with_its_directories():
    """disk-delta needs two directories on top of the mode: one both sides can
    see for the published deltas, and a rollout-host-local checkpoint to patch."""
    args = build_train_args(
        _case(colocate=False, rollout_num_gpus=2, transfer_mode="disk-delta"), wandb_file=__file__
    )
    assert "--update-weight-transfer-mode disk-delta " in args
    assert "--update-weight-disk-dir " in args
    assert "--update-weight-local-checkpoint-dir " in args

    broadcast = build_train_args(
        _case(colocate=False, rollout_num_gpus=2, transfer_mode="broadcast"), wandb_file=__file__
    )
    assert "--update-weight-transfer-mode broadcast " in broadcast
    assert "--update-weight-disk-dir" not in broadcast


def test_no_transfer_mode_leaves_the_protocol_to_the_default():
    """Colocated runs go over IPC without naming a mode; naming one anyway would
    override the choice the protocol factory makes from --colocate."""
    assert "--update-weight-transfer-mode" not in build_train_args(_case(), wandb_file=__file__)


def test_layer_truncation_is_only_passed_when_asked_for():
    """Truncating layers turns a released checkpoint into a structural stand-in;
    passing it by accident would train a different model than the case names."""
    assert "--titan-num-layers 4 " in build_train_args(_case(num_layers=4), wandb_file=__file__)
    assert "--titan-num-layers" not in build_train_args(_case(), wandb_file=__file__)
