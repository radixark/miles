"""The TMax launcher's GPU layout with SGLang OPD teacher servers (run_tmax_qwen3_5.py)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples" / "tmax"))

import run_tmax_qwen3_5 as launcher  # noqa: E402


def _args(**kwargs):
    return launcher.ScriptArgs(model_name="Qwen3.5-4B-tmax-template", opd_teacher="tmax-9b", **kwargs)


def test_teachers_take_rollout_gpus_and_training_keeps_whole_nodes():
    args = _args(num_nodes=4, async_rollout_gpus=14, opd_teacher_gpus=2)
    assert args.opd_mode == "sglang", "a 9B teacher of a 4B student is served by SGLang"
    assert args.train_layout == (2, 8)


def test_teacher_and_rollout_gpus_must_leave_training_whole_nodes():
    with pytest.raises(AssertionError, match="whole nodes"):
        _ = _args(num_nodes=4, async_rollout_gpus=16, opd_teacher_gpus=2).train_layout


def test_on_one_node_training_gets_what_rollout_and_the_teachers_leave():
    assert _args(num_gpus_per_node=4, async_rollout_gpus=1, opd_teacher_gpus=2).train_layout == (1, 1)
    assert _args(num_gpus_per_node=8, opd_teacher_gpus=1).train_layout == (1, 7), "colocated"


def test_colocated_training_with_an_sglang_teacher_is_one_node_only():
    with pytest.raises(AssertionError, match="fully async"):
        _args(num_nodes=2, opd_teacher_gpus=1)


def test_teacher_servers_go_round_robin_over_the_rollout_nodes_last_first():
    assert launcher._opd_teacher_hosts(num_nodes=4, train_nodes=2, num_servers=2) == [3, 2]
    assert launcher._opd_teacher_hosts(num_nodes=4, train_nodes=2, num_servers=4) == [3, 2, 3, 2]
    assert launcher._opd_teacher_hosts(num_nodes=1, train_nodes=1, num_servers=2) == [0, 0]


def test_a_same_architecture_teacher_takes_no_gpu():
    args = launcher.ScriptArgs(opd_teacher="tmax-9b", num_nodes=4, async_rollout_gpus=16)
    assert args.opd_mode == "megatron" and args.teacher_server_gpus == 0 and args.train_layout == (2, 8)
