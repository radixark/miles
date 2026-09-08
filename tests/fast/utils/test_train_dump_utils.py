"""`save_debug_train_data` must write exactly one file per (dp, cp) shard."""

from types import SimpleNamespace

import pytest
import torch

import miles.backends.training_utils.parallel as parallel
from miles.utils import train_dump_utils
from miles.utils.ft_utils.process_group_utils import GroupInfo


def _parallel_state(*, tp_rank: int, is_pp_last_stage: bool) -> parallel.ParallelState:
    trivial = GroupInfo(rank=0, size=1, group=None)
    return parallel.ParallelState(
        intra_dp=trivial,
        intra_dp_cp=trivial,
        cp=trivial,
        tp=GroupInfo(rank=tp_rank, size=2, group=None),
        pp=trivial,
        ep=trivial,
        etp=trivial,
        indep_dp=trivial,
        is_pp_last_stage=is_pp_last_stage,
    )


@pytest.mark.parametrize(
    ("tp_rank", "is_pp_last_stage", "writes"),
    [(0, True, True), (1, True, False), (0, False, False)],
    ids=["tp0_last_stage_writes", "tp_peer_skips", "earlier_pp_stage_skips"],
)
def test_only_one_rank_per_shard_writes(tmp_path, monkeypatch, tp_rank, is_pp_last_stage, writes):
    monkeypatch.setattr(
        parallel, "_parallel_state", _parallel_state(tp_rank=tp_rank, is_pp_last_stage=is_pp_last_stage)
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 5)
    args = SimpleNamespace(
        save_debug_train_data=str(tmp_path / "{rollout_id}_{rank}.pt"),
        qkv_format="thd",
    )

    train_dump_utils.save_debug_train_data(args, rollout_id=3, rollout_data={"sample_indices": [0]})

    assert (tmp_path / "3_5.pt").exists() == writes
