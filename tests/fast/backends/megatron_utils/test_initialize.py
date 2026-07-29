from miles.backends.megatron_utils import initialize
from miles.backends.training_utils.parallel import GroupInfo, ParallelState


def _parallel_state(
    *,
    tp_rank: int = 0,
    pp_rank: int = 0,
    pp_size: int = 1,
    intra_dp_cp_rank: int = 0,
    intra_dp_cp_size: int = 1,
    indep_dp_rank: int = 0,
    indep_dp_size: int = 1,
) -> ParallelState:
    intra_dp = GroupInfo(rank=0, size=1, group=None)
    indep_dp = GroupInfo(rank=indep_dp_rank, size=indep_dp_size, group=None)
    return ParallelState(
        intra_dp=intra_dp,
        intra_dp_cp=GroupInfo(rank=intra_dp_cp_rank, size=intra_dp_cp_size, group=None),
        cp=GroupInfo(rank=0, size=1, group=None),
        tp=GroupInfo(rank=tp_rank, size=max(tp_rank + 1, 1), group=None),
        pp=GroupInfo(rank=pp_rank, size=pp_size, group=None),
        ep=GroupInfo(rank=0, size=1, group=None),
        etp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=indep_dp,
    )


def _patch_state(monkeypatch, state: ParallelState) -> None:
    monkeypatch.setattr(initialize, "get_parallel_state", lambda: state)


def test_intra_mode_first_inner_rank_is_main_rank(monkeypatch):
    """INTRA mode: last pp stage, tp rank 0, intra_dp_cp rank 0 is the main rank."""
    _patch_state(monkeypatch, _parallel_state(pp_rank=1, pp_size=2, intra_dp_cp_rank=0, intra_dp_cp_size=2))
    assert initialize.is_first_replica_megatron_main_rank() is True


def test_intra_mode_non_first_inner_rank_is_not_main_rank(monkeypatch):
    """INTRA mode: a non-zero intra_dp_cp rank is not the main rank."""
    _patch_state(monkeypatch, _parallel_state(pp_rank=1, pp_size=2, intra_dp_cp_rank=1, intra_dp_cp_size=2))
    assert initialize.is_first_replica_megatron_main_rank() is False


def test_indep_mode_first_replica_first_inner_is_main_rank(monkeypatch):
    """INDEP mode: first replica plus first inner rank yields the main rank."""
    _patch_state(
        monkeypatch,
        _parallel_state(
            pp_rank=1, pp_size=2, intra_dp_cp_rank=0, intra_dp_cp_size=2, indep_dp_rank=0, indep_dp_size=2
        ),
    )
    assert initialize.is_first_replica_megatron_main_rank() is True


def test_indep_mode_second_replica_main_rank_is_not_first_replica(monkeypatch):
    """INDEP regression: second replica's main rank is not the first-replica main rank."""
    _patch_state(
        monkeypatch,
        _parallel_state(
            pp_rank=1, pp_size=2, intra_dp_cp_rank=0, intra_dp_cp_size=2, indep_dp_rank=1, indep_dp_size=2
        ),
    )
    assert initialize.is_first_replica_megatron_main_rank() is False


def test_non_zero_tp_rank_is_not_main_rank(monkeypatch):
    """A non-zero tp rank is never the main rank even on the first replica's first inner rank."""
    _patch_state(monkeypatch, _parallel_state(tp_rank=1, pp_rank=1, pp_size=2))
    assert initialize.is_first_replica_megatron_main_rank() is False


def test_non_last_pp_stage_is_not_main_rank(monkeypatch):
    """A pp rank that is not the last pipeline stage is never the main rank."""
    _patch_state(monkeypatch, _parallel_state(tp_rank=0, pp_rank=0, pp_size=2))
    assert initialize.is_first_replica_megatron_main_rank() is False
