import pytest

from miles.backends.training_utils.parallel import GroupInfo, ParallelState, _DPMode


def _parallel_state(
    *,
    intra_dp: GroupInfo,
    indep_dp: GroupInfo,
    intra_dp_cp: GroupInfo | None = None,
) -> ParallelState:
    if intra_dp_cp is None:
        intra_dp_cp = GroupInfo(rank=intra_dp.rank, size=intra_dp.size, group=None)
    return ParallelState(
        intra_dp=intra_dp,
        intra_dp_cp=intra_dp_cp,
        cp=GroupInfo(rank=0, size=1, group=None),
        tp=GroupInfo(rank=0, size=1, group=None),
        pp=GroupInfo(rank=0, size=1, group=None),
        ep=GroupInfo(rank=0, size=1, group=None),
        etp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=indep_dp,
    )


def test_dp_mode_is_intra_when_only_intra_dp_non_trivial():
    """Non-trivial intra_dp with trivial indep_dp selects INTRA mode."""
    state = _parallel_state(
        intra_dp=GroupInfo(rank=1, size=4, group=None),
        indep_dp=GroupInfo(rank=0, size=1, group=None),
    )
    assert state._dp_mode == _DPMode.INTRA


def test_dp_mode_is_indep_when_only_indep_dp_non_trivial():
    """Trivial intra_dp with non-trivial indep_dp selects INDEP mode."""
    state = _parallel_state(
        intra_dp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=GroupInfo(rank=2, size=3, group=None),
    )
    assert state._dp_mode == _DPMode.INDEP


def test_dp_mode_is_intra_when_both_trivial():
    """Both trivial groups fall back to INTRA mode."""
    state = _parallel_state(
        intra_dp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=GroupInfo(rank=0, size=1, group=None),
    )
    assert state._dp_mode == _DPMode.INTRA


def test_dp_mode_raises_when_both_non_trivial():
    """Both non-trivial groups violate the mutual-exclusion invariant."""
    state = _parallel_state(
        intra_dp=GroupInfo(rank=1, size=2, group=None),
        indep_dp=GroupInfo(rank=1, size=2, group=None),
    )
    with pytest.raises(AssertionError, match="cannot both be non-trivial"):
        _ = state._dp_mode


def test_effective_dp_returns_intra_dp_in_intra_mode():
    """effective_dp returns intra_dp when in INTRA mode."""
    intra_dp = GroupInfo(rank=3, size=8, group=None)
    state = _parallel_state(
        intra_dp=intra_dp,
        indep_dp=GroupInfo(rank=0, size=1, group=None),
    )
    assert state.effective_dp is intra_dp


def test_effective_dp_returns_indep_dp_in_indep_mode():
    """effective_dp returns indep_dp when in INDEP mode."""
    indep_dp = GroupInfo(rank=2, size=5, group=None)
    state = _parallel_state(
        intra_dp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=indep_dp,
    )
    assert state.effective_dp is indep_dp


def test_effective_dp_cp_uses_single_group_in_intra_mode():
    """effective_dp_cp wraps intra_dp_cp as a single group in INTRA mode."""
    intra_dp_cp = GroupInfo(rank=1, size=4, group=None)
    state = _parallel_state(
        intra_dp=GroupInfo(rank=1, size=4, group=None),
        indep_dp=GroupInfo(rank=0, size=1, group=None),
        intra_dp_cp=intra_dp_cp,
    )

    result = state.effective_dp_cp

    assert result.rank == 1
    assert result.size == 4
    assert result.groups_inner_to_outer == [None]
    assert result.gloo_groups_inner_to_outer == [None]


def test_effective_dp_cp_uses_inner_outer_pair_in_indep_mode():
    """effective_dp_cp combines intra_dp_cp (inner) and indep_dp (outer) in INDEP mode."""
    intra_dp_cp = GroupInfo(rank=1, size=4, group=None)
    indep_dp = GroupInfo(rank=2, size=3, group=None)
    state = _parallel_state(
        intra_dp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=indep_dp,
        intra_dp_cp=intra_dp_cp,
    )

    result = state.effective_dp_cp

    assert result.rank == 2 * 4 + 1
    assert result.size == 3 * 4
    assert result.groups_inner_to_outer == [None, None]
    assert result.gloo_groups_inner_to_outer == [None, None]
