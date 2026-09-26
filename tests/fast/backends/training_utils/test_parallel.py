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


class TestTrainParallelConfig:
    def test_live_cell_count_and_pipeline_layout_are_derived_after_reconfiguration(self) -> None:
        """The next config reflects three live cells and the current CP and VPP layout."""
        state = _parallel_state(
            intra_dp=GroupInfo(rank=0, size=1, group=None),
            indep_dp=GroupInfo(rank=2, size=4, group=None),
        )
        state.cp = GroupInfo(rank=1, size=2, group=None)
        state.vpp_size = 2
        state.microbatch_group_size_per_vp_stage = 4

        before = state.train_parallel_config(supports_precomputed_schedule=True)
        state.indep_dp = GroupInfo(rank=2, size=3, group=None)
        after = state.train_parallel_config(supports_precomputed_schedule=True)

        assert before.dp_size == 4
        assert after.dp_size == 3
        assert after.cp_size == 2
        assert after.vpp_size == 2
        assert after.microbatch_group_size_per_vp_stage == 4
        assert after.independent_dp
        assert after.supports_precomputed_schedule

    def test_backend_capability_does_not_change_the_live_topology(self) -> None:
        """A backend without precomputed scheduling retains the current DP and CP sizes."""
        state = _parallel_state(
            intra_dp=GroupInfo(rank=1, size=4, group=None),
            indep_dp=GroupInfo(rank=0, size=1, group=None),
        )
        state.cp = GroupInfo(rank=1, size=2, group=None)

        config = state.train_parallel_config(supports_precomputed_schedule=False)

        assert config.dp_size == 4
        assert config.cp_size == 2
        assert not config.independent_dp
        assert not config.supports_precomputed_schedule
