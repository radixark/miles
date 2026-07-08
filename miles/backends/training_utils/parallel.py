from dataclasses import dataclass
from enum import StrEnum, auto


from miles.utils.ft_utils.process_group_utils import GroupInfo, GroupsInfo


_parallel_state: "ParallelState | None" = None


def set_parallel_state(state: "ParallelState") -> None:
    global _parallel_state
    _parallel_state = state


def get_parallel_state() -> "ParallelState":
    assert _parallel_state is not None, "ParallelState not initialized. Call set_parallel_state() first."
    return _parallel_state


class _DPMode(StrEnum):
    INTRA = auto()
    INDEP = auto()


@dataclass
class ParallelState:
    """Core parallel state shared across all backends.
    Required by the general training utils.
    """

    intra_dp: GroupInfo
    intra_dp_cp: GroupInfo
    cp: GroupInfo
    tp: GroupInfo
    pp: GroupInfo
    ep: GroupInfo
    etp: GroupInfo
    indep_dp: GroupInfo
    cp_comm_type: str | list[str] | tuple[str, ...] | None = None
    is_pp_last_stage: bool = True
    vpp_size: int | None = 1
    microbatch_group_size_per_vp_stage: int | None = None

    @property
    def _dp_mode(self) -> _DPMode:
        intra_trivial = self.intra_dp.rank == 0 and self.intra_dp.size == 1
        indep_trivial = self.indep_dp.rank == 0 and self.indep_dp.size == 1
        assert intra_trivial or indep_trivial, "intra_dp and indep_dp cannot both be non-trivial"

        return _DPMode.INTRA if indep_trivial else _DPMode.INDEP

    @property
    def effective_dp(self) -> GroupInfo:
        return {_DPMode.INTRA: self.intra_dp, _DPMode.INDEP: self.indep_dp}[self._dp_mode]

    @property
    def effective_dp_cp(self) -> GroupsInfo:
        return {
            _DPMode.INTRA: GroupsInfo.from_single(self.intra_dp_cp),
            _DPMode.INDEP: GroupsInfo.from_pair(inner=self.intra_dp_cp, outer=self.indep_dp),
        }[self._dp_mode]

    @property
    def is_ulysses_cp(self) -> bool:
        cp_comm_type = self.cp_comm_type
        if isinstance(cp_comm_type, (list, tuple)):
            cp_comm_type = cp_comm_type[0] if cp_comm_type else None
        return self.cp.size > 1 and cp_comm_type == "a2a"
