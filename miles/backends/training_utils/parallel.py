from dataclasses import dataclass

import torch.distributed as dist


_parallel_state: "ParallelState | None" = None


def set_parallel_state(state: "ParallelState") -> None:
    global _parallel_state
    _parallel_state = state


def get_parallel_state() -> "ParallelState":
    assert _parallel_state is not None, "ParallelState not initialized. Call set_parallel_state() first."
    return _parallel_state


@dataclass
class ParallelState:
    """Core parallel state shared across all backends.
    Required by the general training utils.
    """

    intra_dp_rank: int
    intra_dp_cp_src_rank: int
    intra_dp_size: int
    cp_rank: int
    cp_size: int
    intra_dp_cp_rank: int
    intra_dp_cp_size: int
    intra_dp_group: dist.ProcessGroup | None
    intra_dp_cp_group: dist.ProcessGroup | None
    intra_dp_cp_group_gloo: dist.ProcessGroup | None
    cp_group: dist.ProcessGroup | None
    tp_size: int
    tp_rank: int
    tp_group: dist.ProcessGroup | None
    is_pp_last_stage: bool = True
    vpp_size: int | None = 1
    microbatch_group_size_per_vp_stage: int | None = None
