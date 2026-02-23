from dataclasses import dataclass
from enum import Enum
import torch.distributed as dist


class CPSlicing(Enum):
    """Context Parallelism data slicing strategy.

    ZIGZAG: Each rank gets two non-contiguous chunks.

    CONTIGUOUS: Each rank gets one contiguous chunk.
    """

    ZIGZAG = "zigzag"
    CONTIGUOUS = "contiguous"


@dataclass
class ParallelState:
    """Core parallel state shared across all backends.
    Required by the general training utils.
    """

    dp_rank: int
    dp_src_rank: int
    dp_size: int
    cp_rank: int
    cp_size: int
    dp_cp_rank: int
    dp_cp_size: int
    dp_group: dist.ProcessGroup | None
    dp_cp_group: dist.ProcessGroup | None
    dp_cp_group_gloo: dist.ProcessGroup | None
    cp_group: dist.ProcessGroup | None
    tp_size: int
    tp_rank: int
    tp_group: dist.ProcessGroup | None
    is_pp_last_stage: bool = True
    vpp_size: int | None = 1
    microbatch_group_size_per_vp_stage: int | None = None

    cp_slicing: CPSlicing | None = None

    @property
    def uses_zigzag_cp(self) -> bool:
        return self.cp_slicing is CPSlicing.ZIGZAG

    @property
    def uses_contiguous_cp(self) -> bool:
        return self.cp_slicing is CPSlicing.CONTIGUOUS

    @property
    def has_cp(self) -> bool:
        return self.cp_size > 1
