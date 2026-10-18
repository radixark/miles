"""What Step5's full-attention layers hand from region selection to sparse attention.

The trainer recomputes the selection with the frozen indexer (no rollout replay): for each query
token and each KV group, up to 512 scored 8-token regions plus the region holding the token.
"""

from dataclasses import dataclass

import torch

# the sparse kernel's region granularity, so the selection and the attention agree on it
from miles.kernels.attention.region_sparse import REGION_SIZE  # noqa: F401

TOP_REGIONS = 512
# The scored regions plus the query's own (partial) region.
MAX_SELECTED_REGIONS = TOP_REGIONS + 1


@dataclass(frozen=True)
class RegionSelection:
    """The regions each query token may attend to, per local KV group.

    Regions are numbered from the start of the token's own sequence in the THD pack, so region r
    covers that sequence's tokens [8r, 8r + 8). Row t lists ascending region ids for each local
    KV group, padded with -1. The region holding token t is always its last valid entry, and keys
    inside it are visible only up to t.
    """

    region_ids: torch.Tensor  # int32 [T, num_local_kv_groups, MAX_SELECTED_REGIONS]
    cu_seqlens: torch.Tensor  # int32 [num_sequences + 1], the THD boundaries the ids refer to
