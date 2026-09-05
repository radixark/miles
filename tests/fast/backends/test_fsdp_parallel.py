"""FSDP's ParallelState: every non-data axis is a real single-rank group.

The shared weight-update protocols derive the sender from the pp group's
ranks; a None group resolves to the world and makes every rank a sender.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

from miles.backends.fsdp_utils import parallel as parallel_module

_MODULE = "miles.backends.fsdp_utils.parallel"


def test_the_degree_one_axes_get_their_own_single_rank_group():
    self_group = object()
    with (
        patch(f"{_MODULE}.dist") as dist,
        patch(f"{_MODULE}.build_fsdp_meshes", return_value={"dp": MagicMock(), "fsdp": MagicMock()}),
        patch(f"{_MODULE}.get_gloo_group", return_value=object()),
    ):
        dist.get_world_size.return_value = 4
        dist.get_rank.return_value = 3
        dist.new_group.return_value = self_group
        state = parallel_module.create_fsdp_parallel_state(Namespace(dp_replicate_size=1))
    dist.new_group.assert_called_once_with([3])
    for axis in (state.pp, state.ep, state.etp, state.indep_dp, state.tp, state.cp):
        assert axis.group is self_group
        assert (axis.rank, axis.size) == (0, 1)
    assert (state.intra_dp.rank, state.intra_dp.size) == (3, 4)
