from __future__ import annotations

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability

from miles.utils.workers.pod_context import (
    LEADER_ADDRESS_ENV_VAR,
    POD_INDEX_ENV_VAR,
    SUBPROCESS_INDEX_ENV_VAR,
    read_cell_ordinal,
    read_pod_rank,
)


class TestReadCellIndex:
    def test_reads_the_cell_out_of_the_pool_leaders_address(self):
        """A pod is told its leader's hostname, and the group index is the only cell identity it carries."""
        assert read_cell_ordinal({LEADER_ADDRESS_ENV_VAR: "myrun-miles-run-trainer-actor-3.myrun-miles-run"}) == 3

    def test_reads_a_bare_leader_hostname_too(self):
        """A platform may hand over the short name; the group index sits in the same place either way."""
        assert read_cell_ordinal({LEADER_ADDRESS_ENV_VAR: "trainer-actor-11"}) == 11

    def test_a_pod_outside_a_pool_belongs_to_the_only_cell_there_is(self):
        """A static worker runs as a plain workload with one cell, so there is no group index to read."""
        assert read_cell_ordinal({}) == 0

    def test_refuses_a_leader_address_with_no_group_index(self):
        """Guessing zero here would silently merge every cell of an indep-dp pool_id into the first one."""
        with pytest.raises(AssertionError, match=LEADER_ADDRESS_ENV_VAR):
            read_cell_ordinal({LEADER_ADDRESS_ENV_VAR: "leader.svc"})


class TestReadPodRank:
    def test_numbers_the_ranks_of_a_multi_pod_cell_end_to_end(self):
        """A cell's ranks are numbered across its pods, so each pod has to offset by its own index."""
        rank = read_pod_rank(
            ranks_per_pod=8,
            gpu_slots_per_rank=1,
            environ={POD_INDEX_ENV_VAR: "2", SUBPROCESS_INDEX_ENV_VAR: "3"},
        )

        assert rank.worker_in_cell_index == 19

    def test_gives_each_rank_of_a_pod_its_own_gpus(self):
        """Two ranks that claimed the same gpu would each initialise a full model on it."""
        first = read_pod_rank(ranks_per_pod=4, gpu_slots_per_rank=2, environ={SUBPROCESS_INDEX_ENV_VAR: "0"})
        second = read_pod_rank(ranks_per_pod=4, gpu_slots_per_rank=2, environ={SUBPROCESS_INDEX_ENV_VAR: "1"})

        assert (first.gpu_ids, second.gpu_ids) == ([0, 1], [2, 3])

    def test_a_single_rank_pod_needs_no_environment_at_all(self):
        """A static worker runs one rank with no supervisor, and must not depend on variables nobody sets."""
        rank = read_pod_rank(ranks_per_pod=1, gpu_slots_per_rank=0, environ={})

        assert (rank.cell_ordinal, rank.worker_in_cell_index, rank.gpu_ids) == (0, 0, [])

    def test_the_ctor_context_carries_the_cell_the_pod_is_in(self):
        """ctor kwargs are computed from this context, and a trainer keys its checkpoints off the cell index."""
        rank = read_pod_rank(
            ranks_per_pod=2,
            gpu_slots_per_rank=1,
            environ={LEADER_ADDRESS_ENV_VAR: "trainer-actor-1.trainer-actor", SUBPROCESS_INDEX_ENV_VAR: "1"},
        )

        context = rank.ctor_context(pool_id="trainer-actor", capability=FakeBackendCapability())

        assert (context.cell_id, context.cell_ordinal, context.worker_in_cell_index, context.gpu_ids) == (
            "trainer-actor-1",
            1,
            1,
            [1],
        )

    def test_the_ctor_context_always_carries_a_provider_factory(self):
        """A spec asking for its engines must never find the factory missing, so it cannot be omitted."""
        rank = read_pod_rank(ranks_per_pod=1, gpu_slots_per_rank=0, environ={})

        with pytest.raises(TypeError):
            rank.ctor_context()

    def test_refuses_a_subprocess_index_the_pod_was_not_launched_for(self):
        """A rank beyond the pod's share would collide with a rank of the next pod in the cell."""
        with pytest.raises(AssertionError, match=SUBPROCESS_INDEX_ENV_VAR):
            read_pod_rank(ranks_per_pod=2, gpu_slots_per_rank=1, environ={SUBPROCESS_INDEX_ENV_VAR: "2"})
