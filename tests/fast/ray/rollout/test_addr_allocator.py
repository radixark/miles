from __future__ import annotations

from unittest.mock import patch

from tests.fast.ray.rollout.conftest import chunk_engines_into_cells, fake_actor_handle, fake_engine, make_args

import miles.ray.rollout.server_group as server_group_module
from miles.ray.rollout.addr_allocator import PortAllocator
from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.rollout.server_group import ServerGroup


def _start_engines_and_collect_addressing(
    *,
    args,
    port_allocator: PortAllocator,
    rollout_engines,
    worker_type: str = "regular",
    num_gpus_per_engine: int | None = None,
    rank_offset: int = 0,
) -> dict[int, dict]:
    """Run ``ServerGroup.start_engines`` against the given actor mocks and return,
    per global rank, the kwargs its ``init`` was called with."""
    gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    nodes_per_engine = max(1, gpus_per_engine // args.num_gpus_per_node)
    requested = dict(rollout_engines)
    slots = [ServerEngine() for _ in range(max(requested) - rank_offset + 1)]
    group = ServerGroup(
        args=args,
        pg=None,
        cells=chunk_engines_into_cells(
            slots, num_gpus_per_engine=gpus_per_engine, num_gpus_per_node=args.num_gpus_per_node
        ),
        num_gpus_per_engine=gpus_per_engine,
        has_new_engines=False,
        worker_type=worker_type,
        rank_offset=rank_offset,
    )
    for index, slot in enumerate(slots):
        if rank_offset + index not in requested:
            slot.mark_allocated_uninitialized(fake_actor_handle())
    started_cell_indices = sorted({(rank - rank_offset) // nodes_per_engine for rank in requested})

    def _launch(*, global_rank, **kwargs):
        return requested[global_rank]

    with patch.object(server_group_module, "launch_sglang_ray_actor", side_effect=_launch):
        group.start_engines(port_allocator, start_cell_indices=started_cell_indices)

    return {rank: dict(engine.init.remote.call_args.kwargs) for rank, engine in requested.items()}


class TestPortAllocator:
    def test_empty_has_no_values(self):
        c = PortAllocator.empty()
        assert c._values == {}

    def test_alloc_advances_the_cursor_of_its_node(self, patch_ray_get):
        """Two allocations on the same node must hand out non-overlapping ports."""
        cursors = PortAllocator.empty()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        first = cursors.alloc(engine=engine, node_ip="10.0.0.1")
        second = cursors.alloc(engine=engine, node_ip="10.0.0.1")
        assert second > first
        assert cursors._values["10.0.0.1"] == second + 1

    def test_alloc_starts_from_the_base_port_on_an_unseen_node(self, patch_ray_get):
        """A node with no cursor yet starts at the base port, away from ray's range."""
        cursors = PortAllocator.empty()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        assert cursors.alloc(engine=engine, node_ip="10.0.0.1") == 15000

    def test_alloc_consecutive_reserves_a_whole_block(self, patch_ray_get):
        """A consecutive=N allocation must move this node's cursor past the entire block."""
        cursors = PortAllocator.empty()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        first = cursors.alloc(engine=engine, node_ip="10.0.0.1", consecutive=5)
        assert cursors._values["10.0.0.1"] == first + 5

    def test_alloc_tracks_nodes_independently(self, patch_ray_get):
        """Each node ip owns its own cursor."""
        cursors = PortAllocator.empty()
        engine_a = fake_engine(host="10.0.0.1", port_seed=0)
        engine_b = fake_engine(host="10.0.0.2", port_seed=0)
        cursors.alloc(engine=engine_a, node_ip="10.0.0.1")
        cursors.alloc(engine=engine_b, node_ip="10.0.0.2")
        assert set(cursors._values.keys()) == {"10.0.0.1", "10.0.0.2"}


def _all_ports(addr_and_ports: dict) -> list[int]:
    """Flatten every numeric port in every rank's entry."""
    out: list[int] = []
    for entry in addr_and_ports.values():
        for k, v in entry.items():
            if k == "host":
                continue
            if k == "dist_init_addr":
                # "host:port" → grab the port half
                out.append(int(v.rsplit(":", 1)[1]))
            elif v is not None:
                out.append(int(v))
    return out


class TestAddressingOfStartedEngines:
    def test_single_node_8_cards_tp1(self, patch_ray_get):
        """Eight single-gpu engines on one node get complete, mutually distinct addressing."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, fake_engine(host="10.0.0.1", port_seed=30000)) for rank in range(8)]
        cursors = PortAllocator.empty()
        addr_and_ports = _start_engines_and_collect_addressing(
            args=args, port_allocator=cursors, rollout_engines=engines, num_gpus_per_engine=1
        )

        assert set(addr_and_ports.keys()) == set(range(8))
        for rank in range(8):
            assert addr_and_ports[rank]["host"] == "10.0.0.1"
            assert isinstance(addr_and_ports[rank]["port"], int)
            assert isinstance(addr_and_ports[rank]["nccl_port"], int)
            assert isinstance(addr_and_ports[rank]["engine_info_bootstrap_port"], int)
            # dist_init_addr has the form "host:port" → check both halves.
            host, _, port_str = addr_and_ports[rank]["dist_init_addr"].partition(":")
            assert host == "10.0.0.1"
            assert int(port_str) >= 30000
            # No same-rank collisions among the port fields.
            same_rank_ports = {
                addr_and_ports[rank]["port"],
                addr_and_ports[rank]["nccl_port"],
                addr_and_ports[rank]["engine_info_bootstrap_port"],
                int(port_str),
            }
            assert len(same_rank_ports) == 4, f"rank {rank} reused a port: {addr_and_ports[rank]}"

        # Cursor must reflect the *node*'s next free port (single node → its ip).
        assert set(cursors._values.keys()) == {"10.0.0.1"}
        # And it must sit past every port we handed out.
        assert cursors._values["10.0.0.1"] >= max(_all_ports(addr_and_ports)) + 1

        # Cross-rank: every numeric port across all 8 engines must be unique.
        all_ports = _all_ports(addr_and_ports)
        assert len(all_ports) == len(set(all_ports)), f"port collision across engines on the same node: {all_ports}"

    def test_prefill_worker_gets_disagg_bootstrap_port(self, patch_ray_get):
        """A prefill engine's disaggregation bootstrap port is distinct from its other ports."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, fake_engine()) for rank in range(2)]
        addr_and_ports = _start_engines_and_collect_addressing(
            args=args,
            port_allocator=PortAllocator.empty(),
            rollout_engines=engines,
            worker_type="prefill",
            num_gpus_per_engine=1,
        )
        for rank in range(2):
            entry = addr_and_ports[rank]
            assert isinstance(entry["disaggregation_bootstrap_port"], int)
            assert entry["disaggregation_bootstrap_port"] not in (
                entry["port"],
                entry["nccl_port"],
                entry["engine_info_bootstrap_port"],
            )

    def test_regular_worker_does_not_get_disagg_bootstrap_port(self, patch_ray_get):
        """Only prefill engines carry a disaggregation bootstrap port; others must not reserve one."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, fake_engine()) for rank in range(2)]
        addr_and_ports = _start_engines_and_collect_addressing(
            args=args, port_allocator=PortAllocator.empty(), rollout_engines=engines, num_gpus_per_engine=1
        )
        for rank in range(2):
            assert "disaggregation_bootstrap_port" not in addr_and_ports[rank]

    def test_gpus_per_engine_greater_than_node_shares_dist_init_addr(self, patch_ray_get):
        """When `gpus_per_engine > num_gpus_per_node`, all ranks of one engine
        share a single ``dist_init_addr`` (multi-node engine)."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        # 2-node engine: 16 gpus total, 8 per node, 2 ranks share dist_init_addr
        engines = [(rank, fake_engine(host="10.0.0.42")) for rank in range(2)]
        addr_and_ports = _start_engines_and_collect_addressing(
            args=args, port_allocator=PortAllocator.empty(), rollout_engines=engines, num_gpus_per_engine=16
        )
        assert addr_and_ports[0]["dist_init_addr"] == addr_and_ports[1]["dist_init_addr"]
        host, _, port_str = addr_and_ports[0]["dist_init_addr"].partition(":")
        assert host == "10.0.0.42"
        assert int(port_str) > 0

    def test_rank_offset_does_not_break_indexing(self, patch_ray_get):
        """A group starting at rank 4 populates exactly ranks 4..7 with collision-free ports."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, fake_engine(host="10.0.0.7", port_seed=40000)) for rank in (4, 5, 6, 7)]
        addr_and_ports = _start_engines_and_collect_addressing(
            args=args,
            port_allocator=PortAllocator.empty(),
            rollout_engines=engines,
            num_gpus_per_engine=1,
            rank_offset=4,
        )
        # Exactly the requested ranks are populated; no leakage into 0..3.
        assert set(addr_and_ports.keys()) == {4, 5, 6, 7}
        for r in (4, 5, 6, 7):
            assert addr_and_ports[r]["host"] == "10.0.0.7"
            assert isinstance(addr_and_ports[r]["port"], int)
            assert addr_and_ports[r]["port"] >= 40000
        # Ports across all populated ranks must not collide.
        all_ports = _all_ports(addr_and_ports)
        assert len(all_ports) == len(set(all_ports))

    def test_mid_rank_restart_populates_only_the_requested_rank(self, patch_ray_get):
        """Restarting rank 3 must allocate for rank 3 alone; other slots on the
        node keep their existing engines and ports."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(3, fake_engine())]
        addr_and_ports = _start_engines_and_collect_addressing(
            args=args, port_allocator=PortAllocator.empty(), rollout_engines=engines, num_gpus_per_engine=1
        )
        assert set(addr_and_ports.keys()) == {3}
        for k in ("host", "port", "nccl_port", "engine_info_bootstrap_port", "dist_init_addr"):
            assert k in addr_and_ports[3]

    def test_cursor_ends_past_every_issued_port(self, patch_ray_get):
        """The node cursor is left beyond every port handed out, including reserved blocks."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(0, fake_engine(port_seed=22000))]
        cursors = PortAllocator.empty()
        addr_and_ports = _start_engines_and_collect_addressing(
            args=args, port_allocator=cursors, rollout_engines=engines, num_gpus_per_engine=1
        )
        # Cursor must sit strictly past every port we handed out (the allocation
        # also reserves consecutive blocks for dist_init_addr that aren't all
        # visible in the output, so we can't pin to max_issued + 1).
        max_issued = max(_all_ports(addr_and_ports))
        assert cursors._values["10.0.0.1"] > max_issued


class TestSharedPortAllocatorAcrossGroups:
    """Two ``ServerGroup``s sharing one ``PortAllocator`` must produce disjoint
    port allocations across nodes — required for parallel recover."""

    def test_sequential_groups_share_cursor_and_avoid_overlap(self, patch_ray_get):
        """Groups started one after another off a shared allocator never reuse a port."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        cursors = PortAllocator.empty()

        engines_a = [(rank, fake_engine(port_seed=0)) for rank in range(4)]
        addrs_a = _start_engines_and_collect_addressing(
            args=args,
            port_allocator=cursors,
            rollout_engines=engines_a,
            num_gpus_per_engine=1,
        )

        engines_b = [(rank, fake_engine(port_seed=0)) for rank in range(4, 8)]
        addrs_b = _start_engines_and_collect_addressing(
            args=args,
            port_allocator=cursors,
            rollout_engines=engines_b,
            num_gpus_per_engine=1,
            rank_offset=4,
        )

        ports_a = {addrs_a[r]["port"] for r in addrs_a} | {addrs_a[r]["nccl_port"] for r in addrs_a}
        ports_b = {addrs_b[r]["port"] for r in addrs_b} | {addrs_b[r]["nccl_port"] for r in addrs_b}
        assert ports_a.isdisjoint(ports_b), f"port overlap A={ports_a} B={ports_b}"


class TestRankPortConsistency:
    """rank ↔ addr_and_ports consistency inside ``ServerGroup.start_engines``.

    The init loop iterates ``new_engines`` as ``(global_rank, engine)`` pairs, so
    the addressing must be keyed by global rank even when ``rank_offset != 0`` or
    ``nodes_per_engine > 1``."""

    def test_rank_offset_kwargs_keyed_by_global_rank(self, patch_ray_get):
        """When rank_offset=4, addr_and_ports must be keyed by ranks 4..7, not 0..3."""
        args = make_args(num_gpus_per_node=4, sglang_dp_size=1)
        engines = [(rank, fake_engine(port_seed=0)) for rank in range(4, 8)]
        addr_and_ports = _start_engines_and_collect_addressing(
            args=args,
            port_allocator=PortAllocator.empty(),
            rollout_engines=engines,
            num_gpus_per_engine=1,
            rank_offset=4,
        )
        assert set(addr_and_ports.keys()) == {4, 5, 6, 7}

    def test_each_global_rank_has_complete_kwargs(self, patch_ray_get):
        """Every started rank receives the full addressing kwarg set."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, fake_engine(port_seed=0)) for rank in range(4)]
        addr_and_ports = _start_engines_and_collect_addressing(
            args=args,
            port_allocator=PortAllocator.empty(),
            rollout_engines=engines,
            num_gpus_per_engine=1,
        )
        for rank in range(4):
            kw = addr_and_ports[rank]
            for key in ("host", "port", "nccl_port", "dist_init_addr"):
                assert key in kw, f"rank {rank} missing {key}"

    def test_multinode_engine_shares_dist_init_addr_across_node_ranks(self, patch_ray_get):
        """nodes_per_engine=2 (16 gpus, 8 per node) — both ranks of one
        multi-node engine MUST get the same dist_init_addr."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(0, fake_engine(port_seed=0)), (1, fake_engine(port_seed=0))]
        addr_and_ports = _start_engines_and_collect_addressing(
            args=args,
            port_allocator=PortAllocator.empty(),
            rollout_engines=engines,
            num_gpus_per_engine=16,
        )
        assert addr_and_ports[0]["dist_init_addr"] == addr_and_ports[1]["dist_init_addr"]

    def test_init_kwargs_exist_for_every_started_rank(self, patch_ray_get):
        """For every (rank, engine) pair the init loop walks, the addressing dict has an entry."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        new_engines = [(rank, fake_engine(port_seed=0)) for rank in range(2, 6)]
        addr_and_ports = _start_engines_and_collect_addressing(
            args=args,
            port_allocator=PortAllocator.empty(),
            rollout_engines=new_engines,
            num_gpus_per_engine=1,
            rank_offset=2,
        )
        for index, _engine in new_engines:
            assert index in addr_and_ports, f"missing addr_and_ports for global_rank={index}"
            for key in ("host", "port", "nccl_port", "dist_init_addr"):
                assert key in addr_and_ports[index]

    def test_ports_are_unique_within_a_node(self, patch_ray_get):
        """No two engines on the same node share any of their allocated ports."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, fake_engine(port_seed=0)) for rank in range(8)]
        addr_and_ports = _start_engines_and_collect_addressing(
            args=args,
            port_allocator=PortAllocator.empty(),
            rollout_engines=engines,
            num_gpus_per_engine=1,
        )
        all_ports = []
        for kw in addr_and_ports.values():
            all_ports.extend([kw["port"], kw["nccl_port"], kw["engine_info_bootstrap_port"]])
        assert len(set(all_ports)) == len(all_ports), f"duplicate ports: {all_ports}"
