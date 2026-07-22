from __future__ import annotations

from unittest.mock import MagicMock

from tests.fast.ray.rollout.conftest import fake_engine, make_args

from miles.ray.rollout.addr_allocator import (
    PortCursors,
    allocate_rollout_engine_addr_and_ports_external,
    allocate_rollout_engine_addr_and_ports_normal,
)


class TestPortCursors:
    def test_empty_has_no_values(self):
        c = PortCursors.empty()
        assert c._values == {}

    def test_next_base_port_default_when_empty(self):
        assert PortCursors.empty().next_base_port() == 15000

    def test_next_base_port_returns_max_value(self):
        c = PortCursors(_values={0: 17000, 1: 16500, 2: 18000})
        assert c.next_base_port() == 18000

    def test_assign_copies_values(self):
        a = PortCursors.empty()
        b = PortCursors(_values={0: 19000, 1: 19500})
        a.assign(b)
        assert a._values == {0: 19000, 1: 19500}

    def test_assign_is_decoupled(self):
        """After assign, mutating source must not bleed into target."""
        a = PortCursors.empty()
        b = PortCursors(_values={0: 19000})
        a.assign(b)
        b._values[0] = 99999
        assert a._values == {0: 19000}, "assign must deep-copy the inner dict"


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


class TestAllocateNormal:
    def test_single_node_8_cards_tp1(self, patch_ray_get):
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, fake_engine(host="10.0.0.1", port_seed=30000)) for rank in range(8)]
        addr_and_ports, cursors = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=1, base_port=30000
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
            # No same-rank collisions among the four port fields.
            same_rank_ports = {
                addr_and_ports[rank]["port"],
                addr_and_ports[rank]["nccl_port"],
                addr_and_ports[rank]["engine_info_bootstrap_port"],
                int(port_str),
            }
            assert len(same_rank_ports) == 4, f"rank {rank} reused a port: {addr_and_ports[rank]}"

        # Cursor must reflect the *node*'s next free port (single-node here → key 0).
        assert isinstance(cursors, PortCursors)
        assert set(cursors._values.keys()) == {0}
        # And it must sit past every port we handed out.
        assert cursors._values[0] >= max(_all_ports(addr_and_ports)) + 1

        # Cross-rank: every numeric port across all 8 engines must be unique.
        all_ports = _all_ports(addr_and_ports)
        assert len(all_ports) == len(set(all_ports)), f"port collision across engines on the same node: {all_ports}"

    def test_prefill_worker_gets_disagg_bootstrap_port(self, patch_ray_get):
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, fake_engine()) for rank in range(2)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args,
            rollout_engines=engines,
            worker_type="prefill",
            num_gpus_per_engine=1,
        )
        for rank in range(2):
            assert isinstance(addr_and_ports[rank]["disaggregation_bootstrap_port"], int)
        # The disagg port must be distinct from the other ports on the same rank.
        for rank in range(2):
            entry = addr_and_ports[rank]
            assert entry["disaggregation_bootstrap_port"] not in (
                entry["port"],
                entry["nccl_port"],
                entry["engine_info_bootstrap_port"],
            )

    def test_regular_worker_does_not_get_disagg_bootstrap_port(self, patch_ray_get):
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, fake_engine()) for rank in range(2)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=1
        )
        for rank in range(2):
            assert "disaggregation_bootstrap_port" not in addr_and_ports[rank]

    def test_gpus_per_engine_greater_than_node_shares_dist_init_addr(self, patch_ray_get):
        """When `_gpus_per_engine > num_gpus_per_node`, all ranks of one engine
        share a single ``dist_init_addr`` (multi-node engine)."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        # 2-node engine: 16 gpus total, 8 per node, 2 ranks share dist_init_addr
        engines = [(rank, fake_engine(host="10.0.0.42")) for rank in range(2)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=16
        )
        # Same string for both ranks — not just equal, identity of representation.
        assert addr_and_ports[0]["dist_init_addr"] == addr_and_ports[1]["dist_init_addr"]
        host, _, port_str = addr_and_ports[0]["dist_init_addr"].partition(":")
        assert host == "10.0.0.42"
        assert int(port_str) > 0

    def test_rank_offset_does_not_break_indexing(self, patch_ray_get):
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, fake_engine(host="10.0.0.7", port_seed=40000)) for rank in (4, 5, 6, 7)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=1, rank_offset=4
        )
        # Allocator fills remaining slots on the node starting from rank 4
        # (see source comment: "we will set port for engine 3,4,5,6,7 on this
        # node"); so the requested ranks {4,5,6,7} must be a subset, with no
        # leakage into ranks 0..3.
        assert {4, 5, 6, 7} <= set(addr_and_ports.keys())
        assert set(addr_and_ports.keys()).isdisjoint({0, 1, 2, 3})
        for r in (4, 5, 6, 7):
            assert addr_and_ports[r]["host"] == "10.0.0.7"
            assert isinstance(addr_and_ports[r]["port"], int)
            assert addr_and_ports[r]["port"] >= 40000
        # Ports across all populated ranks must not collide.
        all_ports = _all_ports(addr_and_ports)
        assert len(all_ports) == len(set(all_ports))

    def test_mid_rank_restart_fills_remaining_slots_on_node(self, patch_ray_get):
        """Restarting starting from rank 3 on an 8-card node should populate
        addr_and_ports[3..7], i.e. ``num_engines_on_this_node`` accounts for
        the offset within the node."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(3, fake_engine())]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=1
        )
        # Exact key set: mid-rank fill must cover [3..7] and ONLY those.
        assert set(addr_and_ports.keys()) == {3, 4, 5, 6, 7}
        # Each filled slot has a complete addr/port set.
        for r in range(3, 8):
            for k in ("host", "port", "nccl_port", "engine_info_bootstrap_port", "dist_init_addr"):
                assert k in addr_and_ports[r]

    def test_base_port_propagates_into_cursor(self, patch_ray_get):
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(0, fake_engine(port_seed=22000))]
        addr_and_ports, cursors = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=1, base_port=22000
        )
        # Cursor must sit strictly past every port we handed out (the allocator
        # also reserves consecutive blocks for dist_init_addr that aren't all
        # visible in the output, so we can't pin to max_issued + 1).
        max_issued = max(_all_ports(addr_and_ports))
        assert cursors._values[0] > max_issued
        # And the lowest port must be >= base_port (allocator never went below it).
        assert min(_all_ports(addr_and_ports)) >= 22000


class TestAllocateExternal:
    def test_basic_split(self):
        args = make_args(rollout_external_engine_addrs=["10.0.0.1:30000", "10.0.0.2:30001"])
        engines = [(0, MagicMock()), (1, MagicMock())]
        result = allocate_rollout_engine_addr_and_ports_external(args=args, rollout_engines=engines)
        # Whole-dict equality (not just a key check) — pins the entire shape.
        assert result == {
            0: dict(dist_init_addr="10.0.0.1:30000", nccl_port=None, host="10.0.0.1", port=30000),
            1: dict(dist_init_addr="10.0.0.2:30001", nccl_port=None, host="10.0.0.2", port=30001),
        }

    def test_ipv4_addr_split_is_consistent(self):
        args = make_args(rollout_external_engine_addrs=["192.168.1.10:31000"])
        engines = [(0, MagicMock())]
        result = allocate_rollout_engine_addr_and_ports_external(args=args, rollout_engines=engines)
        assert result == {
            0: dict(dist_init_addr="192.168.1.10:31000", nccl_port=None, host="192.168.1.10", port=31000),
        }
        # `port` is an int, not a string accidentally split through.
        assert isinstance(result[0]["port"], int)


class TestSharedPortCursorsAcrossGroups:
    """Two ``ServerGroup``s sharing one ``PortCursors`` must produce disjoint
    port allocations across nodes — required for parallel recover."""

    def test_sequential_groups_share_cursor_and_avoid_overlap(self, patch_ray_get):
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        cursors = PortCursors.empty()

        engines_a = [(rank, fake_engine(port_seed=0)) for rank in range(4)]
        addrs_a, next_a = allocate_rollout_engine_addr_and_ports_normal(
            args=args,
            rollout_engines=engines_a,
            num_gpus_per_engine=1,
            base_port=cursors.next_base_port(),
        )
        cursors.assign(next_a)

        engines_b = [(rank, fake_engine(port_seed=0)) for rank in range(4, 8)]
        addrs_b, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args,
            rollout_engines=engines_b,
            num_gpus_per_engine=1,
            base_port=cursors.next_base_port(),
            rank_offset=4,
        )

        ports_a = {addrs_a[r]["port"] for r in addrs_a} | {addrs_a[r]["nccl_port"] for r in addrs_a}
        ports_b = {addrs_b[r]["port"] for r in addrs_b} | {addrs_b[r]["nccl_port"] for r in addrs_b}
        assert ports_a.isdisjoint(ports_b), f"port overlap A={ports_a} B={ports_b}"


class TestRankPortConsistency:
    """rank ↔ addr_and_ports index consistency in ``ServerGroup.start_engines``.

    The init-handles loop iterates ``new_engines`` as ``(global_rank, engine)``
    pairs while the allocator keys its output dict on ``rank + i``. When
    ``rank_offset != 0`` or ``nodes_per_engine > 1`` the two index spaces
    must still agree."""

    def test_rank_offset_kwargs_keyed_by_global_rank(self, patch_ray_get):
        """When rank_offset=4, addr_and_ports must be keyed by ranks 4..7,
        not 0..3.

        ``num_gpus_per_node=4`` so a 4-engine group exactly fills one node —
        otherwise the allocator pads up to ``num_engines_per_node``."""
        args = make_args(num_gpus_per_node=4, sglang_dp_size=1)
        engines = [(rank, fake_engine(port_seed=0)) for rank in range(4, 8)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args,
            rollout_engines=engines,
            num_gpus_per_engine=1,
            rank_offset=4,
        )
        assert set(addr_and_ports.keys()) == {4, 5, 6, 7}

    def test_each_global_rank_has_complete_kwargs(self, patch_ray_get):
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, fake_engine(port_seed=0)) for rank in range(4)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args,
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
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args,
            rollout_engines=engines,
            num_gpus_per_engine=16,
        )
        assert addr_and_ports[0]["dist_init_addr"] == addr_and_ports[1]["dist_init_addr"]

    def test_init_handles_iteration_pairs_match_addr_dict(self, patch_ray_get):
        """For every (index, engine) pair in server_group.py's loop,
        addr_and_ports[index] must contain all required kwargs."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        new_engines = [(rank, fake_engine(port_seed=0)) for rank in range(2, 6)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args,
            rollout_engines=new_engines,
            num_gpus_per_engine=1,
            rank_offset=2,
        )
        for index, _engine in new_engines:
            assert index in addr_and_ports, f"missing addr_and_ports for global_rank={index}"
            for key in ("host", "port", "nccl_port", "dist_init_addr"):
                assert key in addr_and_ports[index]

    def test_ports_are_unique_within_a_node(self, patch_ray_get):
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, fake_engine(port_seed=0)) for rank in range(8)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args,
            rollout_engines=engines,
            num_gpus_per_engine=1,
        )
        all_ports = []
        for kw in addr_and_ports.values():
            all_ports.extend([kw["port"], kw["nccl_port"], kw["engine_info_bootstrap_port"]])
        assert len(set(all_ports)) == len(all_ports), f"duplicate ports: {all_ports}"
