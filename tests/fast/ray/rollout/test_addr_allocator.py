from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import ray

from tests.fast.ray.rollout.conftest import fake_engine, make_args

import miles.ray.rollout.server_cell as server_cell_module
from miles.ray.rollout.server_cell import ServerCell
from miles.utils.test_utils.mock_sglang_engine import parse_cmd_flags
from miles.utils.workers.addr_allocator import PortAllocator


def _start_engines_and_collect_addressing(
    *,
    args,
    port_allocator: PortAllocator,
    rollout_engines,
    worker_type: str = "regular",
) -> dict[int, dict]:
    """Run ``ServerCell.start_engines`` against the given actor mocks and return,
    per global rank, the addressing flags of the launch command its ``run`` got."""
    requested = dict(rollout_engines)
    cell = ServerCell(
        args=args,
        num_nodes=len(requested),
        worker_type=worker_type,
        cell_id="cell-0",
        rank_offset=min(requested),
        pg=(None, [], list(range(8))),
    )
    for engine in requested.values():
        engine.__class__ = ray.actor.ActorHandle
        engine.run.remote.side_effect = lambda **kwargs: asyncio.sleep(0)

    def _launch(*, global_rank, **kwargs):
        return requested[global_rank]

    with (
        patch.object(server_cell_module, "launch_sglang_ray_actor", side_effect=_launch),
        patch.object(server_cell_module, "wait_server_healthy", new=AsyncMock()),
    ):
        asyncio.run(cell.start_engines(port_allocator))

    return {rank: parse_cmd_flags(engine.run.remote.call_args.kwargs["cmd"]) for rank, engine in requested.items()}


_PORT_KEYS = ("port", "nccl_port", "engine_info_bootstrap_port", "disaggregation_bootstrap_port")


def _all_ports(addr_and_ports: dict) -> list[int]:
    """Flatten every numeric port in every rank's entry."""
    out: list[int] = []
    for entry in addr_and_ports.values():
        out.extend(int(entry[key]) for key in _PORT_KEYS if key in entry)
        # dist_init_addr is "host:port" → grab the port half
        out.append(int(entry["dist_init_addr"].rsplit(":", 1)[1]))
    return out


def _alloc_single_engine_cells(args, cursors: PortAllocator, engines, worker_type: str = "regular") -> dict:
    """One allocator call per single-engine cell, mirroring how cells allocate."""
    addr_and_ports: dict[int, dict] = {}
    for rank, engine in engines:
        addr_and_ports.update(
            _start_engines_and_collect_addressing(
                args=args, port_allocator=cursors, rollout_engines=[(rank, engine)], worker_type=worker_type
            )
        )
    return addr_and_ports


class TestAddressingOfStartedEngines:
    def test_single_node_8_cards_tp1(self, patch_ray_get):
        """Eight single-gpu cells on one node get complete, mutually distinct addressing."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, fake_engine(host="10.0.0.1", port_seed=30000)) for rank in range(8)]
        cursors = PortAllocator()
        addr_and_ports = _alloc_single_engine_cells(args, cursors, engines)

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
        addr_and_ports = _alloc_single_engine_cells(args, PortAllocator(), engines, worker_type="prefill")
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
        """Only prefill engines carry a disaggregation bootstrap port; others must not reserve one."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, fake_engine()) for rank in range(2)]
        addr_and_ports = _alloc_single_engine_cells(args, PortAllocator(), engines)
        for rank in range(2):
            assert "disaggregation_bootstrap_port" not in addr_and_ports[rank]

    def test_multinode_cell_shares_a_dist_init_addr_on_the_primary_node(self, patch_ray_get):
        """A 2-node cell (16 gpus, 8 per node) gets one dist_init_addr, allocated
        on the primary (first) engine's node and shared by both node-ranks."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(0, fake_engine(host="10.0.0.42")), (1, fake_engine(host="10.0.0.43"))]
        addr_and_ports = _start_engines_and_collect_addressing(
            args=args, port_allocator=PortAllocator(), rollout_engines=engines
        )
        assert addr_and_ports[0]["dist_init_addr"] == addr_and_ports[1]["dist_init_addr"]
        host, _, port_str = addr_and_ports[0]["dist_init_addr"].partition(":")
        assert host == "10.0.0.42"
        assert int(port_str) > 0

    def test_nonzero_ranks_key_the_result_by_global_rank(self, patch_ray_get):
        """A batch starting at rank 4 populates exactly ranks 4..7 with collision-free ports."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, fake_engine(host="10.0.0.7", port_seed=40000)) for rank in (4, 5, 6, 7)]
        addr_and_ports = _alloc_single_engine_cells(args, PortAllocator(), engines)
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
            args=args, port_allocator=PortAllocator(), rollout_engines=engines
        )
        assert set(addr_and_ports.keys()) == {3}
        for k in ("host", "port", "nccl_port", "engine_info_bootstrap_port", "dist_init_addr"):
            assert k in addr_and_ports[3]

    def test_cursor_ends_past_every_issued_port(self, patch_ray_get):
        """The node cursor is left beyond every port handed out, including reserved blocks."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(0, fake_engine(port_seed=22000))]
        cursors = PortAllocator()
        addr_and_ports = _start_engines_and_collect_addressing(
            args=args, port_allocator=cursors, rollout_engines=engines
        )
        # Cursor must sit strictly past every port we handed out (the allocator
        # also reserves consecutive blocks for dist_init_addr that aren't all
        # visible in the output, so we can't pin to max_issued + 1).
        max_issued = max(_all_ports(addr_and_ports))
        assert cursors._values["10.0.0.1"] > max_issued


class TestSharedPortAllocatorAcrossCells:
    """Cell batches sharing one ``PortAllocator`` must produce disjoint
    port allocations across nodes — required for parallel recover."""

    def test_sequential_batches_share_cursor_and_avoid_overlap(self, patch_ray_get):
        """Batches started one after another off a shared allocator never reuse a port."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        cursors = PortAllocator()

        engines_a = [(rank, fake_engine(port_seed=0)) for rank in range(4)]
        addrs_a = _alloc_single_engine_cells(args, cursors, engines_a)

        engines_b = [(rank, fake_engine(port_seed=0)) for rank in range(4, 8)]
        addrs_b = _alloc_single_engine_cells(args, cursors, engines_b)

        ports_a = {addrs_a[r]["port"] for r in addrs_a} | {addrs_a[r]["nccl_port"] for r in addrs_a}
        ports_b = {addrs_b[r]["port"] for r in addrs_b} | {addrs_b[r]["nccl_port"] for r in addrs_b}
        assert ports_a.isdisjoint(ports_b), f"port overlap A={ports_a} B={ports_b}"


class TestConcurrentNodeProbes:
    async def test_a_cell_probes_all_of_its_nodes_concurrently(self, patch_ray_get):
        """Serializing the node probes would make cell startup scale with the node count."""
        events: list[tuple[str, int]] = []
        num_nodes = 3

        def _instrumented(index: int):
            engine = fake_engine(host=f"10.0.0.{index + 1}", port_seed=0)
            engine.__class__ = ray.actor.ActorHandle
            engine.run.remote.side_effect = lambda **kwargs: asyncio.sleep(0)

            async def _probe():
                events.append(("enter", index))
                await asyncio.sleep(0.05)
                events.append(("exit", index))
                return f"10.0.0.{index + 1}"

            engine._get_node_ip.remote.side_effect = _probe
            return engine

        actors = {rank: _instrumented(rank) for rank in range(num_nodes)}
        cell = ServerCell(
            args=make_args(num_gpus_per_node=8, sglang_dp_size=1),
            num_nodes=num_nodes,
            worker_type="regular",
            cell_id="cell-0",
            pg=(None, [], list(range(8))),
        )
        with (
            patch.object(
                server_cell_module,
                "launch_sglang_ray_actor",
                side_effect=lambda *, global_rank, **kw: actors[global_rank],
            ),
            patch.object(server_cell_module, "wait_server_healthy", new=AsyncMock()),
        ):
            await cell.start_engines(PortAllocator())

        assert [kind for kind, _ in events[:num_nodes]] == ["enter"] * num_nodes, events
