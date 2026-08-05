from __future__ import annotations

from tests.fast.ray.rollout.conftest import fake_engine

from miles.utils.workers.addr_allocator import BASE_PORT, TRAIN_MASTER_PORT_RANGE, PortAllocator


class TestPortBands:
    def test_the_base_port_clears_rays_worker_port_range(self):
        """Ray hands its own workers 10002-19999, and a reserved port stays unbound long
        enough for ray to give it away, so the allocator must start above that range."""
        assert BASE_PORT > 19999
        assert BASE_PORT < 32768

    def test_the_trainer_master_port_band_is_clear_of_the_allocator(self):
        """The trainer probes for a free master port instead of reserving one, so its band
        must stay clear of the ports this allocator hands out before anyone binds them."""
        low, high = TRAIN_MASTER_PORT_RANGE
        assert low > BASE_PORT
        assert high < 32768


class TestPortAllocator:
    def test_a_fresh_allocator_has_no_cursors(self):
        """A brand new allocator starts with no per-node cursors."""
        c = PortAllocator()
        assert c._next_port_of_ip == {}

    def test_alloc_advances_the_cursor_of_its_node(self, patch_ray_get):
        """Two allocations on the same node must hand out non-overlapping ports."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        first = cursors.alloc(engine, node_ip="10.0.0.1")
        second = cursors.alloc(engine, node_ip="10.0.0.1")
        assert second > first
        assert cursors._next_port_of_ip["10.0.0.1"] == second + 1

    def test_alloc_starts_from_the_base_port_on_an_unseen_node(self, patch_ray_get):
        """A node with no cursor yet starts at the base port, away from ray's range."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        assert cursors.alloc(engine, node_ip="10.0.0.1") == BASE_PORT

    def test_alloc_consecutive_reserves_a_whole_block(self, patch_ray_get):
        """A consecutive=N allocation must move this node's cursor past the entire block."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        first = cursors.alloc(engine, node_ip="10.0.0.1", consecutive=5)
        assert cursors._next_port_of_ip["10.0.0.1"] == first + 5

    def test_alloc_tracks_nodes_independently(self, patch_ray_get):
        """Each node ip owns its own cursor."""
        cursors = PortAllocator()
        engine_a = fake_engine(host="10.0.0.1", port_seed=0)
        engine_b = fake_engine(host="10.0.0.2", port_seed=0)
        cursors.alloc(engine_a, node_ip="10.0.0.1")
        cursors.alloc(engine_b, node_ip="10.0.0.2")
        assert set(cursors._next_port_of_ip.keys()) == {"10.0.0.1", "10.0.0.2"}
