"""Unit tests for NodeAgentRegistry (P0 item 3)."""

from __future__ import annotations

from unittest.mock import MagicMock

from miles.utils.ft.controller.node_agents.registry import NodeAgentRegistry


class TestRegister:
    def test_register_without_metadata(self) -> None:
        registry = NodeAgentRegistry()
        agent = MagicMock()
        registry.register(node_id="node-0", agent=agent)

        assert registry.get("node-0") is agent
        assert registry.get_metadata("node-0") is None

    def test_register_with_metadata(self) -> None:
        registry = NodeAgentRegistry()
        agent = MagicMock()
        metadata = {"ip": "10.0.0.1", "gpu_count": "8"}
        registry.register(node_id="node-0", agent=agent, metadata=metadata)

        assert registry.get("node-0") is agent
        assert registry.get_metadata("node-0") == metadata

    def test_register_overwrites_existing_agent(self) -> None:
        registry = NodeAgentRegistry()
        agent1 = MagicMock()
        agent2 = MagicMock()
        registry.register(node_id="node-0", agent=agent1)
        registry.register(node_id="node-0", agent=agent2)

        assert registry.get("node-0") is agent2


class TestGet:
    def test_get_returns_none_for_unknown_node_id(self) -> None:
        registry = NodeAgentRegistry()
        assert registry.get("nonexistent") is None

    def test_get_returns_agent_for_registered_node(self) -> None:
        registry = NodeAgentRegistry()
        agent = MagicMock()
        registry.register(node_id="node-0", agent=agent)
        assert registry.get("node-0") is agent


class TestGetAll:
    def test_get_all_returns_copy(self) -> None:
        """Mutating the returned dict must not affect the registry."""
        registry = NodeAgentRegistry()
        agent = MagicMock()
        registry.register(node_id="node-0", agent=agent)

        result = registry.get_all()
        result["node-99"] = MagicMock()

        assert "node-99" not in registry.get_all()

    def test_get_all_returns_all_registered(self) -> None:
        registry = NodeAgentRegistry()
        agents = {f"node-{i}": MagicMock() for i in range(3)}
        for node_id, agent in agents.items():
            registry.register(node_id=node_id, agent=agent)

        result = registry.get_all()
        assert set(result.keys()) == set(agents.keys())
        for node_id, agent in agents.items():
            assert result[node_id] is agent


class TestRegisteredNodeIds:
    def test_empty_registry(self) -> None:
        registry = NodeAgentRegistry()
        assert registry.registered_node_ids() == set()

    def test_returns_correct_set(self) -> None:
        registry = NodeAgentRegistry()
        for i in range(3):
            registry.register(node_id=f"node-{i}", agent=MagicMock())

        assert registry.registered_node_ids() == {"node-0", "node-1", "node-2"}


class TestMetadata:
    def test_get_metadata_returns_none_when_no_metadata_registered(self) -> None:
        registry = NodeAgentRegistry()
        registry.register(node_id="node-0", agent=MagicMock())
        assert registry.get_metadata("node-0") is None

    def test_get_metadata_returns_none_for_unknown_node(self) -> None:
        registry = NodeAgentRegistry()
        assert registry.get_metadata("nonexistent") is None

    def test_all_metadata_property(self) -> None:
        registry = NodeAgentRegistry()
        registry.register(node_id="node-0", agent=MagicMock(), metadata={"role": "worker"})
        registry.register(node_id="node-1", agent=MagicMock(), metadata={"role": "driver"})
        registry.register(node_id="node-2", agent=MagicMock())

        all_meta = registry.all_metadata
        assert all_meta == {
            "node-0": {"role": "worker"},
            "node-1": {"role": "driver"},
        }


class TestUnregister:
    def test_unregister_removes_agent_and_metadata(self) -> None:
        """Previously there was no way to remove a node agent from the registry.
        After node eviction, stale agents would accumulate and interfere with
        diagnostics and scrape loops."""
        registry = NodeAgentRegistry()
        registry.register(node_id="node-0", agent=MagicMock(), metadata={"ip": "10.0.0.1"})
        registry.register(node_id="node-1", agent=MagicMock(), metadata={"ip": "10.0.0.2"})

        registry.unregister("node-0")

        assert registry.get("node-0") is None
        assert registry.get_metadata("node-0") is None
        assert registry.registered_node_ids() == {"node-1"}

    def test_unregister_nonexistent_node_is_no_op(self) -> None:
        registry = NodeAgentRegistry()
        registry.unregister("ghost")
        assert registry.registered_node_ids() == set()


class TestClear:
    def test_clear_removes_all_agents_and_metadata(self) -> None:
        """Previously there was no bulk reset for the registry. After multiple
        recovery cycles, stale entries would accumulate."""
        registry = NodeAgentRegistry()
        for i in range(3):
            registry.register(node_id=f"node-{i}", agent=MagicMock(), metadata={"idx": str(i)})

        registry.clear()

        assert registry.get_all() == {}
        assert registry.all_metadata == {}
        assert registry.registered_node_ids() == set()


class TestAllMetadataReturnsSnapshot:
    """all_metadata used to return self._metadata directly. If a caller iterated
    the dict (e.g. for nid, meta in controller.node_metadata.items()) while
    register_node_agent was scheduled by Ray and modified _metadata, the iteration
    would crash with 'RuntimeError: dictionary changed size during iteration'.
    Fix: return dict(self._metadata) — a shallow copy."""

    def test_all_metadata_returns_copy_not_reference(self) -> None:
        registry = NodeAgentRegistry()
        registry.register(node_id="node-0", agent=MagicMock(), metadata={"k": "v"})

        snapshot = registry.all_metadata
        assert snapshot is not registry._metadata
        assert snapshot == registry._metadata

    def test_mutating_all_metadata_does_not_affect_registry(self) -> None:
        registry = NodeAgentRegistry()
        registry.register(node_id="node-0", agent=MagicMock(), metadata={"k": "v"})

        snapshot = registry.all_metadata
        snapshot["node-99"] = {"injected": "true"}

        assert "node-99" not in registry.all_metadata

    def test_register_during_iteration_of_snapshot_does_not_crash(self) -> None:
        """Simulate the race: get snapshot, then register new node, then iterate."""
        registry = NodeAgentRegistry()
        registry.register(node_id="node-0", agent=MagicMock(), metadata={"a": "1"})

        snapshot = registry.all_metadata
        registry.register(node_id="node-1", agent=MagicMock(), metadata={"b": "2"})

        items = list(snapshot.items())
        assert len(items) == 1
        assert items[0] == ("node-0", {"a": "1"})
