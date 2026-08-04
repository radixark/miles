from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

from miles.ray.train import actor_factory


class _RecordingRemote:
    def __init__(self, record: list[tuple[str, dict[str, Any]]], name: str, result: Any = None):
        self._record = record
        self._name = name
        self._result = result

    def remote(self, **kwargs: Any) -> Any:
        self._record.append((self._name, kwargs))
        return self._result


class _RecordingActor:
    def __init__(self, record: list[tuple[str, dict[str, Any]]], rank: int, proposal: tuple[str, int]):
        self.rank = rank
        self.propose_master_addr_and_port = _RecordingRemote(record, f"propose-{rank}", proposal)
        self.configure_master_addr_and_port = _RecordingRemote(record, f"configure-{rank}")


class _RecordingActorClass:
    def __init__(self, record: list[tuple[str, dict[str, Any]]], actors: Iterator[_RecordingActor]):
        self._record = record
        self._actors = actors

    def options(self, **kwargs: Any) -> "_RecordingActorClass":
        return self

    def remote(self, **kwargs: Any) -> _RecordingActor:
        actor = next(self._actors)
        self._record.append((f"create-{actor.rank}", kwargs))
        return actor


class TestMasterAddrIsAssignedByTheDriver:
    PROPOSAL = ("10.0.0.1", 20001)

    def _allocate(self, record: list[tuple[str, dict[str, Any]]], world_size: int) -> list[_RecordingActor]:
        actors = iter([_RecordingActor(record, rank, self.PROPOSAL) for rank in range(world_size)])
        actor_class = _RecordingActorClass(record, actors)

        args = MagicMock()
        args.train_backend = "fsdp"
        args.use_fault_tolerance = False
        args.offload_train = False
        args.offload_train_target = "cpu"

        with (
            patch.object(actor_factory.ray, "remote", side_effect=lambda **kwargs: lambda actor_impl: actor_class),
            patch.object(actor_factory.ray, "get", side_effect=lambda value: value),
            patch.object(actor_factory, "PlacementGroupSchedulingStrategy", MagicMock()),
        ):
            return actor_factory.allocate_gpus_for_actor(
                args=args,
                gpus_per_cell=world_size,
                pg=(MagicMock(), list(range(world_size)), list(range(world_size))),
                num_gpus_per_actor=0.4,
                indep_dp_store_addr=None,
                role="actor",
                cell_index=0,
            )

    def test_no_worker_is_created_after_the_master_addr_is_known(self):
        """Every worker is created before rank 0 is asked, so creation never serializes on the proposal."""
        record: list[tuple[str, dict[str, Any]]] = []

        self._allocate(record=record, world_size=3)

        first_propose = next(i for i, (name, _) in enumerate(record) if name.startswith("propose"))
        assert [name for name, _ in record[:first_propose]] == ["create-0", "create-1", "create-2"]

    def test_every_worker_is_told_the_same_master_addr(self):
        """All ranks are configured with exactly the one master addr/port that was proposed."""
        record: list[tuple[str, dict[str, Any]]] = []

        self._allocate(record=record, world_size=3)

        configured = {name: kwargs for name, kwargs in record if name.startswith("configure")}
        assert sorted(configured) == ["configure-0", "configure-1", "configure-2"]
        assert {(kwargs["master_addr"], kwargs["master_port"]) for kwargs in configured.values()} == {self.PROPOSAL}

    def test_only_rank_zero_is_asked_to_propose_the_master_addr(self):
        """The proposal is requested once, from rank 0, instead of once per rank."""
        record: list[tuple[str, dict[str, Any]]] = []

        self._allocate(record=record, world_size=3)

        assert [name for name, _ in record if name.startswith("propose")] == ["propose-0"]

    def test_no_actor_is_configured_when_the_cell_has_no_worker(self):
        """A zero-size cell asks nobody to propose and configures nobody."""
        record: list[tuple[str, dict[str, Any]]] = []

        handles = self._allocate(record=record, world_size=0)

        assert handles == []
        assert record == []
