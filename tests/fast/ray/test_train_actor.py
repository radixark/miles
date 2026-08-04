import os
from types import SimpleNamespace

import pytest

from tests.fast.ray.fake_rollout_executor import FakeRolloutExecutor

from miles.ray import train_actor as train_actor_module
from miles.ray.train_actor import TrainRayActor

_TRAIN_PARALLEL_CONFIG = {"dp_size": 2, "cp_size": 1}


def _make_actor(*, rank: int) -> TrainRayActor:
    actor = object.__new__(TrainRayActor)
    actor.args = SimpleNamespace(rank=rank)
    actor.train_parallel_config = _TRAIN_PARALLEL_CONFIG
    return actor


class TestSetRolloutExecutor:
    def test_rank_zero_publishes_the_train_parallel_config(self, monkeypatch) -> None:
        """On rank zero the executor handle is stored and the train parallel config is pushed to it."""
        awaited_refs: list[str] = []
        monkeypatch.setattr(train_actor_module, "ray", SimpleNamespace(get=awaited_refs.append))
        actor = _make_actor(rank=0)
        executor = FakeRolloutExecutor()

        actor.set_rollout_executor(executor)

        assert actor.rollout_executor is executor
        assert executor.set_train_parallel_config.calls == [(_TRAIN_PARALLEL_CONFIG,)]
        assert awaited_refs == ["object-ref-1"]

    def test_only_rank_zero_configures_the_rollout_executor(self, monkeypatch) -> None:
        """A nonzero rank stores the executor handle but issues no configuration RPC."""
        awaited_refs: list[str] = []
        monkeypatch.setattr(train_actor_module, "ray", SimpleNamespace(get=awaited_refs.append))
        actor = _make_actor(rank=1)
        executor = FakeRolloutExecutor()

        actor.set_rollout_executor(executor)

        assert actor.rollout_executor is executor
        assert executor.set_train_parallel_config.calls == []
        assert awaited_refs == []


class TestConfigureMasterAddrAndPort:
    def _make_actor(self) -> TrainRayActor:
        return TrainRayActor.__new__(TrainRayActor)

    def test_the_master_addr_and_port_land_in_the_environment(self, monkeypatch: pytest.MonkeyPatch):
        """The driver-assigned addr/port must reach the env vars that torch's env:// init reads."""
        monkeypatch.delenv("MASTER_ADDR", raising=False)
        monkeypatch.delenv("MASTER_PORT", raising=False)

        self._make_actor().configure_master_addr_and_port(master_addr="10.0.0.1", master_port=20001)

        assert os.environ["MASTER_ADDR"] == "10.0.0.1"
        assert os.environ["MASTER_PORT"] == "20001"

    def test_a_stale_master_addr_and_port_are_overwritten(self, monkeypatch: pytest.MonkeyPatch):
        """A worker inheriting another run's env must end up on the addr/port the driver assigned."""
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "1")

        self._make_actor().configure_master_addr_and_port(master_addr="10.0.0.2", master_port=20002)

        assert os.environ["MASTER_ADDR"] == "10.0.0.2"
        assert os.environ["MASTER_PORT"] == "20002"
