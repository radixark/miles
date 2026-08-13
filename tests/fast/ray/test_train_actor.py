import os
import socket
from types import SimpleNamespace

import pytest

from miles.ray import placement_group, train_actor
from miles.ray.train_actor import TrainRayActor


class TestConstructorSignature:
    def test_positional_constructor_arguments_are_rejected(self):
        """Workers are built from a spec's kwargs, so silently shifted positional args must not construct one."""
        with pytest.raises(TypeError):
            TrainRayActor(SimpleNamespace(), 2, 1, "10.0.0.1:1234", "actor", 0)


class TestProposeMasterAddrAndPort:
    def test_the_proposal_steps_past_a_port_that_is_already_taken(self, monkeypatch: pytest.MonkeyPatch):
        """A cell rendezvouses on the proposing worker's own node, on a port no other process already holds."""
        monkeypatch.setattr(train_actor, "get_current_node_ip", lambda: "10.0.0.3")

        with socket.socket() as occupied:
            occupied.bind(("", train_actor.get_free_port(start_port=20500)))
            occupied.listen(1)
            taken_port = occupied.getsockname()[1]
            monkeypatch.setattr(train_actor.random, "randint", lambda _low, _high: taken_port)

            addr, port = TrainRayActor.__new__(TrainRayActor).propose_master_addr_and_port()

        assert addr == "10.0.0.3"
        assert port > taken_port
        with socket.socket() as probe:
            probe.bind(("", port))


class TestKillSelf:
    def test_kill_self_exits_with_a_failure_status(self, monkeypatch: pytest.MonkeyPatch):
        """A worker asked to die must leave no survivor and must not look like a clean shutdown."""
        exit_statuses: list[int] = []
        monkeypatch.setattr(train_actor.os, "_exit", exit_statuses.append)

        TrainRayActor.__new__(TrainRayActor).kill_self()

        assert exit_statuses == [1]


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


class TestTrainParallelConfigWiring:
    async def test_the_driver_passes_the_resolved_actor_config_to_the_rollout_executor(self, monkeypatch):
        """The driver resolves the actor config before handing it to the rollout executor."""
        train_parallel_config = {"dp_size": 4, "topology": {"tp_size": 2}}

        class FakeTrainerController:
            def __init__(self, *, role, **_kwargs):
                self.role = role

            async def init(self):
                return [0]

            async def get_train_parallel_config(self):
                return train_parallel_config if self.role == "actor" else {"dp_size": 99}

        class FakeRolloutExecutor:
            def __init__(self):
                self.received_config = None
                self.set_train_parallel_config = SimpleNamespace(remote=self._set_train_parallel_config)
                self.load = SimpleNamespace(remote=self._load)

            async def _set_train_parallel_config(self, config):
                assert not isinstance(config, FakeTrainerController)
                self.received_config = config

            async def _load(self, rollout_id):
                pass

        rollout_executor = FakeRolloutExecutor()
        monkeypatch.setattr(placement_group, "TrainerController", FakeTrainerController)

        await placement_group.create_training_models(
            SimpleNamespace(
                kl_coef=0.0,
                use_kl_loss=False,
                use_opd=False,
                opd_type=None,
                use_critic=True,
                start_rollout_id=None,
            ),
            inference_controller=object(),
            rollout_executor=rollout_executor,
        )

        assert rollout_executor.received_config is train_parallel_config


class TestInitRunsExactlyOnce:
    def test_a_second_init_is_refused(self):
        """A worker that already initialized is a stale process; reusing it must fail loudly, not train on."""
        actor = TrainRayActor.__new__(TrainRayActor)
        actor._init_called = True

        with pytest.raises(AssertionError, match="stale worker"):
            actor._init_common(None, "actor")
