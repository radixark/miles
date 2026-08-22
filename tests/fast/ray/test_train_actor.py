import os
import socket
from types import SimpleNamespace

import pytest

from miles.ray import train_actor
from miles.ray.train_actor import TrainRayActor
from miles.utils.init_once import InitOnce
from miles.utils.workers.env_vars import CELL_INDEX_ENV_VAR, SUBPROCESS_INDEX_ENV_VAR


def _inited_guard() -> InitOnce:
    guard = InitOnce("TrainRayActor")
    with guard.guarding():
        pass
    return guard


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


def _actor_with(guard: InitOnce) -> TrainRayActor:
    actor = TrainRayActor.__new__(TrainRayActor)
    actor._init_once = guard
    return actor


class TestInitRunsExactlyOnce:
    def test_a_second_init_is_refused(self):
        """A worker that already initialized is a stale process; reusing it must fail loudly, not train on."""
        actor = _actor_with(_inited_guard())

        with pytest.raises(AssertionError, match="stale worker"):
            actor._init_common(None, "actor")

    def test_a_worker_that_never_ran_init_reports_itself_uninitialized(self):
        """A restarted script asks a worker it found running whether to initialize it or to resume it."""
        assert _actor_with(InitOnce("TrainRayActor")).is_initialized() is False

    def test_a_worker_that_ran_init_reports_itself_initialized(self):
        """The take-over path has to see the worker the previous script built as built."""
        assert _actor_with(_inited_guard()).is_initialized() is True


class TestTheLocalGpuIsFoundWithoutRay:
    def test_a_supervised_rank_reads_its_index_rather_than_asking_ray(self, monkeypatch):
        """A served worker is a plain process, not an actor, so ray owns no assignment and answers []."""
        monkeypatch.setenv(CELL_INDEX_ENV_VAR, "0")
        monkeypatch.setenv(SUBPROCESS_INDEX_ENV_VAR, "3")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(train_actor.ray, "get_gpu_ids", lambda: [])

        assert train_actor.get_local_gpu_id() == 3

    def test_a_pod_running_one_worker_needs_no_supervisor_to_know_its_card(self, monkeypatch):
        """A cell one worker wide is launched without the supervisor, so the index it would set is absent."""
        monkeypatch.setenv(CELL_INDEX_ENV_VAR, "0")
        monkeypatch.delenv(SUBPROCESS_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(train_actor.ray, "get_gpu_ids", lambda: [])

        assert train_actor.get_local_gpu_id() == 0

    def test_a_ray_placed_actor_still_reads_its_assignment_from_ray(self, monkeypatch):
        """Every existing run takes this path, where ray does own the assignment."""
        monkeypatch.delenv(CELL_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv(SUBPROCESS_INDEX_ENV_VAR, raising=False)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
        monkeypatch.setattr(train_actor.ray, "get_gpu_ids", lambda: [6])

        assert train_actor.get_local_gpu_id() == 2

    def test_a_ray_actor_without_a_visible_device_list_still_reads_its_assignment(self, monkeypatch):
        """The other ray shape: no mask, so the assignment is the id itself."""
        monkeypatch.delenv(CELL_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv(SUBPROCESS_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(train_actor.ray, "get_gpu_ids", lambda: [5])

        assert train_actor.get_local_gpu_id() == 5
