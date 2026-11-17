from dataclasses import dataclass, field

import pytest

pytest.importorskip("monarch")

from miles.utils.object_store_config import MOONCAKE_MASTER_ADDRESS_KEY
from miles.utils.workers.monarch import hosts
from miles.utils.workers.monarch.env_vars import MONARCH_JOB_STATE_ENV_VAR
from miles.utils.workers.monarch.hosts import (
    WORKER_MESH_NAME,
    acquire_worker_hosts,
    check_mooncake_master_reaches_every_host,
)


@dataclass
class FakeHostMesh:
    name: str
    initialized_count: int = 0
    error: Exception | None = None

    @property
    def initialized(self) -> "FakeHostMesh":
        return self

    def get(self) -> None:
        self.initialized_count += 1
        if self.error is not None:
            raise self.error


@dataclass
class FakeJob:
    meshes: dict[str, FakeHostMesh]
    loaded_from: list[str] = field(default_factory=list)

    def load(self, path: str) -> "FakeJob":
        self.loaded_from.append(path)
        return self

    def state(self):
        return type("JobState", (), self.meshes)


@pytest.fixture(autouse=True)
def _no_hosts_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MONARCH_JOB_STATE_ENV_VAR, raising=False)


class TestAcquireWorkerHosts:
    def test_a_run_with_no_hosts_named_stays_on_the_driver_s_host(self):
        assert acquire_worker_hosts() is None

    def test_a_run_in_a_monarch_job_attaches_to_the_job_s_worker_mesh(self, monkeypatch: pytest.MonkeyPatch):
        """The submitted job already started a worker loop on every host; the driver only connects to them."""
        workers = FakeHostMesh(WORKER_MESH_NAME)
        job = FakeJob(meshes={WORKER_MESH_NAME: workers})
        monkeypatch.setattr(hosts, "job_load", job.load)
        monkeypatch.setenv(MONARCH_JOB_STATE_ENV_VAR, "/shared/job/.monarch/job_state.pkl")

        assert acquire_worker_hosts() is workers
        assert job.loaded_from == ["/shared/job/.monarch/job_state.pkl"]
        assert workers.initialized_count == 1

    def test_hosts_that_never_answer_fail_with_where_to_look(self, monkeypatch: pytest.MonkeyPatch):
        """The bare attach timeout says nothing of the worker loops' logs, their port or the dual-stack trap."""
        workers = FakeHostMesh(WORKER_MESH_NAME, error=TimeoutError("attach timed out"))
        monkeypatch.setattr(hosts, "job_load", FakeJob(meshes={WORKER_MESH_NAME: workers}).load)
        monkeypatch.setenv(MONARCH_JOB_STATE_ENV_VAR, "/shared/job/.monarch/job_state.pkl")

        with pytest.raises(RuntimeError, match="never answered.*slurm_<job id>_.*--bind-to ::"):
            acquire_worker_hosts()


class TestTheMooncakeMasterOfAMultiHostRun:
    @pytest.mark.parametrize("master", ["127.0.0.1:50051", "localhost:50051", "[::1]:50051"])
    def test_a_loopback_master_is_refused(self, master: str):
        """Workers on the other hosts would fail to reach it, each with a bare connection refused."""
        with pytest.raises(AssertionError, match="cannot reach the mooncake master"):
            check_mooncake_master_reaches_every_host({MOONCAKE_MASTER_ADDRESS_KEY: master}, num_hosts=2)

    def test_a_master_on_the_head_node_s_address_passes(self):
        check_mooncake_master_reaches_every_host({MOONCAKE_MASTER_ADDRESS_KEY: "10.0.0.1:50051"}, num_hosts=2)

    def test_a_one_host_run_may_keep_its_loopback_master(self):
        """A one-node slurm job is the usual way to get a gpu node, and its master is on that node."""
        check_mooncake_master_reaches_every_host({MOONCAKE_MASTER_ADDRESS_KEY: "127.0.0.1:50051"}, num_hosts=1)
