import pytest
from tests.fast.utils.workers.fake_ray import FakeRayCluster, FakeRayModule

from miles.backends.sglang_utils.sglang_config import SglangScalingConfig
from miles.utils.args.configs.scaling import ScalingConfig


@pytest.fixture
def fake_ray_cluster(monkeypatch) -> FakeRayCluster:
    """In-process stand-in for Ray, letting the manager's whole launch pipeline run without a cluster."""
    import miles.utils.workers.ray_worker_manager as ray_worker_manager_mod

    cluster = FakeRayCluster()
    fake_ray = FakeRayModule(cluster=cluster)
    monkeypatch.setattr(ray_worker_manager_mod, "ray", fake_ray)
    return cluster


class WorkerManagerArgs(ScalingConfig):
    save_debug_event_data: str | None = None
    env_report: str = ""
    env_report_interval_seconds: float | None = None
    ci_test: bool = False
    ci_disable_config_snapshot: bool = False


def worker_manager_args(**overrides) -> WorkerManagerArgs:
    """The slice of a training run's args the worker manager reads: its logger settings and the run's scaling."""
    return WorkerManagerArgs(**{"sglang_scaling": SglangScalingConfig(groups={}), **overrides})
