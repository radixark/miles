import os

import pytest

from tests.fast.fixtures.generation_fixtures import generation_env
from tests.fast.fixtures.rollout_fixtures import rollout_env

_ = rollout_env, generation_env


@pytest.fixture(autouse=True)
def no_env_reporting(monkeypatch):
    """Constructing a worker configures its logger, which in a real process starts a thread that
    shells out to pip and git; tests exercise that reporter directly instead."""
    monkeypatch.setattr("miles.utils.logging_utils.start_env_reporting", lambda args: None)


@pytest.fixture(autouse=True)
def clear_legacy_rollout_gate(monkeypatch):
    # an ambient value changes which arguments the parser registers
    monkeypatch.delenv("MILES_USE_LEGACY_ROLLOUT_V1", raising=False)


@pytest.fixture(scope="session")
def ray_local_mode():
    """Session-scoped Ray init. On CI ``RAY_ADDRESS`` points at an existing
    cluster, so we connect without ``num_cpus`` (Ray rejects it when joining).
    Tests that only need pure-Python helpers should not depend on this."""
    import ray

    if not ray.is_initialized():
        kwargs: dict = dict(
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
        )
        if not os.environ.get("RAY_ADDRESS"):
            # address="local" forces a fresh cluster: with no address, ray.init
            # auto-connects to any leaked local cluster (via /tmp/ray), and
            # connecting with num_cpus/num_gpus set is a hard ValueError.
            kwargs["address"] = "local"
            kwargs["num_cpus"] = 32
            # Logical GPU resource so real_ray placement-group tests (engines
            # are mocked via MockSGLangEngine; no real GPU is used) can satisfy
            # their {"GPU": 0.2} bundles on GPU-less CPU CI runners.
            kwargs["num_gpus"] = 8
        ray.init(**kwargs)
    yield
    # Don't shut down — other session-scoped suites may share this cluster.
