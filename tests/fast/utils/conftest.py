from argparse import Namespace
from typing import Any

import pytest

from miles.utils.external_utils.ray_job import _run_launcher_owned_job
from miles.utils.tracking_utils.base import TrackingBackend, TrackingManager


@pytest.fixture
def unavailable_ray_job_client(commands: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    from miles.utils.external_utils import ray_job

    def fail_connect(address: str) -> None:
        raise RuntimeError("Ray job service unavailable")

    monkeypatch.setattr(ray_job, "_run_launcher_owned_job", _run_launcher_owned_job)
    monkeypatch.setattr(ray_job, "JobSubmissionClient", fail_connect)


@pytest.fixture
def partially_failing_tracking_manager() -> TrackingManager:
    return TrackingManager({"working": (_TrackingResource, "enabled"), "failing": (_FailingTracking, "enabled")})


class _TrackingResource(TrackingBackend):
    def init(self, args: Namespace, *, primary: bool = True, **kwargs: Any) -> None:
        self._resources = args.resources
        self._resources.append(self)

    def log(self, metrics: dict[str, Any], step: int | None = None, **kwargs: Any) -> None:
        pass

    def finish(self) -> None:
        self._resources.remove(self)


class _FailingTracking(_TrackingResource):
    def init(self, args: Namespace, *, primary: bool = True, **kwargs: Any) -> None:
        raise RuntimeError("tracking backend unavailable")
