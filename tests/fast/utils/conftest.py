from argparse import Namespace
from typing import Any
from unittest.mock import Mock

import pytest

from miles.utils import object_store
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


@pytest.fixture
def mooncake_reader(monkeypatch: pytest.MonkeyPatch) -> Mock:
    transfer = Mock()
    monkeypatch.setattr(object_store, "_MOONCAKE_AVAILABLE", True)
    monkeypatch.setattr(
        object_store, "MooncakeDistributedStore", Mock(return_value=Mock(setup=Mock(return_value=0))), raising=False
    )
    monkeypatch.setattr(object_store, "MooncakeBundleTransfer", Mock(return_value=transfer), raising=False)
    monkeypatch.setattr(object_store, "import_ref", lambda ref: ref, raising=False)
    return transfer
