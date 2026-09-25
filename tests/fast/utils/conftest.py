from argparse import Namespace
from typing import Any

import pytest

from miles.utils.tracking_utils.base import TrackingBackend, TrackingManager


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
