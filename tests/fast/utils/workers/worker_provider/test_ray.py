from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from miles.utils.workers.worker_provider.ray import RayWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort


@dataclass
class _FakeRemoteMethod:
    answers: list[HostAndPort]
    requested_names: list[str] = field(default_factory=list)

    def remote(self, worker_name: str) -> Any:
        self.requested_names.append(worker_name)
        return _resolved(self.answers[len(self.requested_names) - 1])


@dataclass
class _FakeManagerHandle:
    get_worker_addr: _FakeRemoteMethod


async def _resolved(value: HostAndPort) -> HostAndPort:
    return value


def _make_handle(*answers: HostAndPort) -> _FakeManagerHandle:
    return _FakeManagerHandle(get_worker_addr=_FakeRemoteMethod(answers=list(answers)))


class TestRayWorkerProviderCreate:
    async def test_looks_addresses_up_through_the_named_manager_actor(self, monkeypatch: pytest.MonkeyPatch):
        """The provider finds the manager by its well-known actor name and asks it for the address."""
        import miles.utils.workers.ray_worker_manager as ray_worker_manager_mod

        handle = _make_handle(HostAndPort(host="10.0.0.7", port=15000))
        looked_up: list[str] = []

        class _FakeRay:
            @staticmethod
            def get_actor(name: str) -> _FakeManagerHandle:
                looked_up.append(name)
                return handle

        monkeypatch.setattr(ray_worker_manager_mod, "ray", _FakeRay)

        provider = RayWorkerProvider.create()
        addr = await provider.get_addr(worker_name="router-0-0")

        assert looked_up == ["ray_worker_manager"]
        assert handle.get_worker_addr.requested_names == ["router-0-0"]
        assert addr == HostAndPort(host="10.0.0.7", port=15000)


class TestRayWorkerProviderGetAddr:
    async def test_every_lookup_asks_the_manager_again(self):
        """Addresses are never cached, so a relaunched worker is not answered with a stale endpoint."""
        handle = _make_handle(
            HostAndPort(host="10.0.0.7", port=15000),
            HostAndPort(host="10.0.0.7", port=15001),
        )
        provider = RayWorkerProvider(worker_manager_handle=handle)

        first = await provider.get_addr(worker_name="router-0-0")
        second = await provider.get_addr(worker_name="router-0-0")

        assert (first.port, second.port) == (15000, 15001)
        assert handle.get_worker_addr.requested_names == ["router-0-0", "router-0-0"]
