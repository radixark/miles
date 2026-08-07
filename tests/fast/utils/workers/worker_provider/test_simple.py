import asyncio

import pytest

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_provider.simple import SimpleWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort


class FakeController:
    def health(self) -> int:
        return 1


def _provider() -> SimpleWorkerProvider:
    return SimpleWorkerProvider(
        addrs={
            "trainer-controller-0-0": {
                "primary": HostAndPort(host="controller.rl.svc", port=7000),
                "rpc": HostAndPort(host="controller.rl.svc", port=8000),
            }
        },
        cells={"trainer-controller-0": ["trainer-controller-0-0"]},
        pool_ids={"trainer-controller-0": "trainer-controller"},
        worker_classes={"trainer-controller": f"{__name__}.FakeController"},
    )


class TestAddresses:
    def test_answers_from_the_address_book_it_was_handed(self):
        """A statically addressed component needs no cluster at all, only the address the chart rendered."""
        assert asyncio.run(_provider().get_addrs("trainer-controller-0-0"))["primary"].port == 7000

    def test_refuses_a_worker_nobody_declared(self):
        """Guessing an address would send calls to whatever happens to answer there."""
        with pytest.raises(AssertionError, match="not in the address book"):
            asyncio.run(_provider().get_addrs("trainer-controller-9-9"))


class TestHandles:
    def test_builds_an_rpc_handle_at_the_declared_rpc_port(self):
        """The caller drives the component over rpc, which is only reachable at the port it declared."""
        handle = _provider().get_handle("trainer-controller-0-0")

        assert isinstance(handle, RpcWorkerHandle)
        assert handle._transport._server_url == "http://controller.rl.svc:8000"

    def test_worker_infos_carry_the_handles_of_a_cell(self):
        """A consumer that owns a static cell drives it exactly like an observed one."""
        infos = _provider().get_worker_infos(cell_ids=["trainer-controller-0"])[0]

        assert [info.name for info in infos] == ["trainer-controller-0-0"]
        assert isinstance(infos[0].handle, RpcWorkerHandle)
