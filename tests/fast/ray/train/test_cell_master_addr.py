import pytest
import ray
from tests.fast.ray.train.conftest import make_cell, make_indep_dp_info

pytestmark = pytest.mark.asyncio


def _calls_of(cell, method: str) -> list:
    return [
        [call for call in ray.get(handle.get_calls.remote()) if call[0] == method]
        for handle in cell._get_actor_handles()
    ]


class TestMasterAddrConfiguration:
    async def test_every_worker_is_told_the_master_address_before_init(self):
        """Workers rendezvous on the address the worker manager allocated for the cell."""
        cell = make_cell(0, actor_count=2)

        await cell.init(indep_dp_info=make_indep_dp_info(quorum_id=0))

        for [call] in _calls_of(cell, "configure_master_addr_and_port"):
            assert call[2] == {"master_addr": "10.0.0.1", "master_port": 20000}

    async def test_the_master_address_is_configured_before_the_process_group_is_built(self):
        """A rank that runs init first would build the process group without the address."""
        cell = make_cell(0, actor_count=1)

        await cell.init(indep_dp_info=make_indep_dp_info(quorum_id=0))

        methods = [call[0] for call in ray.get(cell._get_actor_handles()[0].get_calls.remote())]
        assert methods.index("configure_master_addr_and_port") < methods.index("init")
