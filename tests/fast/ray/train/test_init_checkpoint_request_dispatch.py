import pytest
import ray
from tests.fast.ray.train.conftest import get_raw_actor_handles, make_cell, make_indep_dp_info

from miles.backends.megatron_utils.checkpoint_request import MegatronCkptLoadConfig


class TestCellInitializationCheckpointDispatch:
    @pytest.mark.parametrize("healing", [False, True])
    async def test_initialization_sends_the_current_disk_request_or_peer_source_without_caching_it(
        self, healing: bool
    ) -> None:
        """Cell startup forwards disk inputs while peer healing forwards only the surviving source rank."""
        cell = make_cell(cell_index=1, actor_count=2)
        info = make_indep_dp_info(cell_index=1, alive_cell_indices=[0, 1])
        request = MegatronCkptLoadConfig.for_peer_recovery().model_copy(update={"load": "/checkpoint/current"})
        if healing:
            await cell.prepare_indep_dp_mode_healing(
                indep_dp_info=info, indep_dp_store_addr="10.0.0.9:1234", recv_ckpt_src_rank=0
            )
        else:
            await cell.init(ckpt_load_config=request, indep_dp_info=info, indep_dp_store_addr="10.0.0.9:1234")

        for handle in get_raw_actor_handles(cell):
            calls = ray.get(handle.get_calls.remote())
            [kwargs] = [call[2] for call in calls if call[0] == "init"]
            assert kwargs["ckpt_load_config"] == (None if healing else request)
            assert kwargs["recv_ckpt_src_rank"] == (0 if healing else None)
            assert kwargs["indep_dp_info"] == info
        assert cell.is_alive
        assert not {"ckpt_load_config", "_ckpt_load_config"} & vars(cell).keys()
        assert "ckpt_load_config" not in vars(cell.args)
