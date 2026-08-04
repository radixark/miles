import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.megatron_utils.ft import indep_dp
from miles.utils.ft_utils.indep_dp import IndepDPInfo

_LOGGER_NAME = "miles.backends.megatron_utils.ft.indep_dp"


def _indep_dp_messages(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == _LOGGER_NAME]


class FakeTorchftProcessGroup:
    def __init__(self, timeout=None) -> None:
        self._replica_id = ""
        self._rank = 0
        self.configure_kwargs: dict | None = None

    def configure(self, **kwargs) -> None:
        self.configure_kwargs = kwargs
        self._replica_id = kwargs["replica_id"]
        self._rank = kwargs["rank"]

    def size(self) -> int:
        assert self.configure_kwargs is not None
        return self.configure_kwargs["world_size"]

    def shutdown(self) -> None:
        pass


class FakeCrossCellPGUtil:
    def __init__(self, *, all_reduce_error: Exception | None = None) -> None:
        self.all_reduce_error = all_reduce_error
        self.reduced_tensors: list[torch.Tensor] = []

    def all_reduce(self, tensor: torch.Tensor, group, op) -> None:
        if self.all_reduce_error is not None:
            raise self.all_reduce_error
        self.reduced_tensors.append(tensor)


def _make_model_chunk() -> SimpleNamespace:
    bucket = SimpleNamespace(grad_data=torch.full((4,), 2.5))
    return SimpleNamespace(bucket_groups=[SimpleNamespace(buckets=[bucket])], expert_parallel_bucket_groups=[])


def _make_parallel_state(pg) -> SimpleNamespace:
    return SimpleNamespace(
        intra_dp=SimpleNamespace(size=1),
        indep_dp=SimpleNamespace(rank=1, size=3, group=pg, debug_info={"quorum": 7}),
    )


class TestCreateIndepDpGroup:
    @pytest.fixture()
    def fake_torchft(self, monkeypatch):
        module = types.ModuleType("torchft.process_group")
        module.ProcessGroupNCCL = FakeTorchftProcessGroup
        module.ProcessGroupGloo = FakeTorchftProcessGroup
        monkeypatch.setitem(sys.modules, "torchft", types.ModuleType("torchft"))
        monkeypatch.setitem(sys.modules, "torchft.process_group", module)
        return module

    def test_creating_a_cross_cell_group_emits_an_ft_tagged_create_record(self, fake_torchft, caplog) -> None:
        """The create_pg record stays discoverable by the ft structured-log tag."""
        info = IndepDPInfo(
            cell_index=1, num_cells=3, alive_rank=1, alive_size=2, quorum_id=7, alive_cell_indices=[0, 1]
        )

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            group_info = indep_dp.create_indep_dp_group(
                store_addr="tcp://store:1234",
                indep_dp_info=info,
                megatron_rank=5,
                megatron_world_size=8,
            )

        messages = _indep_dp_messages(caplog)
        assert group_info.rank == 1
        assert len(messages) == 1
        assert messages[0].startswith("ft ")
        assert "op=create_pg" in messages[0]
        assert "quorum=7" in messages[0]


class TestReconfigureIndepDpGroup:
    def test_reconfigure_emits_ft_tagged_start_and_end_records(self, caplog) -> None:
        """Both reconfigure records stay discoverable by the ft structured-log tag."""
        old_group = MagicMock()
        parallel_state = SimpleNamespace(indep_dp=SimpleNamespace(group=old_group, gloo_group=None))
        info = IndepDPInfo(cell_index=2, num_cells=3, alive_rank=0, alive_size=1, quorum_id=5, alive_cell_indices=[2])

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            indep_dp.reconfigure_indep_dp_group(
                parallel_state=parallel_state,
                store_addr="tcp://store:1234",
                indep_dp_info=info,
                megatron_rank=5,
                megatron_world_size=8,
            )

        messages = _indep_dp_messages(caplog)
        old_group.shutdown.assert_called_once()
        assert all(message.startswith("ft ") for message in messages)
        assert "op=reconfig phase=start" in messages[0]
        assert "quorum_to=5" in messages[0]
        assert "op=reconfig phase=end" in messages[1]
        assert "quorum=5" in messages[1]


class TestAllreduceGradsAndLossesAcrossReplicas:
    @pytest.fixture()
    def megatron_env(self):
        with (
            patch.object(indep_dp, "mpu") as mock_mpu,
            patch.object(indep_dp, "get_gloo_group", return_value=None),
            patch.object(indep_dp, "collective_bool_and", side_effect=lambda value, group: value),
        ):
            mock_mpu.is_pipeline_last_stage.return_value = False
            yield mock_mpu

    @staticmethod
    def _run(pg, util) -> tuple[bool, dict[str, float]]:
        args = SimpleNamespace(calculate_per_token_loss=False)
        with patch.object(indep_dp.GeneralPGUtil, "create", return_value=util):
            return indep_dp.allreduce_grads_and_losses_across_replicas(
                args, [_make_model_chunk()], _make_parallel_state(pg), losses_reduced=[]
            )

    def test_a_successful_allreduce_emits_ft_tagged_start_and_end_records(self, megatron_env, caplog) -> None:
        """The happy-path cross-cell records stay discoverable by the ft structured-log tag."""
        pg = SimpleNamespace(errored=lambda: None)
        util = FakeCrossCellPGUtil()

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            consensus, loss_reduced = self._run(pg, util)

        messages = _indep_dp_messages(caplog)
        assert consensus is True
        assert loss_reduced == {}
        assert len(util.reduced_tensors) == 1
        assert all(message.startswith("ft ") for message in messages)
        assert "op=cross_cell phase=start kind=grad_allreduce" in messages[0]
        assert "op=cross_cell phase=end kind=grad_allreduce" in messages[1]
        assert "this_rank_ok=true consensus_ok=true" in messages[1]

    def test_a_raising_allreduce_emits_an_ft_tagged_fail_record(self, megatron_env, caplog) -> None:
        """A synchronous collective failure is reported as an ft-tagged fail record and discards the step."""
        pg = SimpleNamespace(errored=lambda: None)
        util = FakeCrossCellPGUtil(all_reduce_error=RuntimeError("NCCL communicator was aborted on rank 2"))

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            consensus, _loss_reduced = self._run(pg, util)

        messages = _indep_dp_messages(caplog)
        assert consensus is False
        fail_messages = [message for message in messages if "phase=fail" in message]
        assert len(fail_messages) == 1
        assert fail_messages[0].startswith("ft ")
        assert "kind=grad_allreduce" in fail_messages[0]
        assert "this_rank_ok=false consensus_ok=false" in messages[-1]

    def test_an_asynchronously_errored_group_emits_an_ft_tagged_async_error_record(self, megatron_env, caplog) -> None:
        """An error surfacing only via pg.errored() is reported as an ft-tagged async_error record."""
        pg = SimpleNamespace(errored=lambda: RuntimeError("peer 2 left the quorum"))
        util = FakeCrossCellPGUtil()

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            consensus, _loss_reduced = self._run(pg, util)

        messages = _indep_dp_messages(caplog)
        assert consensus is False
        async_messages = [message for message in messages if "phase=async_error" in message]
        assert len(async_messages) == 1
        assert async_messages[0].startswith("ft ")
        assert "kind=grad_allreduce" in async_messages[0]
        assert "peer 2 left the quorum" in async_messages[0]
