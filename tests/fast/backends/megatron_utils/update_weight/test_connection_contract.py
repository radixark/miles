from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
    UpdateWeightFromDistributed,
)
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta import UpdateWeightFromDiskDelta
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p import UpdateWeightP2P
from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor
from miles.backends.training_utils.conn_status import ConnStatusManager

_BROADCAST_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast"
_P2P_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p"
_TENSOR_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_tensor"


def _build_broadcast_updater(tmp_path: Path) -> UpdateWeightFromDistributed:
    parallel_state = SimpleNamespace(
        pp=SimpleNamespace(size=1, rank=0),
        tp=SimpleNamespace(rank=0),
        intra_dp_cp=SimpleNamespace(rank=0),
    )
    with (
        patch(f"{_BROADCAST_MODULE}.get_parallel_state", return_value=parallel_state),
        patch.object(UpdateWeightFromDistributed, "_init_lora"),
    ):
        return UpdateWeightFromDistributed(
            Namespace(),
            [],
            lambda: {},
            model_name="test-model",
            quantization_config=None,
        )


def _build_disk_delta_updater(tmp_path: Path) -> UpdateWeightFromDiskDelta:
    args = Namespace(
        update_weight_disk_dir=str(tmp_path / "delta"),
        update_weight_delta_encoding="xor",
        update_weight_delta_checksum="xxh3",
        custom_update_weight_post_write_path=None,
    )
    with patch.object(UpdateWeightFromDiskDelta, "_init_lora"):
        return UpdateWeightFromDiskDelta(
            args,
            [],
            lambda: {},
            model_name="test-model",
            quantization_config=None,
        )


def _build_p2p_updater(tmp_path: Path) -> UpdateWeightP2P:
    with (
        patch(f"{_P2P_MODULE}.dist") as dist_mock,
        patch(f"{_P2P_MODULE}.get_gloo_group", return_value=MagicMock(name="gloo_group")),
        patch(f"{_P2P_MODULE}.RemoteTransferPlan"),
        patch(f"{_P2P_MODULE}.P2PTransferManager"),
    ):
        dist_mock.get_rank.return_value = 0
        return UpdateWeightP2P(
            Namespace(),
            [],
            lambda: {},
            model_name="test-model",
            quantization_config=None,
        )


def _build_tensor_updater(tmp_path: Path) -> UpdateWeightFromTensor:
    with (
        patch(f"{_TENSOR_MODULE}.dist") as dist_mock,
        patch(f"{_TENSOR_MODULE}.HfWeightIteratorBase") as hf_weight_iterator_base,
    ):
        dist_mock.get_world_size.return_value = 8
        dist_mock.get_rank.return_value = 0
        hf_weight_iterator_base.create.return_value = MagicMock(name="hf_weight_iterator")
        return UpdateWeightFromTensor(
            Namespace(rollout_num_gpus_per_engine=4),
            [],
            lambda: {},
            model_name="test-model",
            quantization_config=None,
        )


class TestWeightUpdaterConnectionContract:
    @pytest.mark.parametrize(
        "build_updater",
        [_build_broadcast_updater, _build_disk_delta_updater, _build_p2p_updater, _build_tensor_updater],
        ids=["broadcast", "disk_delta", "p2p", "tensor"],
    )
    def test_every_megatron_weight_updater_exposes_the_connection_status_contract(
        self, build_updater: Callable[[Path], object], tmp_path: Path
    ) -> None:
        """The actor unconditionally drives reconnection through conn_status, so every updater it can build must own a working manager."""
        updater = build_updater(tmp_path)
        snapshot_cell_id_to_hashes = {"cell-0": "hash-0"}

        assert isinstance(updater.conn_status, ConnStatusManager)
        assert updater.conn_status.needs_reconnect(snapshot_cell_id_to_hashes) is True
        updater.conn_status.mark_reconnected(snapshot_cell_id_to_hashes)
        assert updater.conn_status.needs_reconnect(snapshot_cell_id_to_hashes) is False
        assert updater.conn_status.needs_reconnect({"cell-0": "hash-1"}) is True
        updater.conn_status.mark_trainer_stale()
        assert updater.conn_status.needs_reconnect(snapshot_cell_id_to_hashes) is True
