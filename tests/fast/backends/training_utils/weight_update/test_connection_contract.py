from argparse import Namespace
from unittest.mock import MagicMock, patch

from miles.backends.training_utils.conn_status import ConnStatusManager
from miles.backends.training_utils.weight_update.updater import WeightUpdater

_UPDATER_MODULE = "miles.backends.training_utils.weight_update.updater"


def _build_updater() -> WeightUpdater:
    with patch(f"{_UPDATER_MODULE}.get_weight_transfer_protocol", return_value=MagicMock(supports_lora=False)):
        return WeightUpdater(
            Namespace(),
            [],
            weights_getter=lambda: {},
            model_name="test-model",
            quantization_config=None,
            iterator_factory=lambda *a, **k: MagicMock(name="hf_weight_iterator"),
            parallel_state=MagicMock(),
            is_lora=False,
        )


class TestWeightUpdaterConnectionContract:
    def test_the_weight_updater_exposes_the_connection_status_contract(self) -> None:
        """The actor unconditionally drives reconnection through conn_status, so the updater must own a working manager."""
        updater = _build_updater()
        snapshot_cell_id_to_hashes = {"cell-0": "hash-0"}

        assert isinstance(updater.conn_status, ConnStatusManager)
        assert updater.conn_status.needs_reconnect(snapshot_cell_id_to_hashes) is True
        updater.conn_status.mark_reconnected(snapshot_cell_id_to_hashes)
        assert updater.conn_status.needs_reconnect(snapshot_cell_id_to_hashes) is False
        assert updater.conn_status.needs_reconnect({"cell-0": "hash-1"}) is True
        updater.conn_status.mark_trainer_stale()
        assert updater.conn_status.needs_reconnect(snapshot_cell_id_to_hashes) is True
