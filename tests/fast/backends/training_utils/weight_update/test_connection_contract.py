from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.training_utils.weight_update import receiver_pool as pools_module
from miles.backends.training_utils.weight_update.conn_status import ConnStatusManager
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.updater import WeightUpdater

_UPDATER_MODULE = "miles.backends.training_utils.weight_update.updater"


def _build_updater(supports_receiver_pools=False) -> WeightUpdater:
    protocol = MagicMock(
        spec=WeightTransferProtocol,
        supports_lora=False,
        supports_receiver_pools=supports_receiver_pools,
    )
    with patch(f"{_UPDATER_MODULE}.get_weight_transfer_protocol", return_value=protocol):
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
        updater.mark_trainer_stale()
        assert updater.conn_status.needs_reconnect(snapshot_cell_id_to_hashes) is True

    def test_receiver_pools_reuse_independent_protocol_connections_and_adapters(self, monkeypatch):
        updater = _build_updater(supports_receiver_pools=True)
        rollout_protocol, rollout_connection = updater.protocol, updater.conn_status
        updater._registered_adapters.add("rollout-adapter")
        rollout_connection.mark_reconnected({"rollout-cell": "incarnation-1"})
        eval_protocol = MagicMock(spec=WeightTransferProtocol, supports_receiver_pools=True)
        create = MagicMock(return_value=eval_protocol)
        monkeypatch.setattr(pools_module, "get_weight_transfer_protocol", create)

        with updater.receiver_pool("rollout"):
            assert updater.protocol is rollout_protocol
        create.assert_not_called()
        with updater.receiver_pool("eval"):
            assert updater.protocol is eval_protocol
            eval_connection = updater.conn_status
            assert eval_connection is not rollout_connection
            assert updater._registered_adapters == set()
            updater._registered_adapters.add("eval-adapter")
            assert eval_connection.needs_reconnect({"eval-cell": "incarnation-1"})
            eval_connection.mark_reconnected({"eval-cell": "incarnation-1"})
            with updater.receiver_pool("eval"):
                assert updater.protocol is eval_protocol
                assert updater.conn_status is eval_connection
        assert updater.protocol is rollout_protocol
        assert updater.conn_status is rollout_connection
        assert updater._registered_adapters == {"rollout-adapter"}
        assert not rollout_connection.needs_reconnect({"rollout-cell": "incarnation-1"})
        with updater.receiver_pool("eval"):
            assert updater.protocol is eval_protocol
            assert updater.conn_status is eval_connection
            assert updater._registered_adapters == {"eval-adapter"}
            assert not eval_connection.needs_reconnect({"eval-cell": "incarnation-1"})
            assert eval_connection.needs_reconnect({"eval-cell": "incarnation-2"})
        create.assert_called_once_with(updater.args)

    def test_failed_eval_restores_rollout_without_discarding_eval_state(self, monkeypatch):
        updater = _build_updater(supports_receiver_pools=True)
        rollout_protocol, rollout_connection = updater.protocol, updater.conn_status
        eval_protocol = MagicMock(spec=WeightTransferProtocol, supports_receiver_pools=True)
        monkeypatch.setattr(pools_module, "get_weight_transfer_protocol", lambda args: eval_protocol)

        with pytest.raises(RuntimeError, match="after update"):
            with updater.receiver_pool("eval"):
                updater._registered_adapters.add("applied-adapter")
                updater.conn_status.mark_reconnected({"eval-cell": "incarnation-1"})
                raise RuntimeError("after update")
        assert updater.protocol is rollout_protocol
        assert updater.conn_status is rollout_connection
        assert updater._registered_adapters == set()
        with updater.receiver_pool("eval"):
            assert updater.protocol is eval_protocol
            assert updater._registered_adapters == {"applied-adapter"}
            assert not updater.conn_status.needs_reconnect({"eval-cell": "incarnation-1"})

    def test_trainer_reconfiguration_invalidates_active_and_saved_pools(self, monkeypatch):
        updater = _build_updater(supports_receiver_pools=True)
        updater.conn_status.mark_reconnected({"rollout-cell": "incarnation-1"})
        monkeypatch.setattr(
            pools_module,
            "get_weight_transfer_protocol",
            lambda args: MagicMock(spec=WeightTransferProtocol, supports_receiver_pools=True),
        )
        with updater.receiver_pool("eval"):
            updater.conn_status.mark_reconnected({"eval-cell": "incarnation-1"})
            updater.mark_trainer_stale()
            assert updater.conn_status.needs_reconnect({"eval-cell": "incarnation-1"})
        assert updater.conn_status.needs_reconnect({"rollout-cell": "incarnation-1"})

    @pytest.mark.parametrize(
        "supported,pool_id,error",
        [
            (False, "eval", "receiver-pool support"),
            (False, "unknown", "Unknown receiver pool"),
            (True, "unknown", "Unknown receiver pool"),
        ],
    )
    def test_unsupported_pool_is_rejected_before_protocol_creation(self, monkeypatch, supported, pool_id, error):
        assert WeightTransferProtocol.supports_receiver_pools is False
        updater = _build_updater(supports_receiver_pools=supported)
        original = updater.protocol
        create = MagicMock()
        monkeypatch.setattr(pools_module, "get_weight_transfer_protocol", create)
        with updater.receiver_pool("rollout"):
            assert updater.protocol is original
        with pytest.raises(ValueError, match=error):
            with updater.receiver_pool(pool_id):
                pytest.fail("unsupported pool must not be entered")
        assert updater.protocol is original
        create.assert_not_called()
