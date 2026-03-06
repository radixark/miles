"""Tests for hardware_checks edge cases and boundary conditions."""
import logging

import pytest

from miles.utils.ft.controller.detectors.hardware_checks import (
    _check_critical_xid,
    _check_disk_fault,
    _check_majority_nic_down,
)
from miles.utils.ft.models.metric_names import NODE_FILESYSTEM_AVAIL_BYTES, NODE_NETWORK_UP, XID_CODE_RECENT
from miles.utils.ft.models import GaugeSample
from tests.fast.utils.ft.conftest import make_fake_metric_store


class TestCheckCriticalXidErrorPaths:
    def test_unparseable_xid_returns_empty_and_logs_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            GaugeSample(name=XID_CODE_RECENT, labels={"xid": "NaN"}, value=1.0),
        ])

        with caplog.at_level(logging.WARNING):
            result = _check_critical_xid(store)

        assert result == []
        assert "unparseable" in caplog.text

    def test_non_critical_xid_returns_empty(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            GaugeSample(name=XID_CODE_RECENT, labels={"xid": "999"}, value=1.0),
        ])

        result = _check_critical_xid(store)

        assert result == []

    def test_critical_xid_returns_fault(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            GaugeSample(name=XID_CODE_RECENT, labels={"xid": "48"}, value=1.0),
        ])

        result = _check_critical_xid(store)

        assert len(result) == 1
        assert result[0].node_id == "node-0"
        assert "48" in result[0].reason

    def test_mix_of_valid_and_invalid_xid_returns_partial(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            GaugeSample(name=XID_CODE_RECENT, labels={"xid": "48"}, value=1.0),
        ])
        store.ingest_samples(target_id="node-1", samples=[
            GaugeSample(name=XID_CODE_RECENT, labels={"xid": "not_a_number"}, value=1.0),
        ])

        with caplog.at_level(logging.WARNING):
            result = _check_critical_xid(store)

        assert len(result) == 1
        assert result[0].node_id == "node-0"
        assert "unparseable" in caplog.text

    def test_empty_xid_string_logs_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            GaugeSample(name=XID_CODE_RECENT, labels={"xid": ""}, value=1.0),
        ])

        with caplog.at_level(logging.WARNING):
            result = _check_critical_xid(store)

        assert result == []
        assert "unparseable" in caplog.text

    def test_no_xid_data_returns_empty(self) -> None:
        store = make_fake_metric_store()

        result = _check_critical_xid(store)

        assert result == []

    def test_custom_critical_xid_codes(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            GaugeSample(name=XID_CODE_RECENT, labels={"xid": "999"}, value=1.0),
        ])

        result = _check_critical_xid(store, critical_xid_codes=frozenset({999}))
        assert len(result) == 1
        assert result[0].node_id == "node-0"

    def test_node_id_none_skipped(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            GaugeSample(name=XID_CODE_RECENT, labels={"xid": "48"}, value=1.0),
        ])
        df = store.query_latest(XID_CODE_RECENT)
        assert not df.is_empty()

        result = _check_critical_xid(store)
        assert len(result) == 1


class TestCheckMajorityNicDown:
    def test_exactly_half_nics_down_does_not_trigger(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            GaugeSample(name=NODE_NETWORK_UP, labels={"interface": "eth0"}, value=1.0),
            GaugeSample(name=NODE_NETWORK_UP, labels={"interface": "eth1"}, value=0.0),
        ])

        result = _check_majority_nic_down(store)
        assert result == []

    def test_majority_nics_down_triggers(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            GaugeSample(name=NODE_NETWORK_UP, labels={"interface": "eth0"}, value=0.0),
            GaugeSample(name=NODE_NETWORK_UP, labels={"interface": "eth1"}, value=0.0),
            GaugeSample(name=NODE_NETWORK_UP, labels={"interface": "eth2"}, value=1.0),
        ])

        result = _check_majority_nic_down(store)
        assert len(result) == 1
        assert result[0].node_id == "node-0"

    def test_all_nics_up_returns_empty(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            GaugeSample(name=NODE_NETWORK_UP, labels={"interface": "eth0"}, value=1.0),
            GaugeSample(name=NODE_NETWORK_UP, labels={"interface": "eth1"}, value=1.0),
        ])

        result = _check_majority_nic_down(store)
        assert result == []

    def test_empty_metric_store_returns_empty(self) -> None:
        store = make_fake_metric_store()
        result = _check_majority_nic_down(store)
        assert result == []


class TestCheckDiskFault:
    def test_below_threshold_returns_fault(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            GaugeSample(name=NODE_FILESYSTEM_AVAIL_BYTES, labels={"mountpoint": "/"}, value=500e6),
        ])

        result = _check_disk_fault(store)

        assert len(result) == 1
        assert result[0].node_id == "node-0"
        assert "disk space low" in result[0].reason

    def test_above_threshold_returns_empty(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            GaugeSample(name=NODE_FILESYSTEM_AVAIL_BYTES, labels={"mountpoint": "/"}, value=100e9),
        ])

        assert _check_disk_fault(store) == []

    def test_empty_store_returns_empty(self) -> None:
        store = make_fake_metric_store()

        assert _check_disk_fault(store) == []

    def test_multiple_nodes_only_low_ones_flagged(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            GaugeSample(name=NODE_FILESYSTEM_AVAIL_BYTES, labels={"mountpoint": "/"}, value=200e6),
        ])
        store.ingest_samples(target_id="node-1", samples=[
            GaugeSample(name=NODE_FILESYSTEM_AVAIL_BYTES, labels={"mountpoint": "/"}, value=50e9),
        ])

        result = _check_disk_fault(store)

        assert len(result) == 1
        assert result[0].node_id == "node-0"

    def test_custom_threshold(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            GaugeSample(name=NODE_FILESYSTEM_AVAIL_BYTES, labels={"mountpoint": "/"}, value=5e9),
        ])

        result_default = _check_disk_fault(store)
        assert result_default == []

        result_high = _check_disk_fault(store, disk_available_threshold_bytes=10e9)
        assert len(result_high) == 1
        assert result_high[0].node_id == "node-0"

    def test_exactly_at_threshold_does_not_trigger(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            GaugeSample(name=NODE_FILESYSTEM_AVAIL_BYTES, labels={"mountpoint": "/"}, value=1e9),
        ])

        result = _check_disk_fault(store, disk_available_threshold_bytes=1e9)
        assert result == []
