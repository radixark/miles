"""Tests for hardware_checks edge cases and boundary conditions."""

import pytest

from datetime import timedelta

from miles.utils.ft.controller.detectors.hardware_checks import (
    _check_disk_fault,
    _check_majority_nic_down,
    check_nic_down_in_window,
)
from miles.utils.ft.models.metric_names import NODE_FILESYSTEM_AVAIL_BYTES, NODE_NETWORK_UP
from miles.utils.ft.models.metrics import GaugeSample
from tests.fast.utils.ft.conftest import inject_nic_down, inject_nic_up, make_fake_metric_store


class TestCheckNicDownInWindow:
    def test_returns_ephemeral_faults(self) -> None:
        """NodeFault from check_nic_down_in_window must have ephemeral=True."""
        store = make_fake_metric_store()
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")

        faults = check_nic_down_in_window(store, window=timedelta(minutes=5), threshold=2)

        assert len(faults) == 1
        assert faults[0].node_id == "node-0"
        assert faults[0].ephemeral is True


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
