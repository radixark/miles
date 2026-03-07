"""Tests for GPU-specific fault checks (gpu_lost, non-auto-recoverable XID)."""
from miles.utils.ft.controller.detectors.checks.gpu.checks import (
    _check_non_auto_recoverable_xid,
)
from miles.utils.ft.models.metric_names import XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL
from miles.utils.ft.models.metrics import CounterSample
from tests.fast.utils.ft.conftest import make_fake_metric_store


class TestCheckNonAutoRecoverableXid:
    def test_no_data_returns_empty(self) -> None:
        store = make_fake_metric_store()

        result = _check_non_auto_recoverable_xid(store)

        assert result == []

    def test_zero_counter_returns_empty(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            CounterSample(name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL, labels={}, delta=0.0),
        ])

        result = _check_non_auto_recoverable_xid(store)

        assert result == []

    def test_positive_counter_returns_fault(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            CounterSample(name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL, labels={}, delta=1.0),
        ])

        result = _check_non_auto_recoverable_xid(store)

        assert len(result) == 1
        assert result[0].node_id == "node-0"
        assert "non-auto-recoverable XID" in result[0].reason

    def test_multiple_nodes_with_faults(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            CounterSample(name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL, labels={}, delta=1.0),
        ])
        store.ingest_samples(target_id="node-1", samples=[
            CounterSample(name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL, labels={}, delta=3.0),
        ])

        result = _check_non_auto_recoverable_xid(store)

        node_ids = {f.node_id for f in result}
        assert node_ids == {"node-0", "node-1"}

    def test_mixed_nodes_only_positive_faults(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            CounterSample(name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL, labels={}, delta=1.0),
        ])
        store.ingest_samples(target_id="node-1", samples=[
            CounterSample(name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL, labels={}, delta=0.0),
        ])

        result = _check_non_auto_recoverable_xid(store)

        assert len(result) == 1
        assert result[0].node_id == "node-0"
