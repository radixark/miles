"""Tests for GPU-specific fault checks (gpu_lost, non-auto-recoverable XID)."""

from tests.fast.utils.ft.conftest import make_fake_metric_store

from miles.utils.ft.agents.types import CounterSample, GaugeSample
from miles.utils.ft.controller.detectors.checks.gpu.checks import (
    _check_gpu_lost,
    _check_non_auto_recoverable_xid,
    check_gpu_faults,
)
from miles.utils.ft.utils.metric_names import GPU_AVAILABLE, XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL


class TestCheckNonAutoRecoverableXid:
    def test_no_data_returns_empty(self) -> None:
        store = make_fake_metric_store()

        result = _check_non_auto_recoverable_xid(store)

        assert result == []

    def test_zero_counter_returns_empty(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[
                CounterSample(name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL, labels={}, delta=0.0),
            ],
        )

        result = _check_non_auto_recoverable_xid(store)

        assert result == []

    def test_positive_counter_returns_fault(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[
                CounterSample(name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL, labels={}, delta=1.0),
            ],
        )

        result = _check_non_auto_recoverable_xid(store)

        assert len(result) == 1
        assert result[0].node_id == "node-0"
        assert "non-auto-recoverable XID" in result[0].reason

    def test_multiple_nodes_with_faults(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[
                CounterSample(name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL, labels={}, delta=1.0),
            ],
        )
        store.ingest_samples(
            target_id="node-1",
            samples=[
                CounterSample(name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL, labels={}, delta=3.0),
            ],
        )

        result = _check_non_auto_recoverable_xid(store)

        node_ids = {f.node_id for f in result}
        assert node_ids == {"node-0", "node-1"}

    def test_mixed_nodes_only_positive_faults(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[
                CounterSample(name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL, labels={}, delta=1.0),
            ],
        )
        store.ingest_samples(
            target_id="node-1",
            samples=[
                CounterSample(name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL, labels={}, delta=0.0),
            ],
        )

        result = _check_non_auto_recoverable_xid(store)

        assert len(result) == 1
        assert result[0].node_id == "node-0"


# ---------------------------------------------------------------------------
# P1 item 10: check_gpu_faults() and _check_gpu_lost()
# ---------------------------------------------------------------------------


class TestCheckGpuLost:
    def test_no_data_returns_empty(self) -> None:
        store = make_fake_metric_store()
        assert _check_gpu_lost(store) == []

    def test_all_healthy_returns_empty(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[GaugeSample(name=GPU_AVAILABLE, value=1.0, labels={"gpu": "0"})],
        )
        assert _check_gpu_lost(store) == []

    def test_one_gpu_lost(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[GaugeSample(name=GPU_AVAILABLE, value=0.0, labels={"gpu": "3"})],
        )
        result = _check_gpu_lost(store)
        assert len(result) == 1
        assert result[0].node_id == "node-0"
        assert "unavailable" in result[0].reason.lower()

    def test_multiple_nodes_with_gpu_lost(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[GaugeSample(name=GPU_AVAILABLE, value=0.0, labels={"gpu": "0"})],
        )
        store.ingest_samples(
            target_id="node-1",
            samples=[GaugeSample(name=GPU_AVAILABLE, value=0.0, labels={"gpu": "2"})],
        )
        result = _check_gpu_lost(store)
        node_ids = {f.node_id for f in result}
        assert node_ids == {"node-0", "node-1"}

    def test_mixed_nodes_only_lost_ones_returned(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[GaugeSample(name=GPU_AVAILABLE, value=0.0, labels={"gpu": "0"})],
        )
        store.ingest_samples(
            target_id="node-1",
            samples=[GaugeSample(name=GPU_AVAILABLE, value=1.0, labels={"gpu": "0"})],
        )
        result = _check_gpu_lost(store)
        assert len(result) == 1
        assert result[0].node_id == "node-0"


class TestCheckGpuFaults:
    def test_combines_gpu_lost_and_xid(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[GaugeSample(name=GPU_AVAILABLE, value=0.0, labels={"gpu": "0"})],
        )
        store.ingest_samples(
            target_id="node-1",
            samples=[CounterSample(name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL, labels={}, delta=1.0)],
        )
        result = check_gpu_faults(store)
        node_ids = {f.node_id for f in result}
        assert node_ids == {"node-0", "node-1"}

    def test_no_faults_returns_empty(self) -> None:
        store = make_fake_metric_store()
        assert check_gpu_faults(store) == []
