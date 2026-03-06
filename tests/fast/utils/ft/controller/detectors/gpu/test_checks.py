"""Tests for GPU-specific fault checks (gpu_lost, critical_xid)."""
import logging

import pytest

from miles.utils.ft.controller.detectors.gpu.checks import (
    _check_critical_xid,
)
from miles.utils.ft.models.metric_names import XID_CODE_RECENT
from miles.utils.ft.models.metrics import GaugeSample
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
