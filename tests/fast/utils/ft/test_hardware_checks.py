"""Tests for hardware_checks edge cases (check_critical_xid error paths)."""
import logging

import pytest

from miles.utils.ft.controller.detectors.hardware_checks import check_critical_xid
from miles.utils.ft.metric_names import XID_CODE_RECENT
from miles.utils.ft.models import MetricSample
from tests.fast.utils.ft.helpers import make_fake_metric_store


class TestCheckCriticalXidErrorPaths:
    def test_unparseable_xid_returns_empty_and_logs_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            MetricSample(name=XID_CODE_RECENT, labels={"xid": "NaN"}, value=1.0),
        ])

        with caplog.at_level(logging.WARNING):
            result = check_critical_xid(store)

        assert result == []
        assert "unparseable" in caplog.text

    def test_non_critical_xid_returns_empty(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            MetricSample(name=XID_CODE_RECENT, labels={"xid": "999"}, value=1.0),
        ])

        result = check_critical_xid(store)

        assert result == []

    def test_critical_xid_returns_fault(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            MetricSample(name=XID_CODE_RECENT, labels={"xid": "48"}, value=1.0),
        ])

        result = check_critical_xid(store)

        assert len(result) == 1
        assert result[0].node_id == "node-0"
        assert "48" in result[0].reason

    def test_mix_of_valid_and_invalid_xid_returns_partial(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            MetricSample(name=XID_CODE_RECENT, labels={"xid": "48"}, value=1.0),
        ])
        store.ingest_samples(target_id="node-1", samples=[
            MetricSample(name=XID_CODE_RECENT, labels={"xid": "not_a_number"}, value=1.0),
        ])

        with caplog.at_level(logging.WARNING):
            result = check_critical_xid(store)

        assert len(result) == 1
        assert result[0].node_id == "node-0"
        assert "unparseable" in caplog.text

    def test_empty_xid_string_logs_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            MetricSample(name=XID_CODE_RECENT, labels={"xid": ""}, value=1.0),
        ])

        with caplog.at_level(logging.WARNING):
            result = check_critical_xid(store)

        assert result == []
        assert "unparseable" in caplog.text

    def test_no_xid_data_returns_empty(self) -> None:
        store = make_fake_metric_store()

        result = check_critical_xid(store)

        assert result == []
