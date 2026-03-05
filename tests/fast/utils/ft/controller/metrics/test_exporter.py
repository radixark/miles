from __future__ import annotations

from prometheus_client import CollectorRegistry

import miles.utils.ft.metric_names as mn
from miles.utils.ft.controller.controller_exporter import ControllerExporter
from miles.utils.ft.models import RecoveryPhase
from miles.utils.ft.platform.protocols import JobStatus
from tests.fast.utils.ft.conftest import get_sample_value, make_test_exporter


class TestControllerExporterGauges:
    def test_initial_mode_is_zero(self) -> None:
        registry, _ = make_test_exporter()
        assert get_sample_value(registry, mn.CONTROLLER_MODE) == 0.0

    def test_update_mode(self) -> None:
        registry, exporter = make_test_exporter()

        exporter.update_mode(is_recovery=True)

        assert get_sample_value(registry, mn.CONTROLLER_MODE) == 1.0

    def test_update_tick_count_increments(self) -> None:
        registry, exporter = make_test_exporter()

        exporter.update_tick_count()
        exporter.update_tick_count()
        exporter.update_tick_count()

        assert get_sample_value(registry, mn.CONTROLLER_TICK_COUNT + "_total") == 3.0

    def test_update_recovery_phase(self) -> None:
        registry, exporter = make_test_exporter()

        exporter.update_recovery_phase(RecoveryPhase.REATTEMPTING)

        assert get_sample_value(registry, mn.CONTROLLER_RECOVERY_PHASE) == 2.0

    def test_update_training_job_status(self) -> None:
        registry, exporter = make_test_exporter()

        exporter.update_training_job_status(JobStatus.FAILED)

        assert get_sample_value(registry, mn.TRAINING_JOB_STATUS) == -1.0

    def test_update_training_metrics_loss_and_mfu(self) -> None:
        registry, exporter = make_test_exporter()

        exporter.update_training_metrics(loss=2.5, mfu=0.42)

        assert get_sample_value(registry, mn.TRAINING_LOSS_LATEST) == 2.5
        assert get_sample_value(registry, mn.TRAINING_MFU_LATEST) == 0.42

    def test_update_training_metrics_none_values_no_change(self) -> None:
        registry, exporter = make_test_exporter()
        exporter.update_training_metrics(loss=1.0, mfu=0.5)

        exporter.update_training_metrics(loss=None, mfu=None)

        assert get_sample_value(registry, mn.TRAINING_LOSS_LATEST) == 1.0
        assert get_sample_value(registry, mn.TRAINING_MFU_LATEST) == 0.5


class TestControllerExporterLastTickTimestamp:
    def test_update_last_tick_timestamp_sets_gauge(self) -> None:
        registry, exporter = make_test_exporter()

        exporter.update_last_tick_timestamp(1709654400.0)

        assert get_sample_value(registry, mn.CONTROLLER_LAST_TICK_TIMESTAMP) == 1709654400.0

    def test_last_tick_timestamp_increases_on_subsequent_calls(self) -> None:
        registry, exporter = make_test_exporter()

        exporter.update_last_tick_timestamp(1000.0)
        first = get_sample_value(registry, mn.CONTROLLER_LAST_TICK_TIMESTAMP)

        exporter.update_last_tick_timestamp(2000.0)
        second = get_sample_value(registry, mn.CONTROLLER_LAST_TICK_TIMESTAMP)

        assert second > first


class TestControllerExporterAddress:
    def test_default_port(self) -> None:
        _, exporter = make_test_exporter()
        assert exporter.address == "http://localhost:9400"

    def test_custom_port(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(port=9500, registry=registry)
        assert exporter.address == "http://localhost:9500"


class TestControllerExporterDecisionCounter:
    def test_record_decision_increments_labeled_counter(self) -> None:
        registry, exporter = make_test_exporter()

        exporter.record_decision(action="enter_recovery", trigger="training_crash")
        exporter.record_decision(action="enter_recovery", trigger="training_crash")
        exporter.record_decision(action="mark_bad_and_restart", trigger="hardware")

        assert get_sample_value(
            registry, mn.CONTROLLER_DECISION_TOTAL + "_total",
            labels={"action": "enter_recovery", "trigger": "training_crash"},
        ) == 2.0
        assert get_sample_value(
            registry, mn.CONTROLLER_DECISION_TOTAL + "_total",
            labels={"action": "mark_bad_and_restart", "trigger": "hardware"},
        ) == 1.0

    def test_record_decision_unknown_trigger(self) -> None:
        registry, exporter = make_test_exporter()

        exporter.record_decision(action="enter_recovery", trigger="unknown")

        assert get_sample_value(
            registry, mn.CONTROLLER_DECISION_TOTAL + "_total",
            labels={"action": "enter_recovery", "trigger": "unknown"},
        ) == 1.0


class TestControllerExporterRecoveryDuration:
    def test_observe_recovery_duration(self) -> None:
        registry, exporter = make_test_exporter()

        exporter.observe_recovery_duration(45.3)

        count = get_sample_value(
            registry, mn.CONTROLLER_RECOVERY_DURATION_SECONDS + "_count",
        )
        assert count == 1.0

        total = get_sample_value(
            registry, mn.CONTROLLER_RECOVERY_DURATION_SECONDS + "_sum",
        )
        assert total == 45.3


class TestControllerExporterLifecycle:
    def test_stop_shuts_down_http_server(self) -> None:
        _, exporter = make_test_exporter()

        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()

        exporter._port = port
        exporter.start()
        assert exporter._httpd is not None

        exporter.stop()
        assert exporter._httpd is None

    def test_stop_idempotent_when_not_started(self) -> None:
        _, exporter = make_test_exporter()
        exporter.stop()
        assert exporter._httpd is None
