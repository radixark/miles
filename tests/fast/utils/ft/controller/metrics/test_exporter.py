from __future__ import annotations

from prometheus_client import CollectorRegistry
from tests.fast.utils.ft.conftest import get_sample_value, make_test_exporter

import miles.utils.ft.controller.metrics.metric_names as mn
from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.metrics.exporter import ControllerExporter


class TestControllerExporterGauges:
    def test_update_mode_default_subsystem(self) -> None:
        # 2.7: mode gauge now uses per-subsystem labels instead of a single global value
        registry, exporter = make_test_exporter()

        exporter.update_mode(is_recovery=True)

        assert get_sample_value(registry, mn.CONTROLLER_MODE, labels={"subsystem": "training"}) == 1.0

    def test_update_mode_custom_subsystem(self) -> None:
        # 2.7: each subsystem gets its own labeled metric
        registry, exporter = make_test_exporter()

        exporter.update_mode(is_recovery=True, subsystem="networking")
        exporter.update_mode(is_recovery=False, subsystem="training")

        assert get_sample_value(registry, mn.CONTROLLER_MODE, labels={"subsystem": "networking"}) == 1.0
        assert get_sample_value(registry, mn.CONTROLLER_MODE, labels={"subsystem": "training"}) == 0.0

    def test_update_tick_count_increments(self) -> None:
        registry, exporter = make_test_exporter()

        exporter.update_tick_count()
        exporter.update_tick_count()
        exporter.update_tick_count()

        assert get_sample_value(registry, mn.CONTROLLER_TICK_COUNT + "_total") == 3.0

    def test_update_recovery_phase_default_subsystem(self) -> None:
        # 2.7: recovery_phase gauge now uses per-subsystem labels
        registry, exporter = make_test_exporter()

        exporter.update_recovery_phase(2)

        assert get_sample_value(registry, mn.CONTROLLER_RECOVERY_PHASE, labels={"subsystem": "training"}) == 2.0

    def test_update_recovery_phase_custom_subsystem(self) -> None:
        # 2.7: each subsystem reports its own recovery phase
        registry, exporter = make_test_exporter()

        exporter.update_recovery_phase(3, subsystem="networking")

        assert get_sample_value(registry, mn.CONTROLLER_RECOVERY_PHASE, labels={"subsystem": "networking"}) == 3.0

    def test_update_main_job_status(self) -> None:
        registry, exporter = make_test_exporter()

        exporter.update_main_job_status(JobStatus.FAILED)

        assert get_sample_value(registry, mn.MAIN_JOB_STATUS) == -1.0

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


class TestControllerExporterSubsystemState:
    def test_update_subsystem_state_sets_mode_and_phase(self) -> None:
        # 2.7: update_subsystem_state is a convenience method that sets both
        # mode and recovery_phase for a named subsystem in one call
        registry, exporter = make_test_exporter()

        exporter.update_subsystem_state(subsystem="networking", is_recovery=True, recovery_phase_int=2)

        assert get_sample_value(registry, mn.CONTROLLER_MODE, labels={"subsystem": "networking"}) == 1.0
        assert get_sample_value(registry, mn.CONTROLLER_RECOVERY_PHASE, labels={"subsystem": "networking"}) == 2.0

    def test_update_from_state_iterates_all_subsystems(self) -> None:
        # 2.7: previously update_from_state only reported a single hardcoded "training"
        # subsystem; now it iterates the subsystem_modes dict
        registry, exporter = make_test_exporter()

        exporter.update_from_state(
            job_status=JobStatus.RUNNING,
            subsystem_modes={
                "training": (True, 1),
                "networking": (False, 0),
            },
            latest_loss=2.5,
            latest_mfu=0.4,
        )

        assert get_sample_value(registry, mn.CONTROLLER_MODE, labels={"subsystem": "training"}) == 1.0
        assert get_sample_value(registry, mn.CONTROLLER_MODE, labels={"subsystem": "networking"}) == 0.0
        assert get_sample_value(registry, mn.CONTROLLER_RECOVERY_PHASE, labels={"subsystem": "training"}) == 1.0
        assert get_sample_value(registry, mn.CONTROLLER_RECOVERY_PHASE, labels={"subsystem": "networking"}) == 0.0
        assert get_sample_value(registry, mn.MAIN_JOB_STATUS) == 1.0
        assert get_sample_value(registry, mn.TRAINING_LOSS_LATEST) == 2.5
        assert get_sample_value(registry, mn.TRAINING_MFU_LATEST) == 0.4


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

        assert (
            get_sample_value(
                registry,
                mn.CONTROLLER_DECISION_TOTAL,
                labels={"action": "enter_recovery", "trigger": "training_crash"},
            )
            == 2.0
        )
        assert (
            get_sample_value(
                registry,
                mn.CONTROLLER_DECISION_TOTAL,
                labels={"action": "mark_bad_and_restart", "trigger": "hardware"},
            )
            == 1.0
        )

    def test_record_decision_unknown_trigger(self) -> None:
        registry, exporter = make_test_exporter()

        exporter.record_decision(action="enter_recovery", trigger="unknown")

        assert (
            get_sample_value(
                registry,
                mn.CONTROLLER_DECISION_TOTAL,
                labels={"action": "enter_recovery", "trigger": "unknown"},
            )
            == 1.0
        )


class TestControllerExporterRecoveryDuration:
    def test_observe_recovery_duration(self) -> None:
        registry, exporter = make_test_exporter()

        exporter.observe_recovery_duration(45.3)

        count = get_sample_value(
            registry,
            mn.CONTROLLER_RECOVERY_DURATION_SECONDS + "_count",
        )
        assert count == 1.0

        total = get_sample_value(
            registry,
            mn.CONTROLLER_RECOVERY_DURATION_SECONDS + "_sum",
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
