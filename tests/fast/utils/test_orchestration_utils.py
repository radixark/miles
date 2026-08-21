from argparse import Namespace

import pytest

import miles.utils.orchestration_utils as orchestration_utils
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity


class TestInitOrchestrationScript:
    def test_initializes_the_shared_driver_machinery_in_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every driver must initialize the shared machinery once, in dependency order, and return its manager."""
        args = Namespace(run="test")
        worker_manager = object()
        calls: list[str] = []
        captured: dict[str, object] = {}

        def fake_configure_logger(actual_args: Namespace, *, source: SimpleProcessIdentity) -> None:
            calls.append("configure_logger")
            captured["logger_args"] = actual_args
            captured["source"] = source

        def fake_maybe_start_periodic_pyspy_dump() -> None:
            calls.append("maybe_start_periodic_pyspy_dump")

        def fake_init_tracking(actual_args: Namespace) -> None:
            calls.append("init_tracking")
            captured["tracking_args"] = actual_args

        def fake_launch_worker_manager(actual_args: Namespace) -> object:
            calls.append("launch_worker_manager")
            captured["worker_manager_args"] = actual_args
            return worker_manager

        def fake_init_object_store(actual_args: Namespace, *, contribute_segment: bool) -> None:
            calls.append("object_store.init_instance")
            captured["object_store_args"] = actual_args
            captured["contribute_segment"] = contribute_segment

        monkeypatch.setattr(orchestration_utils, "configure_logger", fake_configure_logger)
        monkeypatch.setattr(
            orchestration_utils,
            "maybe_start_periodic_pyspy_dump",
            fake_maybe_start_periodic_pyspy_dump,
        )
        monkeypatch.setattr(orchestration_utils, "init_tracking", fake_init_tracking)
        monkeypatch.setattr(orchestration_utils, "launch_worker_manager", fake_launch_worker_manager)
        monkeypatch.setattr(orchestration_utils.object_store, "init_instance", fake_init_object_store)

        result = orchestration_utils.init_orchestration_script(args)

        assert calls == [
            "configure_logger",
            "maybe_start_periodic_pyspy_dump",
            "init_tracking",
            "launch_worker_manager",
            "object_store.init_instance",
        ]
        assert captured["logger_args"] is args
        assert captured["source"] == SimpleProcessIdentity(component="main")
        assert captured["tracking_args"] is args
        assert captured["worker_manager_args"] is args
        assert captured["object_store_args"] is args
        assert captured["contribute_segment"] is False
        assert result is worker_manager
