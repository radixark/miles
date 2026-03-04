from __future__ import annotations

from miles.utils.ft.controller.detectors import build_detector_chain
from miles.utils.ft.controller.mini_prometheus import MiniPrometheus
from miles.utils.ft.platform.controller_actor import build_ft_controller
from miles.utils.ft.platform.stubs import StubNodeManager, StubTrainingJob


class TestLauncherWiring:
    """Verify that build_ft_controller correctly wires all components
    for both stub and k8s-ray platforms (stub only tested here since
    k8s-ray requires actual K8s/Ray cluster).
    """

    def test_stub_platform_wiring(self) -> None:
        ctrl = build_ft_controller(platform="stub", start_exporter=False)

        assert isinstance(ctrl._node_manager, StubNodeManager)
        assert isinstance(ctrl._training_job, StubTrainingJob)
        assert isinstance(ctrl._metric_store, MiniPrometheus)
        assert ctrl._scrape_target_manager is not None

    def test_detector_chain_is_populated(self) -> None:
        ctrl = build_ft_controller(platform="stub", start_exporter=False)
        expected_count = len(build_detector_chain())
        assert len(ctrl._detectors) == expected_count
        assert expected_count > 0

    def test_detector_chain_types_match(self) -> None:
        ctrl = build_ft_controller(platform="stub", start_exporter=False)
        expected_chain = build_detector_chain()
        actual_types = [type(d).__name__ for d in ctrl._detectors]
        expected_types = [type(d).__name__ for d in expected_chain]
        assert actual_types == expected_types

    def test_controller_exporter_registered_as_scrape_target(self) -> None:
        ctrl = build_ft_controller(platform="stub", start_exporter=False)
        assert isinstance(ctrl._metric_store, MiniPrometheus)
        assert "controller" in ctrl._metric_store._scrape_targets
