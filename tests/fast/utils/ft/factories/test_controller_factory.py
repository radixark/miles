from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.adapters.config import FtControllerConfig
from miles.utils.ft.factories.controller import build_ft_controller


class TestFtControllerConfig:
    def test_default_values(self) -> None:
        config = FtControllerConfig()
        assert config.platform == "stub"
        assert config.metric_store_backend == "mini"
        assert config.tick_interval == 30.0

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FtControllerConfig(unknown_field="bad")  # type: ignore[call-arg]

    def test_invalid_platform_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FtControllerConfig(platform="docker")  # type: ignore[arg-type]

    def test_invalid_metric_store_backend_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FtControllerConfig(metric_store_backend="influx")  # type: ignore[arg-type]


class TestBuildFtControllerConfigConflict:
    def test_config_and_kwargs_raises_value_error(self) -> None:
        config = FtControllerConfig()
        with pytest.raises(ValueError, match="Cannot provide both"):
            build_ft_controller(config=config, platform="stub")  # type: ignore[call-overload]


class TestBuildFtControllerStub:
    def test_stub_platform_returns_controller(self) -> None:
        controller = build_ft_controller(
            config=FtControllerConfig(platform="stub"),
            start_exporter=False,
        )
        assert controller is not None
        assert isinstance(controller, FtController)

    def test_start_exporter_false_skips_exporter_start(self) -> None:
        controller = build_ft_controller(
            config=FtControllerConfig(platform="stub"),
            start_exporter=False,
        )
        assert controller._controller_exporter is not None
        assert controller._controller_exporter._httpd is None, "exporter should not have been started"

    def test_kwargs_forwarded_to_config(self) -> None:
        controller = build_ft_controller(
            platform="stub",
            tick_interval=5.0,
            start_exporter=False,
        )
        assert controller._tick_interval == 5.0

    def test_empty_ft_id_gets_auto_generated(self) -> None:
        controller1 = build_ft_controller(
            config=FtControllerConfig(ft_id=""),
            start_exporter=False,
        )
        controller2 = build_ft_controller(
            config=FtControllerConfig(ft_id=""),
            start_exporter=False,
        )
        assert controller1 is not controller2

    def test_explicit_ft_id_preserved(self) -> None:
        controller = build_ft_controller(
            config=FtControllerConfig(ft_id="my-ft-123"),
            start_exporter=False,
        )
        assert controller is not None
        assert isinstance(controller, FtController)


class TestBuildFtControllerNotifier:
    def test_stub_platform_gets_stub_notifier(self) -> None:
        from miles.utils.ft.adapters.stubs import StubNotifier

        controller = build_ft_controller(
            config=FtControllerConfig(platform="stub"),
            start_exporter=False,
        )
        assert isinstance(controller._platform_deps.notifier, StubNotifier)

    def test_webhook_url_creates_lark_notifier(self) -> None:
        from miles.utils.ft.adapters.impl.notifiers.lark_notifier import LarkWebhookNotifier

        controller = build_ft_controller(
            config=FtControllerConfig(
                platform="stub",
                notify_webhook_url="https://lark.example.com/hook",
            ),
            start_exporter=False,
        )
        assert isinstance(controller._platform_deps.notifier, LarkWebhookNotifier)


class TestBuildPlatformComponentsUnknown:
    def test_unknown_platform_raises_value_error(self) -> None:
        from miles.utils.ft.factories.controller import _build_platform_components

        with pytest.raises(ValueError, match="Unknown platform"):
            _build_platform_components(platform="docker", ray_address="", entrypoint="")
