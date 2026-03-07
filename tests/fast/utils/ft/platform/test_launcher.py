from __future__ import annotations

from contextlib import contextmanager
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from miles.utils.ft.platform.launcher import app
from miles.utils.ft.platform.notifiers.factory import build_notifier
from miles.utils.ft.platform.notifiers.lark_notifier import LarkWebhookNotifier
from miles.utils.ft.platform.stubs import StubNotifier

runner = CliRunner()


@contextmanager
def _patch_build_and_run() -> Generator[tuple[MagicMock, MagicMock], None, None]:
    mock_actor_cls = MagicMock()
    mock_actor_instance = MagicMock()
    mock_actor_cls.options.return_value.remote.return_value = mock_actor_instance
    mock_actor_instance.submit_and_run.remote.return_value = MagicMock()

    with (
        patch("miles.utils.ft.platform.launcher.FtControllerActor", mock_actor_cls),
        patch("miles.utils.ft.platform.launcher.ray") as mock_ray,
    ):
        yield mock_actor_cls, mock_ray


# ---------------------------------------------------------------------------
# FT Controller CLI tests
# ---------------------------------------------------------------------------


class TestLauncherCli:
    @pytest.mark.parametrize("expected_text", [
        "tick-interval",
        "FT Controller",
        "--platform",
        "--ray-address",
        "--metric-store-backe",
        "--prometheus-url",
        "--controller-exporte",
        "--runtime-env-json",
        "--ft-id",
        "--k8s-label-prefix",
    ])
    def test_help_includes_option(self, expected_text: str) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert expected_text in result.output


class TestBuildNotifier:
    def test_webhook_url_returns_lark_notifier(self) -> None:
        with patch.dict("os.environ", {"MILES_FT_NOTIFY_WEBHOOK_URL": "https://hook.example.com"}):
            notifier = build_notifier(platform="stub")
        assert isinstance(notifier, LarkWebhookNotifier)

    def test_stub_mode_without_webhook_returns_stub(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            notifier = build_notifier(platform="stub")
        assert isinstance(notifier, StubNotifier)

    def test_k8s_ray_mode_without_webhook_returns_none(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            notifier = build_notifier(platform="k8s-ray")
        assert notifier is None

    def test_empty_webhook_url_treated_as_unset(self) -> None:
        with patch.dict("os.environ", {"MILES_FT_NOTIFY_WEBHOOK_URL": "  "}):
            notifier = build_notifier(platform="stub")
        assert isinstance(notifier, StubNotifier)

    def test_no_webhook_non_stub_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        with patch.dict("os.environ", {}, clear=True):
            with caplog.at_level(logging.WARNING):
                notifier = build_notifier(platform="k8s-ray")
        assert notifier is None
        assert "MILES_FT_NOTIFY_WEBHOOK_URL" in caplog.text


class TestLauncherSubmitAndRun:
    def test_inline_mode_calls_submit_and_run(self) -> None:
        with _patch_build_and_run() as (mock_actor_cls, mock_ray):
            result = runner.invoke(app, ["--platform", "stub", "--", "python3", "train.py"])

        assert result.exit_code == 0, result.output
        mock_ray.get.assert_called_once()

    def test_entrypoint_passed_to_config(self) -> None:
        with _patch_build_and_run() as (mock_actor_cls, _):
            result = runner.invoke(app, [
                "--platform", "stub",
                "--", "python3", "train.py", "--lr", "0.001",
            ])

        assert result.exit_code == 0, result.output
        config = mock_actor_cls.options.return_value.remote.call_args.kwargs["config"]
        assert "python3" in config.entrypoint
        assert "train.py" in config.entrypoint
        assert "--lr" in config.entrypoint

    def test_ft_id_and_label_prefix_passed_to_config(self) -> None:
        with _patch_build_and_run() as (mock_actor_cls, _):
            result = runner.invoke(app, [
                "--platform", "stub",
                "--ft-id", "myft",
                "--k8s-label-prefix", "pfx",
                "--", "python3", "train.py",
            ])

        assert result.exit_code == 0, result.output
        config = mock_actor_cls.options.return_value.remote.call_args.kwargs["config"]
        assert config.ft_id == "myft"
        assert config.k8s_label_prefix == "pfx"

    def test_runtime_env_json_parsed_to_config(self) -> None:
        runtime_env = {"env_vars": {"PYTHONPATH": "/root/Megatron-LM"}}
        with _patch_build_and_run() as (mock_actor_cls, _):
            result = runner.invoke(app, [
                "--platform", "stub",
                "--runtime-env-json", '{"env_vars": {"PYTHONPATH": "/root/Megatron-LM"}}',
                "--", "python3", "train.py",
            ])

        assert result.exit_code == 0, result.output
        config = mock_actor_cls.options.return_value.remote.call_args.kwargs["config"]
        assert config.runtime_env == runtime_env


class TestLauncherInvalidInput:
    def test_invalid_runtime_env_json_fails(self) -> None:
        result = runner.invoke(app, [
            "--platform", "stub",
            "--runtime-env-json", "not-valid-json",
            "--", "python3", "train.py",
        ])
        assert result.exit_code != 0

    def test_empty_entrypoint_produces_empty_string(self) -> None:
        with _patch_build_and_run() as (mock_actor_cls, _):
            result = runner.invoke(app, ["--platform", "stub"])

        assert result.exit_code == 0, result.output
        config = mock_actor_cls.options.return_value.remote.call_args.kwargs["config"]
        assert config.entrypoint == ""


class TestLauncherWiring:
    def test_main_creates_actor_with_config(self) -> None:
        """Verify launcher creates FtControllerActor with a valid config."""
        with _patch_build_and_run() as (mock_actor_cls, _):
            result = runner.invoke(app, ["--platform", "stub", "--", "python3", "train.py"])

        assert result.exit_code == 0, result.output
        mock_actor_cls.options.assert_called_once()
        config = mock_actor_cls.options.return_value.remote.call_args.kwargs["config"]
        assert config.platform == "stub"
        assert "python3" in config.entrypoint
