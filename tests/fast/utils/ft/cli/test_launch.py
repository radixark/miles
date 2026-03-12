from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from miles.utils.ft.adapters.impl.notifiers.discord_notifier import DiscordWebhookNotifier
from miles.utils.ft.adapters.impl.notifiers.factory import build_notifier
from miles.utils.ft.adapters.impl.notifiers.lark_notifier import LarkWebhookNotifier
from miles.utils.ft.adapters.impl.notifiers.slack_notifier import SlackWebhookNotifier
from miles.utils.ft.adapters.stubs import StubNotifier
from miles.utils.ft.cli import app

runner = CliRunner()


@contextmanager
def _patch_build_and_run() -> Generator[tuple[MagicMock, MagicMock], None, None]:
    mock_actor_cls = MagicMock()
    mock_actor_instance = MagicMock()
    mock_actor_cls.options.return_value.remote.return_value = mock_actor_instance
    mock_actor_instance.submit_and_run.remote.return_value = MagicMock()

    with (
        patch("miles.utils.ft.adapters.impl.ray.controller_actor.FtControllerActor", mock_actor_cls),
        patch("miles.utils.ft.cli.launch.ray") as mock_ray,
        patch("miles.utils.ft.factories.scheduling.assert_cpu_only_nodes_exist"),
    ):
        yield mock_actor_cls, mock_ray


# ---------------------------------------------------------------------------
# FT Controller CLI tests
# ---------------------------------------------------------------------------


class TestLauncherCli:
    @pytest.mark.parametrize(
        "expected_text",
        [
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
            "--notify-webhook-url",
            "--notify-platform",
            "--scrape-interval-se",
        ],
    )
    def test_help_includes_option(self, expected_text: str) -> None:
        result = runner.invoke(app, ["launch", "--help"])
        assert result.exit_code == 0
        assert expected_text in result.output


class TestBuildNotifier:
    def test_webhook_url_defaults_to_lark(self) -> None:
        notifier = build_notifier(platform="stub", notify_webhook_url="https://hook.example.com")
        assert isinstance(notifier, LarkWebhookNotifier)

    def test_webhook_url_with_explicit_lark_platform(self) -> None:
        notifier = build_notifier(
            platform="stub",
            notify_webhook_url="https://hook.example.com",
            notify_platform="lark",
        )
        assert isinstance(notifier, LarkWebhookNotifier)

    def test_webhook_url_with_slack_platform(self) -> None:
        notifier = build_notifier(
            platform="stub",
            notify_webhook_url="https://hooks.slack.com/services/T/B/X",
            notify_platform="slack",
        )
        assert isinstance(notifier, SlackWebhookNotifier)

    def test_webhook_url_with_discord_platform(self) -> None:
        notifier = build_notifier(
            platform="stub",
            notify_webhook_url="https://discord.com/api/webhooks/1/abc",
            notify_platform="discord",
        )
        assert isinstance(notifier, DiscordWebhookNotifier)

    def test_stub_mode_without_webhook_returns_stub(self) -> None:
        notifier = build_notifier(platform="stub")
        assert isinstance(notifier, StubNotifier)

    def test_k8s_ray_mode_without_webhook_returns_none(self) -> None:
        notifier = build_notifier(platform="k8s-ray")
        assert notifier is None

    def test_empty_webhook_url_treated_as_unset(self) -> None:
        notifier = build_notifier(platform="stub", notify_webhook_url="  ")
        assert isinstance(notifier, StubNotifier)

    def test_no_webhook_non_stub_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            notifier = build_notifier(platform="k8s-ray")

        assert notifier is None
        assert "--notify-webhook-url" in caplog.text

    def test_unknown_platform_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown notify platform.*'xmpp'"):
            build_notifier(
                platform="stub",
                notify_webhook_url="https://example.com",
                notify_platform="xmpp",
            )

    def test_notify_platform_is_case_insensitive(self) -> None:
        notifier = build_notifier(
            platform="stub",
            notify_webhook_url="https://example.com",
            notify_platform="SLACK",
        )
        assert isinstance(notifier, SlackWebhookNotifier)


class TestLauncherSubmitAndRun:
    def test_inline_mode_calls_submit_and_run(self) -> None:
        with _patch_build_and_run() as (mock_actor_cls, mock_ray):
            result = runner.invoke(app, ["launch", "--platform", "stub", "--", "python3", "train.py"])

        assert result.exit_code == 0, result.output
        mock_ray.get.assert_called_once()

    def test_entrypoint_passed_to_config(self) -> None:
        with _patch_build_and_run() as (mock_actor_cls, _):
            result = runner.invoke(
                app,
                [
                    "launch",
                    "--platform",
                    "stub",
                    "--",
                    "python3",
                    "train.py",
                    "--lr",
                    "0.001",
                ],
            )

        assert result.exit_code == 0, result.output
        config = mock_actor_cls.options.return_value.remote.call_args.kwargs["config"]
        assert "python3" in config.entrypoint
        assert "train.py" in config.entrypoint
        assert "--lr" in config.entrypoint

    def test_ft_id_and_label_prefix_passed_to_config(self) -> None:
        with _patch_build_and_run() as (mock_actor_cls, _):
            result = runner.invoke(
                app,
                [
                    "launch",
                    "--platform",
                    "stub",
                    "--ft-id",
                    "myft",
                    "--k8s-label-prefix",
                    "pfx",
                    "--",
                    "python3",
                    "train.py",
                ],
            )

        assert result.exit_code == 0, result.output
        config = mock_actor_cls.options.return_value.remote.call_args.kwargs["config"]
        assert config.ft_id == "myft"
        assert config.k8s_label_prefix == "pfx"

    def test_notify_args_passed_to_config(self) -> None:
        with _patch_build_and_run() as (mock_actor_cls, _):
            result = runner.invoke(
                app,
                [
                    "launch",
                    "--platform",
                    "stub",
                    "--notify-webhook-url",
                    "https://hook.example.com",
                    "--notify-platform",
                    "slack",
                    "--",
                    "python3",
                    "train.py",
                ],
            )

        assert result.exit_code == 0, result.output
        config = mock_actor_cls.options.return_value.remote.call_args.kwargs["config"]
        assert config.notify_webhook_url == "https://hook.example.com"
        assert config.notify_platform == "slack"

    def test_runtime_env_json_parsed_to_config(self) -> None:
        runtime_env = {"env_vars": {"PYTHONPATH": "/root/Megatron-LM"}}
        with _patch_build_and_run() as (mock_actor_cls, _):
            result = runner.invoke(
                app,
                [
                    "launch",
                    "--platform",
                    "stub",
                    "--runtime-env-json",
                    '{"env_vars": {"PYTHONPATH": "/root/Megatron-LM"}}',
                    "--",
                    "python3",
                    "train.py",
                ],
            )

        assert result.exit_code == 0, result.output
        config = mock_actor_cls.options.return_value.remote.call_args.kwargs["config"]
        assert config.runtime_env == runtime_env


class TestLauncherScrapeInterval:
    def test_scrape_interval_seconds_passed_to_config(self) -> None:
        """scrape_interval_seconds existed in FtControllerConfig but was not
        exposed as a CLI parameter, so it could only use the default value."""
        with _patch_build_and_run() as (mock_actor_cls, _):
            result = runner.invoke(
                app,
                [
                    "launch",
                    "--platform",
                    "stub",
                    "--scrape-interval-seconds",
                    "5.0",
                    "--",
                    "python3",
                    "train.py",
                ],
            )

        assert result.exit_code == 0, result.output
        config = mock_actor_cls.options.return_value.remote.call_args.kwargs["config"]
        assert config.scrape_interval_seconds == 5.0

    def test_scrape_interval_seconds_defaults_to_10(self) -> None:
        with _patch_build_and_run() as (mock_actor_cls, _):
            result = runner.invoke(app, ["launch", "--platform", "stub", "--", "python3", "train.py"])

        assert result.exit_code == 0, result.output
        config = mock_actor_cls.options.return_value.remote.call_args.kwargs["config"]
        assert config.scrape_interval_seconds == 10.0


class TestLauncherInvalidInput:
    def test_invalid_runtime_env_json_fails(self) -> None:
        result = runner.invoke(
            app,
            [
                "launch",
                "--platform",
                "stub",
                "--runtime-env-json",
                "not-valid-json",
                "--",
                "python3",
                "train.py",
            ],
        )
        assert result.exit_code != 0

    def test_empty_entrypoint_produces_empty_string(self) -> None:
        with _patch_build_and_run() as (mock_actor_cls, _):
            result = runner.invoke(app, ["launch", "--platform", "stub"])

        assert result.exit_code == 0, result.output
        config = mock_actor_cls.options.return_value.remote.call_args.kwargs["config"]
        assert config.entrypoint == ""


class TestLauncherWiring:
    def test_main_creates_actor_with_config(self) -> None:
        """Verify launcher creates FtControllerActor with a valid config."""
        with _patch_build_and_run() as (mock_actor_cls, _):
            result = runner.invoke(app, ["launch", "--platform", "stub", "--", "python3", "train.py"])

        assert result.exit_code == 0, result.output
        mock_actor_cls.options.assert_called_once()
        config = mock_actor_cls.options.return_value.remote.call_args.kwargs["config"]
        assert config.platform == "stub"
        assert "python3" in config.entrypoint
