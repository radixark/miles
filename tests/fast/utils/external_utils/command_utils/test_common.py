import json
import logging
import shlex

import pytest

from miles.utils.audit_utils.config_snapshot.generated_value_registry import (
    GENERATED_VALUES_ENV_VAR,
    GeneratedValueRegistry,
)
from miles.utils.env_report.redaction import redact_arg
from miles.utils.external_utils.command_utils import common
from miles.utils.external_utils.command_utils.common import MOONCAKE_INIT_KWARGS_FLAG, get_mooncake_object_store_args
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR


class TestGetMooncakeObjectStoreArgs:
    def test_a_remote_master_host_reaches_the_serialized_store_address(self) -> None:
        """A split deployment connects to the master host supplied by its driving release."""
        argv = shlex.split(get_mooncake_object_store_args(master_port=61234, master_host="mooncake.run.svc"))

        kwargs = json.loads(argv[argv.index(MOONCAKE_INIT_KWARGS_FLAG) + 1])

        assert kwargs["master_server_address"] == "mooncake.run.svc:61234"


class TestGetDefaultWandbArgs:
    def test_missing_credentials_configures_logging_before_reporting_the_skip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The launcher must make its skip message visible before it emits that message."""
        events: list[str] = []
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setattr(
            "miles.utils.logging_utils.configure_logger_raw", lambda name: events.append(f"configure:{name}")
        )

        class RecordingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                events.append(record.getMessage())

        handler = RecordingHandler()
        common.logger.addHandler(handler)
        common.logger.setLevel(logging.INFO)
        try:
            assert common.get_default_wandb_args("tests/e2e/test_run.py") == ""
        finally:
            common.logger.removeHandler(handler)

        assert events == ["configure:launcher", "Skip wandb configuration since WANDB_API_KEY is not found"]

    def test_environment_credentials_register_only_the_digest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Snapshot provenance exports no credential while the launch retains the actual key."""
        secret = "test-environment-wandb-secret"
        monkeypatch.setenv("WANDB_API_KEY", secret)
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, "/snapshot/records")
        monkeypatch.delenv(GENERATED_VALUES_ENV_VAR, raising=False)
        monkeypatch.delenv("GITHUB_COMMIT_NAME", raising=False)
        registry = GeneratedValueRegistry()
        monkeypatch.setattr(common, "generated_value_registry", registry)

        argv = shlex.split(common.get_default_wandb_args("tests/e2e/test_run.py", run_id="fixed"))
        exported = registry.to_env()[GENERATED_VALUES_ENV_VAR]

        assert argv[argv.index("--wandb-key") + 1] == secret
        assert json.loads(exported) == [
            {"kind": "wandb_key", "name": "environment", "value": redact_arg("wandb_key", secret)}
        ]
        assert secret not in exported
