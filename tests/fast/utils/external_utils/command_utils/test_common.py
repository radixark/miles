import json
import logging
import shlex

import pytest

from miles.utils.external_utils.command_utils import common
from miles.utils.external_utils.command_utils.common import MOONCAKE_INIT_KWARGS_FLAG, get_mooncake_object_store_args


class TestGetMooncakeObjectStoreArgs:
    def test_a_remote_master_host_reaches_the_serialized_store_address(self) -> None:
        """A split deployment connects to the master host supplied by its driving release."""
        argv = shlex.split(get_mooncake_object_store_args(master_port=61234, master_host="mooncake.run.svc"))

        kwargs = json.loads(argv[argv.index(MOONCAKE_INIT_KWARGS_FLAG) + 1])

        assert kwargs["master_server_address"] == "mooncake.run.svc:61234"


class TestOwnedMooncakeMaster:
    @pytest.mark.parametrize("address", ["etcd://etcd-1:2379,etcd-2:2379", "etcd://127.0.0.1:2379"])
    def test_ha_endpoints_are_left_to_mooncake(self, address: str) -> None:
        """HA discovery endpoints are not owned static master services, even on localhost."""
        argv = [MOONCAKE_INIT_KWARGS_FLAG, json.dumps({"master_server_address": address})]

        assert common.get_owned_mooncake_master_port(argv) is None

    def test_an_environment_master_does_not_require_a_local_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An external Mooncake client does not need the master executable installed locally."""
        monkeypatch.setenv("MOONCAKE_MASTER", "store.example:61234")

        assert common.get_owned_mooncake_master_port([]) is None

    def test_an_unconfigured_master_keeps_automatic_local_startup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default launcher still owns the default local master."""
        monkeypatch.delenv("MOONCAKE_MASTER", raising=False)

        assert common.get_owned_mooncake_master_port([]) == 50051

    def test_explicit_arguments_take_precedence_over_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit local endpoint overrides an inherited remote endpoint."""
        monkeypatch.setenv("MOONCAKE_MASTER", "store.example:61234")
        argv = [MOONCAKE_INIT_KWARGS_FLAG, json.dumps({"master_server_address": "127.0.0.1:61235"})]

        assert common.get_owned_mooncake_master_port(argv) == 61235


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
