import json
import os
import socket
import sys
from typing import Any

import pytest
from pydantic import ValidationError
from tests.fast.utils.workers.serving.registered_serve import serve_config_argv
from tests.fast.utils.workers.serving.serve_smoke_worker import SmokeServeSpec, SmokeWorkerConfig

from miles.utils.workers.serving import utils as serving_utils
from miles.utils.workers.serving.utils import override_argv, override_env, parse_serve_worker_config, split_worker_argv


def _payload() -> dict:
    (_, payload) = serve_config_argv(
        spec_class=SmokeServeSpec, config=SmokeWorkerConfig(rpc_port=8000, worker_argv=["-v"])
    )
    return json.loads(payload)


class TestOverrideEnv:
    def test_override_env_restores_existing_and_absent_keys_after_an_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Environment overrides restore old values and remove new keys when the context raises."""
        existing_key = "MILES_TEST_OVERRIDE_ENV_EXISTING"
        absent_key = "MILES_TEST_OVERRIDE_ENV_ABSENT"
        monkeypatch.setenv(existing_key, "original")
        monkeypatch.delenv(absent_key, raising=False)

        with pytest.raises(RuntimeError, match="inside context"):
            with override_env({existing_key: "overridden", absent_key: "introduced"}):
                assert os.environ[existing_key] == "overridden"
                assert os.environ[absent_key] == "introduced"
                raise RuntimeError("inside context")

        assert os.environ[existing_key] == "original"
        assert absent_key not in os.environ


class TestSplitWorkerArgv:
    @pytest.mark.parametrize("argv", [["--"], ["--host", "127.0.0.1", "--"]])
    def test_trailing_separator_yields_empty_worker_argv(self, argv: list[str]) -> None:
        """A separator with nothing after it is accepted and leaves the worker argv empty."""
        own_argv, worker_argv = split_worker_argv(argv)

        assert own_argv == argv[:-1]
        assert worker_argv == []


class TestParseServeWorkerConfig:
    def test_a_launcher_payload_round_trips(self) -> None:
        """The pod rebuilds its spec from exactly the worker type and config the launcher serialized."""
        config = parse_serve_worker_config(json.dumps(_payload()))

        assert config.worker_type == SmokeServeSpec.worker_type
        assert SmokeWorkerConfig.model_validate(config.args) == SmokeWorkerConfig(rpc_port=8000, worker_argv=["-v"])

    def test_a_payload_missing_a_field_is_refused(self) -> None:
        """A field silently defaulted on the pod could differ from the value the launcher meant."""
        payload = _payload()
        del payload["static_connections"]["static_conn_infos"]

        with pytest.raises(ValueError, match="Incomplete configuration"):
            parse_serve_worker_config(json.dumps(payload))

    def test_a_payload_with_an_unknown_field_is_refused(self) -> None:
        """A field the pod does not know would be dropped instead of applied."""
        payload = _payload() | {"pool_id": "trainer"}

        with pytest.raises(ValidationError):
            parse_serve_worker_config(json.dumps(payload))


class TestOverrideArgv:
    def test_override_argv_restores_the_original_after_an_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An exceptional context exit restores the exact argv object that preceded the override."""
        original_argv = ["runner.py", "--original"]
        monkeypatch.setattr(sys, "argv", original_argv)

        with pytest.raises(RuntimeError, match="boom"):
            with override_argv(["--replacement"]):
                assert sys.argv == ["runner.py", "--replacement"]
                raise RuntimeError("boom")

        assert sys.argv is original_argv


class TestCreateServerSocket:
    def test_dual_stack_listener_accepts_ipv4_and_ipv6_when_ipv6_sockets_default_to_v6_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicitly dual-stack listener accepts both address families despite a v6-only default."""
        real_socket = socket.socket

        def create_v6_only_socket(*args: Any, **kwargs: Any) -> socket.socket:
            server_socket = real_socket(*args, **kwargs)
            if server_socket.family == socket.AF_INET6:
                server_socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            return server_socket

        monkeypatch.setattr(serving_utils.socket, "socket", create_v6_only_socket)
        monkeypatch.setattr(serving_utils.socket, "has_dualstack_ipv6", lambda: True)

        with serving_utils.create_server_socket(port=0) as server_socket:
            port = server_socket.getsockname()[1]

            assert server_socket.family == socket.AF_INET6
            assert server_socket.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 0
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                pass
            with socket.create_connection(("::1", port), timeout=1):
                pass

    def test_ipv4_listener_remains_reachable_without_dual_stack_support(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A host without dual-stack IPv6 still accepts worker RPC calls over IPv4."""
        monkeypatch.setattr(serving_utils.socket, "has_dualstack_ipv6", lambda: False)

        with serving_utils.create_server_socket(port=0) as server_socket:
            port = server_socket.getsockname()[1]

            assert server_socket.family == socket.AF_INET
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                pass
