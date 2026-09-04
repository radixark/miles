import pytest

from miles.utils.workers.serving.serve_inner import parse_own_args

WORKER_PATH = "tests.fast.utils.workers.e2e.e2e_worker.make_worker"


class TestParseOwnArgs:
    def test_omitted_host_and_port_use_public_defaults(self) -> None:
        """Leaving out --host and --port serves on the public bind address and the default port."""
        args = parse_own_args(["--worker", WORKER_PATH])

        assert (args.host, args.port) == ("0.0.0.0", 8000)

    def test_non_integer_port_is_a_usage_error(self) -> None:
        """A non-integer --port is rejected as a usage error instead of reaching the worker."""
        with pytest.raises(SystemExit) as exc_info:
            parse_own_args(["--worker", WORKER_PATH, "--port", "not-a-port"])

        assert exc_info.value.code == 2

    def test_unknown_inner_option_is_a_usage_error(self) -> None:
        """The inner entrypoint rejects an option it does not define instead of ignoring it."""
        with pytest.raises(SystemExit) as exc_info:
            parse_own_args(["--worker", WORKER_PATH, "--unknown-option", "1"])

        assert exc_info.value.code == 2
