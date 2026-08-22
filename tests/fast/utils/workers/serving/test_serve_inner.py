from types import SimpleNamespace
from typing import Any

import pytest

from miles.utils.workers.serving import serve_inner as serve_inner_module
from miles.utils.workers.serving.serve_inner import main, parse_own_args

SPECS_PATH = "tests.fast.utils.workers.e2e.e2e_worker.compute_specs"
POOL_ID = "e2e-pool"


class TestParseOwnArgs:
    def test_the_spec_table_and_the_pool_it_serves_are_read(self) -> None:
        """These two are the whole of what the pod needs to find the one spec it is a worker of."""
        args = parse_own_args(["--specs", SPECS_PATH, "--pool-id", POOL_ID])

        assert (args.specs, args.pool_id) == (SPECS_PATH, POOL_ID)

    def test_an_omitted_pool_id_is_a_usage_error(self) -> None:
        """A process that does not know which pool it serves would pick a spec at random."""
        with pytest.raises(SystemExit) as exc_info:
            parse_own_args(["--specs", SPECS_PATH])

        assert exc_info.value.code == 2

    def test_an_omitted_spec_table_is_a_usage_error(self) -> None:
        """Without the run's spec table there is nothing to match the pool id against."""
        with pytest.raises(SystemExit) as exc_info:
            parse_own_args(["--pool-id", POOL_ID])

        assert exc_info.value.code == 2

    def test_unknown_inner_option_is_a_usage_error(self) -> None:
        """The inner entrypoint rejects an option it does not define instead of ignoring it."""
        with pytest.raises(SystemExit) as exc_info:
            parse_own_args(["--specs", SPECS_PATH, "--pool-id", POOL_ID, "--unknown-option", "1"])

        assert exc_info.value.code == 2


class TestStartingTheServer:
    def test_the_inner_worker_serves_its_rpc_app_on_its_own_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A split worker serves the app it built on the port its identity resolved to."""
        seen: list[dict[str, Any]] = []
        spec = SimpleNamespace(worker_class="demo.Worker")
        monkeypatch.setattr(serve_inner_module, "split_worker_argv", lambda argv: ([], []))
        monkeypatch.setattr(
            serve_inner_module,
            "parse_own_args",
            lambda argv: SimpleNamespace(specs=SPECS_PATH, pool_id=POOL_ID),
        )
        monkeypatch.setattr(serve_inner_module, "compute_serve_worker_spec", lambda **kwargs: spec)
        monkeypatch.setattr(serve_inner_module, "create_worker", lambda *args, **kwargs: object())
        monkeypatch.setattr(serve_inner_module, "_rpc_port_of", lambda value: 12345)
        monkeypatch.setattr(serve_inner_module, "read_worker_in_pod_index", lambda environ: 0)
        monkeypatch.setattr(serve_inner_module, "create_rpc_app", lambda worker: object())
        monkeypatch.setattr(serve_inner_module.uvicorn, "run", lambda app, **kwargs: seen.append(kwargs) or None)

        main()

        assert set(seen[0]) == {"host", "port"}
        assert seen[0]["host"] == "0.0.0.0"
        assert seen[0]["port"] == 12345
