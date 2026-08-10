import pytest

from miles.utils.workers.serving.serve_inner import parse_own_args

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
