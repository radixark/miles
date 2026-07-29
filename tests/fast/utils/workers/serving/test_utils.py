import pytest

from miles.utils.workers.serving.utils import split_worker_argv


class TestSplitWorkerArgv:
    @pytest.mark.parametrize("argv", [["--"], ["--host", "127.0.0.1", "--"]])
    def test_trailing_separator_yields_empty_worker_argv(self, argv: list[str]) -> None:
        """A separator with nothing after it is accepted and leaves the worker argv empty."""
        own_argv, worker_argv = split_worker_argv(argv)

        assert own_argv == argv[:-1]
        assert worker_argv == []
