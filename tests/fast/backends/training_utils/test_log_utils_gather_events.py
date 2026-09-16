import logging
from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.backends.training_utils import log_utils

_LOG_UTILS_LOGGER = "miles.backends.training_utils.log_utils"


class _FakeMultiPGUtil:
    def __init__(self, gathered: list[dict] | None, error: Exception | None = None) -> None:
        self.gathered = gathered
        self.error = error
        self.calls: list[dict] = []

    def gather_object(self, obj, groups_inner_to_outer):
        self.calls.append(obj)
        if self.error is not None:
            raise self.error
        return self.gathered


@pytest.fixture()
def non_source_rank(monkeypatch) -> None:
    parallel_state = SimpleNamespace(
        effective_dp_cp=SimpleNamespace(rank=1, size=4, gloo_groups_inner_to_outer=[]),
    )
    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)


def _gather_messages(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == _LOG_UTILS_LOGGER]


class TestGatherLogDataEvents:
    def test_successful_gather_brackets_the_collective_with_ft_records(self, non_source_rank, monkeypatch, caplog):
        """A successful gather emits ft-tagged start and success end log_gather records for the rank."""
        monkeypatch.setattr(log_utils, "MultiPGUtil", _FakeMultiPGUtil(gathered=None))

        with caplog.at_level(logging.INFO, logger=_LOG_UTILS_LOGGER):
            result = log_utils.gather_log_data("perf", Namespace(), 3, {"loss": 1.0})

        assert result is None
        assert _gather_messages(caplog) == [
            "ft op=cross_cell phase=start kind=log_gather rank=1",
            "ft op=cross_cell phase=end kind=log_gather rank=1 success=true",
        ]

    def test_gather_runtime_error_is_reported_as_a_degraded_ft_record(self, non_source_rank, monkeypatch, caplog):
        """A RuntimeError from the collective ends the ft record as a degraded failure with the traceback."""
        monkeypatch.setattr(log_utils, "MultiPGUtil", _FakeMultiPGUtil(gathered=None, error=RuntimeError("boom")))

        with caplog.at_level(logging.INFO, logger=_LOG_UTILS_LOGGER):
            result = log_utils.gather_log_data("perf", Namespace(), 3, {"loss": 1.0})

        assert result is None
        assert _gather_messages(caplog) == [
            "ft op=cross_cell phase=start kind=log_gather rank=1",
            "ft op=cross_cell phase=end kind=log_gather rank=1 success=false degraded=true",
        ]
        end = [record for record in caplog.records if record.name == _LOG_UTILS_LOGGER][-1]
        assert end.levelno == logging.WARNING
        assert end.exc_info[0] is RuntimeError
