import logging
from unittest.mock import patch

import pytest

from miles.utils.memory_utils import report_peak_memory

GIB = 1024**3


@pytest.fixture
def mock_reset_peak_stats():
    with (
        patch("miles.utils.memory_utils.torch.cuda.reset_peak_memory_stats") as mock_reset,
        patch("miles.utils.memory_utils.torch.cuda.max_memory_allocated", return_value=25 * GIB),
        patch("miles.utils.memory_utils.torch.cuda.max_memory_reserved", return_value=30 * GIB),
        patch("miles.utils.memory_utils.dist.get_rank", return_value=0),
    ):
        yield mock_reset


class TestReportPeakMemory:
    def test_resets_peak_stats_before_the_body_runs(self, mock_reset_peak_stats):
        with report_peak_memory("actor_train"):
            mock_reset_peak_stats.assert_called_once()

    def test_reports_the_phase_peak_in_gb(self, mock_reset_peak_stats, caplog):
        with caplog.at_level(logging.INFO, logger="miles.utils.memory_utils"):
            with report_peak_memory("actor_train"):
                pass

        assert "Peak-Memory actor_train" in caplog.text
        assert "max_allocated_GB=25.0" in caplog.text
        assert "max_reserved_GB=30.0" in caplog.text

    def test_reports_even_when_the_body_raises(self, mock_reset_peak_stats, caplog):
        with caplog.at_level(logging.INFO, logger="miles.utils.memory_utils"):
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                with report_peak_memory("log_probs"):
                    raise RuntimeError("CUDA out of memory")

        assert "Peak-Memory log_probs" in caplog.text
