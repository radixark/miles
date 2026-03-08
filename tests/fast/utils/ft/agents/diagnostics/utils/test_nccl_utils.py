"""Tests for NCCL test helpers: command builder, output parser, subprocess runner."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from miles.utils.ft.agents.diagnostics.utils.nccl_utils import (
    _interpret_nccl_output,
    build_nccl_test_cmd,
    parse_avg_bus_bandwidth,
    run_nccl_test,
)

_DIAG_TYPE = "nccl_simple"


# ---------------------------------------------------------------------------
# build_nccl_test_cmd
# ---------------------------------------------------------------------------


class TestBuildNcclTestCmd:
    def test_generates_correct_args(self) -> None:
        cmd = build_nccl_test_cmd("/usr/bin/all_reduce_perf", num_gpus=8)
        assert cmd[0] == "/usr/bin/all_reduce_perf"
        assert "-g" in cmd
        assert cmd[cmd.index("-g") + 1] == "8"
        assert "-b" in cmd
        assert "-e" in cmd
        assert "-f" in cmd


# ---------------------------------------------------------------------------
# parse_avg_bus_bandwidth
# ---------------------------------------------------------------------------


_SUMMARY_OUTPUT = """\
# nccl-tests v2.13.8
#    size(B)  count(elems)  type  redop  root  time(us)  algbw(GB/s)  busbw(GB/s)
   1048576         262144  float    sum    -1     78.12        13.42       12.61
   2097152         524288  float    sum    -1    135.21        15.51       14.54
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 150.23
"""

_NO_SUMMARY_OUTPUT = """\
# nccl-tests v2.13.8
#    size(B)  count(elems)  type  redop  root  time(us)  algbw(GB/s)  busbw(GB/s)
   1048576         262144  float    sum    -1     78.12        13.42       12.61
   2097152         524288  float    sum    -1    135.21        15.51       14.54
"""


class TestParseAvgBusBandwidth:
    def test_parses_summary_line(self) -> None:
        bw = parse_avg_bus_bandwidth(_SUMMARY_OUTPUT)
        assert bw == pytest.approx(150.23)

    def test_fallback_to_last_data_row(self) -> None:
        bw = parse_avg_bus_bandwidth(_NO_SUMMARY_OUTPUT)
        assert bw == pytest.approx(14.54)

    def test_returns_none_for_empty_output(self) -> None:
        assert parse_avg_bus_bandwidth("") is None

    def test_returns_none_for_comments_only(self) -> None:
        assert parse_avg_bus_bandwidth("# just a comment\n# another") is None

    def test_returns_none_for_non_numeric_rows(self) -> None:
        output = "header row without numbers\nanother non-numeric line"
        assert parse_avg_bus_bandwidth(output) is None


# ---------------------------------------------------------------------------
# _interpret_nccl_output
# ---------------------------------------------------------------------------


class TestInterpretNcclOutput:
    def test_nonzero_exit_returns_fail(self) -> None:
        result = _interpret_nccl_output(
            stdout="",
            stderr="segfault",
            returncode=1,
            node_id="n0",
            diagnostic_type=_DIAG_TYPE,
            expected_bandwidth_gbps=100.0,
            log_prefix="test",
        )
        assert not result.passed
        assert "exit code 1" in result.details

    def test_parse_failure_returns_fail(self) -> None:
        result = _interpret_nccl_output(
            stdout="garbage output",
            stderr="",
            returncode=0,
            node_id="n0",
            diagnostic_type=_DIAG_TYPE,
            expected_bandwidth_gbps=100.0,
            log_prefix="test",
        )
        assert not result.passed
        assert "parse" in result.details.lower()

    def test_bandwidth_above_threshold_passes(self) -> None:
        result = _interpret_nccl_output(
            stdout=_SUMMARY_OUTPUT,
            stderr="",
            returncode=0,
            node_id="n0",
            diagnostic_type=_DIAG_TYPE,
            expected_bandwidth_gbps=100.0,
            log_prefix="test",
        )
        assert result.passed
        assert "150.23" in result.details

    def test_bandwidth_below_threshold_fails(self) -> None:
        result = _interpret_nccl_output(
            stdout=_SUMMARY_OUTPUT,
            stderr="",
            returncode=0,
            node_id="n0",
            diagnostic_type=_DIAG_TYPE,
            expected_bandwidth_gbps=200.0,
            log_prefix="test",
        )
        assert not result.passed
        assert "150.23" in result.details


# ---------------------------------------------------------------------------
# run_nccl_test (async with mocked subprocess)
# ---------------------------------------------------------------------------


class TestRunNcclTest:
    @pytest.mark.asyncio
    async def test_success_path(self) -> None:
        stdout = _SUMMARY_OUTPUT.encode()
        with patch(
            "miles.utils.ft.agents.diagnostics.utils.nccl_utils.run_subprocess_with_timeout",
            new_callable=AsyncMock,
            return_value=(stdout, b"", 0),
        ):
            result = await run_nccl_test(
                cmd=["fake_binary"],
                node_id="n0",
                diagnostic_type=_DIAG_TYPE,
                expected_bandwidth_gbps=100.0,
                timeout_seconds=60,
                log_prefix="test",
            )
        assert result.passed

    @pytest.mark.asyncio
    async def test_timeout_returns_fail(self) -> None:
        with patch(
            "miles.utils.ft.agents.diagnostics.utils.nccl_utils.run_subprocess_with_timeout",
            new_callable=AsyncMock,
            side_effect=asyncio.TimeoutError(),
        ):
            result = await run_nccl_test(
                cmd=["fake_binary"],
                node_id="n0",
                diagnostic_type=_DIAG_TYPE,
                expected_bandwidth_gbps=100.0,
                timeout_seconds=10,
                log_prefix="test",
            )
        assert not result.passed
        assert "timed out" in result.details

    @pytest.mark.asyncio
    async def test_exec_failure_returns_fail(self) -> None:
        with patch(
            "miles.utils.ft.agents.diagnostics.utils.nccl_utils.run_subprocess_with_timeout",
            new_callable=AsyncMock,
            side_effect=OSError("No such file"),
        ):
            result = await run_nccl_test(
                cmd=["missing_binary"],
                node_id="n0",
                diagnostic_type=_DIAG_TYPE,
                expected_bandwidth_gbps=100.0,
                timeout_seconds=10,
                log_prefix="test",
            )
        assert not result.passed
        assert "execute" in result.details.lower()
