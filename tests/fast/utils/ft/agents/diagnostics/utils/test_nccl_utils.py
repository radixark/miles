"""Tests for NCCL test helpers: command builder, output parser, subprocess runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from miles.utils.ft.agents.diagnostics.utils.nccl_utils import (
    _interpret_nccl_output,
    _parse_json_avg_bus_bandwidth,
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

    def test_includes_json_flag_when_path_provided(self) -> None:
        path = Path("/tmp/nccl_output.json")
        cmd = build_nccl_test_cmd("/usr/bin/all_reduce_perf", num_gpus=8, json_output_path=path)
        assert "-J" in cmd
        assert cmd[cmd.index("-J") + 1] == "/tmp/nccl_output.json"

    def test_no_json_flag_when_path_is_none(self) -> None:
        cmd = build_nccl_test_cmd("/usr/bin/all_reduce_perf", num_gpus=8, json_output_path=None)
        assert "-J" not in cmd


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
# _parse_json_avg_bus_bandwidth
# ---------------------------------------------------------------------------


class TestParseJsonAvgBusBandwidth:
    def test_parses_valid_json(self, tmp_path: Path) -> None:
        json_file = tmp_path / "output.json"
        json_file.write_text(
            json.dumps(
                {
                    "average_bus_bandwidth": {"bandwidth": 380.50},
                }
            )
        )
        assert _parse_json_avg_bus_bandwidth(json_file) == pytest.approx(380.50)

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        assert _parse_json_avg_bus_bandwidth(tmp_path / "nonexistent.json") is None

    def test_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        json_file = tmp_path / "bad.json"
        json_file.write_text("not valid json{{{")
        assert _parse_json_avg_bus_bandwidth(json_file) is None

    def test_returns_none_when_key_missing(self, tmp_path: Path) -> None:
        json_file = tmp_path / "incomplete.json"
        json_file.write_text(json.dumps({"some_other_key": 42}))
        assert _parse_json_avg_bus_bandwidth(json_file) is None

    def test_returns_none_when_bandwidth_not_numeric(self, tmp_path: Path) -> None:
        json_file = tmp_path / "bad_value.json"
        json_file.write_text(
            json.dumps(
                {
                    "average_bus_bandwidth": {"bandwidth": "not_a_number"},
                }
            )
        )
        assert _parse_json_avg_bus_bandwidth(json_file) is None


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

    def test_json_result_preferred_over_text(self, tmp_path: Path) -> None:
        json_file = tmp_path / "output.json"
        json_file.write_text(
            json.dumps(
                {
                    "average_bus_bandwidth": {"bandwidth": 999.99},
                }
            )
        )
        result = _interpret_nccl_output(
            stdout=_SUMMARY_OUTPUT,
            stderr="",
            returncode=0,
            node_id="n0",
            diagnostic_type=_DIAG_TYPE,
            expected_bandwidth_gbps=100.0,
            log_prefix="test",
            json_output_path=json_file,
        )
        assert result.passed
        assert "999.99" in result.details

    def test_json_fallback_to_text_when_json_fails(self, tmp_path: Path) -> None:
        json_file = tmp_path / "bad.json"
        json_file.write_text("invalid json")
        result = _interpret_nccl_output(
            stdout=_SUMMARY_OUTPUT,
            stderr="",
            returncode=0,
            node_id="n0",
            diagnostic_type=_DIAG_TYPE,
            expected_bandwidth_gbps=100.0,
            log_prefix="test",
            json_output_path=json_file,
        )
        assert result.passed
        assert "150.23" in result.details


# ---------------------------------------------------------------------------
# run_nccl_test (async with mocked subprocess)
# ---------------------------------------------------------------------------


class TestRunNcclTest:
    @pytest.mark.asyncio
    async def test_success_path_falls_back_to_text(self) -> None:
        """When JSON file is empty (subprocess mock doesn't write it), text fallback works."""
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
        assert "150.23" in result.details

    @pytest.mark.asyncio
    async def test_success_path_uses_json_when_available(self, tmp_path: Path) -> None:
        """When subprocess writes valid JSON, result uses JSON bandwidth."""
        json_file = tmp_path / "nccl_out.json"
        json_data = json.dumps({"average_bus_bandwidth": {"bandwidth": 420.0}})

        async def mock_subprocess(cmd: list[str], **kwargs: object) -> tuple[bytes, bytes, int]:
            json_file.write_text(json_data)
            return (_SUMMARY_OUTPUT.encode(), b"", 0)

        with (
            patch(
                "miles.utils.ft.agents.diagnostics.utils.nccl_utils.tempfile.mkstemp",
                return_value=(0, str(json_file)),
            ),
            patch("miles.utils.ft.agents.diagnostics.utils.nccl_utils.os.close"),
            patch(
                "miles.utils.ft.agents.diagnostics.utils.nccl_utils.run_subprocess_with_timeout",
                side_effect=mock_subprocess,
            ),
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
        assert "420.00" in result.details

    @pytest.mark.asyncio
    async def test_cmd_includes_json_flag(self) -> None:
        """run_nccl_test appends -J <path> to the command."""
        captured_cmd: list[str] = []

        async def capture_cmd(cmd: list[str], **kwargs: object) -> tuple[bytes, bytes, int]:
            captured_cmd.extend(cmd)
            return (_SUMMARY_OUTPUT.encode(), b"", 0)

        with patch(
            "miles.utils.ft.agents.diagnostics.utils.nccl_utils.run_subprocess_with_timeout",
            side_effect=capture_cmd,
        ):
            await run_nccl_test(
                cmd=["fake_binary", "-g", "8"],
                node_id="n0",
                diagnostic_type=_DIAG_TYPE,
                expected_bandwidth_gbps=100.0,
                timeout_seconds=60,
                log_prefix="test",
            )

        assert "-J" in captured_cmd
        json_path = captured_cmd[captured_cmd.index("-J") + 1]
        assert json_path.endswith(".json")

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
