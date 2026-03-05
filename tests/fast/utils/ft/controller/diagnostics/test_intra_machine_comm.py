"""Tests for IntraMachineCommDiagnostic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miles.utils.ft.controller.diagnostics.intra_machine_comm import IntraMachineCommDiagnostic
from miles.utils.ft.controller.diagnostics.nccl_utils import parse_avg_bus_bandwidth

# ---------------------------------------------------------------------------
# Sample nccl-tests output fixtures
# ---------------------------------------------------------------------------

NCCL_OUTPUT_WITH_SUMMARY = """\
# nThread 1 nGpu 8 minBytes 1048576 maxBytes 1073741824 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid  12345 on     node-0 device  0 [0x07] NVIDIA H200 141GB HBM3e
#  Rank  1 Group  0 Pid  12345 on     node-0 device  1 [0x0a] NVIDIA H200 141GB HBM3e
#  Rank  2 Group  0 Pid  12345 on     node-0 device  2 [0x47] NVIDIA H200 141GB HBM3e
#  Rank  3 Group  0 Pid  12345 on     node-0 device  3 [0x4c] NVIDIA H200 141GB HBM3e
#  Rank  4 Group  0 Pid  12345 on     node-0 device  4 [0x87] NVIDIA H200 141GB HBM3e
#  Rank  5 Group  0 Pid  12345 on     node-0 device  5 [0x8b] NVIDIA H200 141GB HBM3e
#  Rank  6 Group  0 Pid  12345 on     node-0 device  6 [0xc7] NVIDIA H200 141GB HBM3e
#  Rank  7 Group  0 Pid  12345 on     node-0 device  7 [0xcb] NVIDIA H200 141GB HBM3e
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
     1048576        262144     float     sum      -1    101.2   10.36    9.06      0    100.8   10.40    9.10      0
     2097152        524288     float     sum      -1    102.5   20.46   17.90      0    101.9   20.58   18.01      0
     4194304       1048576     float     sum      -1    106.3   39.46   34.53      0    105.7   39.68   34.72      0
     8388608       2097152     float     sum      -1    113.4   73.99   64.74      0    112.8   74.37   65.07      0
    16777216       4194304     float     sum      -1    128.7  130.35  114.06      0    127.9  131.17  114.77      0
    33554432       8388608     float     sum      -1    162.1  207.00  181.13      0    161.2  208.15  182.13      0
    67108864      16777216     float     sum      -1    234.5  286.18  250.41      0    233.7  287.16  251.26      0
   134217728      33554432     float     sum      -1    380.2  353.05  308.92      0    379.1  354.08  309.82      0
   268435456      67108864     float     sum      -1    715.8  375.04  328.16      0    714.3  375.83  328.85      0
   536870912     134217728     float     sum      -1   1397.2  384.18  336.16      0   1395.0  384.79  336.69      0
  1073741824     268435456     float     sum      -1   2766.8  388.05  339.55      0   2763.4  388.53  339.96      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 380.50
#
"""

NCCL_OUTPUT_LOW_BW = """\
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
     1048576        262144     float     sum      -1    201.2    5.21    4.56      0    200.8    5.22    4.57      0
     2097152        524288     float     sum      -1    202.5   10.35    9.06      0    201.9   10.39    9.09      0
  1073741824     268435456     float     sum      -1   5766.8  186.17  162.90      0   5763.4  186.28  162.99      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 120.30
#
"""

NCCL_OUTPUT_NO_SUMMARY = """\
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
     1048576        262144     float     sum      -1    101.2   10.36    9.06      0    100.8   10.40    9.10      0
  1073741824     268435456     float     sum      -1   2766.8  388.05  339.55      0   2763.4  388.53  339.96      0
"""

NCCL_OUTPUT_GARBAGE = """\
This is not valid nccl-tests output
Some random text here
"""


# ---------------------------------------------------------------------------
# parse_avg_bus_bandwidth unit tests
# ---------------------------------------------------------------------------


class TestParseAvgBusBandwidth:
    def test_parse_from_summary_line(self) -> None:
        result = parse_avg_bus_bandwidth(NCCL_OUTPUT_WITH_SUMMARY)
        assert result == pytest.approx(380.50)

    def test_parse_from_low_bw_summary(self) -> None:
        result = parse_avg_bus_bandwidth(NCCL_OUTPUT_LOW_BW)
        assert result == pytest.approx(120.30)

    def test_fallback_to_last_row(self) -> None:
        result = parse_avg_bus_bandwidth(NCCL_OUTPUT_NO_SUMMARY)
        # Last data row busbw column (index 7) = 339.55
        assert result == pytest.approx(339.55)

    def test_garbage_output_returns_none(self) -> None:
        result = parse_avg_bus_bandwidth(NCCL_OUTPUT_GARBAGE)
        assert result is None

    def test_empty_output_returns_none(self) -> None:
        result = parse_avg_bus_bandwidth("")
        assert result is None


# ---------------------------------------------------------------------------
# Subprocess mock helpers
# ---------------------------------------------------------------------------


def _make_mock_process(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> AsyncMock:
    process = AsyncMock()
    process.communicate.return_value = (
        stdout.encode(),
        stderr.encode(),
    )
    process.returncode = returncode
    return process


# ---------------------------------------------------------------------------
# IntraMachineCommDiagnostic.run() tests
# ---------------------------------------------------------------------------


class TestIntraMachineCommDiagnostic:
    async def test_pass_when_bandwidth_above_threshold(self) -> None:
        diag = IntraMachineCommDiagnostic(expected_bandwidth_gbps=350.0)
        mock_proc = _make_mock_process(stdout=NCCL_OUTPUT_WITH_SUMMARY)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await diag.run(node_id="node-0")

        assert result.passed is True
        assert result.node_id == "node-0"
        assert result.diagnostic_type == "intra_machine"
        assert "380.50" in result.details
        assert "350.00" in result.details

    async def test_fail_when_bandwidth_below_threshold(self) -> None:
        diag = IntraMachineCommDiagnostic(expected_bandwidth_gbps=350.0)
        mock_proc = _make_mock_process(stdout=NCCL_OUTPUT_LOW_BW)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await diag.run(node_id="node-1")

        assert result.passed is False
        assert "120.30" in result.details
        assert "350.00" in result.details

    async def test_fail_when_binary_not_found(self) -> None:
        diag = IntraMachineCommDiagnostic()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("no such file"),
        ):
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "failed to execute" in result.details

    async def test_fail_when_subprocess_returns_nonzero(self) -> None:
        diag = IntraMachineCommDiagnostic()
        mock_proc = _make_mock_process(
            stdout="",
            stderr="NCCL WARN: some error",
            returncode=1,
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "NCCL WARN" in result.details
        assert "exit code 1" in result.details

    async def test_fail_when_output_unparseable(self) -> None:
        diag = IntraMachineCommDiagnostic()
        mock_proc = _make_mock_process(stdout=NCCL_OUTPUT_GARBAGE)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "failed to parse bandwidth" in result.details

    async def test_fail_on_timeout(self) -> None:
        diag = IntraMachineCommDiagnostic()
        mock_proc = AsyncMock()
        mock_proc.communicate.side_effect = asyncio.TimeoutError()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                result = await diag.run(node_id="node-0", timeout_seconds=5)

        assert result.passed is False
        assert "timed out" in result.details
        assert "5s" in result.details
        mock_proc.kill.assert_called_once()
        mock_proc.wait.assert_called_once()

    async def test_custom_num_gpus(self) -> None:
        diag = IntraMachineCommDiagnostic(num_gpus=4)
        mock_proc = _make_mock_process(stdout=NCCL_OUTPUT_WITH_SUMMARY)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await diag.run(node_id="node-0")

        call_args = mock_exec.call_args
        assert "-g" in call_args.args
        g_index = list(call_args.args).index("-g")
        assert call_args.args[g_index + 1] == "4"

    async def test_node_id_in_result(self) -> None:
        diag = IntraMachineCommDiagnostic(expected_bandwidth_gbps=100.0)
        mock_proc = _make_mock_process(stdout=NCCL_OUTPUT_WITH_SUMMARY)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await diag.run(node_id="my-special-node")

        assert result.node_id == "my-special-node"
        assert result.diagnostic_type == "intra_machine"

    async def test_custom_binary_name(self) -> None:
        diag = IntraMachineCommDiagnostic(nccl_test_binary="/opt/nccl/all_reduce_perf")
        mock_proc = _make_mock_process(stdout=NCCL_OUTPUT_WITH_SUMMARY)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await diag.run(node_id="node-0")

        assert mock_exec.call_args.args[0] == "/opt/nccl/all_reduce_perf"

    async def test_bandwidth_exactly_at_threshold_passes(self) -> None:
        diag = IntraMachineCommDiagnostic(expected_bandwidth_gbps=380.50)
        mock_proc = _make_mock_process(stdout=NCCL_OUTPUT_WITH_SUMMARY)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await diag.run(node_id="node-0")

        assert result.passed is True

    async def test_fail_when_permission_denied(self) -> None:
        diag = IntraMachineCommDiagnostic()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=PermissionError("permission denied"),
        ):
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "failed to execute" in result.details

    async def test_stderr_truncated_at_500_chars(self) -> None:
        diag = IntraMachineCommDiagnostic()
        long_stderr = "E" * 600
        mock_proc = _make_mock_process(
            stdout="",
            stderr=long_stderr,
            returncode=1,
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert len(result.details) < 600
