"""Tests for InterMachineCommDiagnostic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from miles.utils.ft.controller.diagnostics.nccl.inter_machine import InterMachineCommDiagnostic
from tests.fast.utils.ft.helpers import make_mock_subprocess

SAMPLE_NCCL_OUTPUT_HIGH_BW = """\
# nThread 1 nGpus 8 minBytes 1048576 maxBytes 1073741824 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid 123456 on node-0 device  0 [0x07] NVIDIA H100 80GB HBM3
#  Rank  1 Group  0 Pid 123456 on node-0 device  1 [0x0a] NVIDIA H100 80GB HBM3
#
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
     1048576        262144     float     sum      -1    123.4    8.50   45.00      0    124.5    8.42   44.90      0
     2097152        524288     float     sum      -1    234.5   8.94   47.00      0    235.6    8.90   46.80      0
# Avg bus bandwidth    : 45.5000
"""

SAMPLE_NCCL_OUTPUT_LOW_BW = """\
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
     1048576        262144     float     sum      -1    500.0    2.10   10.00      0    501.0    2.09    9.90      0
# Avg bus bandwidth    : 10.0000
"""


def _make_diag(**kwargs: object) -> InterMachineCommDiagnostic:
    kwargs.setdefault("master_addr", "10.0.0.1")
    return InterMachineCommDiagnostic(**kwargs)  # type: ignore[arg-type]


class TestInterMachineCommDiagnostic:
    async def test_pass_when_bandwidth_above_threshold(self) -> None:
        diag = _make_diag(expected_bandwidth_gbps=40.0, master_port=29500)
        mock_proc = make_mock_subprocess(stdout=SAMPLE_NCCL_OUTPUT_HIGH_BW.encode())

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await diag.run(node_id="node-0", timeout_seconds=180)

        assert result.passed is True
        assert result.node_id == "node-0"
        assert result.diagnostic_type == "inter_machine"
        assert "45.50" in result.details

    async def test_fail_when_bandwidth_below_threshold(self) -> None:
        diag = _make_diag(expected_bandwidth_gbps=40.0)
        mock_proc = make_mock_subprocess(stdout=SAMPLE_NCCL_OUTPUT_LOW_BW.encode())

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "10.00" in result.details

    async def test_fail_when_binary_not_found(self) -> None:
        diag = _make_diag()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=OSError("No such file"),
        ):
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "failed to execute" in result.details

    async def test_fail_when_subprocess_returns_nonzero(self) -> None:
        diag = _make_diag()
        mock_proc = make_mock_subprocess(stderr=b"NCCL error", returncode=1)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "exit code 1" in result.details

    async def test_fail_when_output_unparseable(self) -> None:
        diag = _make_diag()
        mock_proc = make_mock_subprocess(stdout=b"garbage output")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "failed to parse" in result.details

    async def test_timeout_handling(self) -> None:
        diag = _make_diag()
        mock_proc = make_mock_subprocess()
        mock_proc.communicate.side_effect = asyncio.TimeoutError()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await diag.run(node_id="node-0", timeout_seconds=30)

        assert result.passed is False
        assert "failed to execute all_gather_perf" in result.details
        mock_proc.kill.assert_called_once()
        mock_proc.wait.assert_awaited_once()

    async def test_environment_variables_set(self) -> None:
        diag = _make_diag(master_port=29501)
        mock_proc = make_mock_subprocess(stdout=SAMPLE_NCCL_OUTPUT_HIGH_BW.encode())
        captured_env: dict[str, str] = {}

        async def capture_exec(*args: object, **kwargs: object) -> AsyncMock:
            env = kwargs.get("env", {})
            assert isinstance(env, dict)
            captured_env.update(env)
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=capture_exec):
            await diag.run(node_id="node-0")

        assert captured_env["MASTER_ADDR"] == "10.0.0.1"
        assert captured_env["MASTER_PORT"] == "29501"

    async def test_custom_threshold(self) -> None:
        diag = _make_diag(expected_bandwidth_gbps=5.0)
        mock_proc = make_mock_subprocess(stdout=SAMPLE_NCCL_OUTPUT_LOW_BW.encode())

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await diag.run(node_id="node-0")

        assert result.passed is True
        assert "10.00" in result.details

    async def test_node_id_in_result(self) -> None:
        diag = _make_diag()
        mock_proc = make_mock_subprocess(stdout=SAMPLE_NCCL_OUTPUT_HIGH_BW.encode())

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await diag.run(node_id="my-special-node")

        assert result.node_id == "my-special-node"
        assert result.diagnostic_type == "inter_machine"
