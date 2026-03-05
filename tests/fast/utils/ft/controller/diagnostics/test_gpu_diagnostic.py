"""Unit tests for GpuDiagnostic — subprocess-based GPU health check."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from unittest.mock import AsyncMock, patch

import pytest

from miles.utils.ft.controller.diagnostics.gpu_check_script import GpuCheckResult
from miles.utils.ft.controller.diagnostics.gpu_diagnostic import GpuDiagnostic


def _make_gpu_result(
    *,
    gpu_index: int = 0,
    passed: bool = True,
    ecc_errors_uncorrectable: int = 0,
    retired_pages_count: int = 0,
    power_state_abnormal: bool = False,
    row_remap_failure: bool = False,
    matmul_passed: bool = True,
    details: str = "all checks passed",
) -> dict[str, object]:
    return asdict(GpuCheckResult(
        gpu_index=gpu_index,
        passed=passed,
        ecc_errors_uncorrectable=ecc_errors_uncorrectable,
        retired_pages_count=retired_pages_count,
        power_state_abnormal=power_state_abnormal,
        row_remap_failure=row_remap_failure,
        matmul_passed=matmul_passed,
        details=details,
    ))


def _mock_subprocess(
    stdout: str = "[]",
    returncode: int = 0,
    stderr: str = "",
) -> AsyncMock:
    process = AsyncMock()
    process.communicate.return_value = (
        stdout.encode(),
        stderr.encode(),
    )
    process.returncode = returncode
    process.kill = AsyncMock()
    process.wait = AsyncMock()
    return process


class TestGpuDiagnosticAllPass:
    @pytest.mark.anyio
    async def test_all_gpus_pass(self) -> None:
        results = [
            _make_gpu_result(gpu_index=0),
            _make_gpu_result(gpu_index=1),
        ]
        process = _mock_subprocess(stdout=json.dumps(results))

        with patch("asyncio.create_subprocess_exec", return_value=process):
            diag = GpuDiagnostic()
            result = await diag.run(node_id="node-0")

        assert result.passed is True
        assert result.diagnostic_type == "gpu"
        assert result.node_id == "node-0"
        assert "all GPU checks passed" in result.details


class TestGpuDiagnosticEccFailure:
    @pytest.mark.anyio
    async def test_ecc_uncorrectable(self) -> None:
        results = [
            _make_gpu_result(
                gpu_index=0,
                passed=False,
                ecc_errors_uncorrectable=5,
                details="uncorrectable ECC errors: 5",
            ),
            _make_gpu_result(gpu_index=1),
        ]
        process = _mock_subprocess(stdout=json.dumps(results))

        with patch("asyncio.create_subprocess_exec", return_value=process):
            diag = GpuDiagnostic()
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "ECC" in result.details
        assert "GPU 0" in result.details


class TestGpuDiagnosticMatmulFailure:
    @pytest.mark.anyio
    async def test_matmul_mismatch(self) -> None:
        results = [
            _make_gpu_result(
                gpu_index=2,
                passed=False,
                matmul_passed=False,
                details="matmul mismatch",
            ),
        ]
        process = _mock_subprocess(stdout=json.dumps(results))

        with patch("asyncio.create_subprocess_exec", return_value=process):
            diag = GpuDiagnostic()
            result = await diag.run(node_id="node-1")

        assert result.passed is False
        assert "matmul" in result.details


class TestGpuDiagnosticRetiredPages:
    @pytest.mark.anyio
    async def test_retired_pages_detected(self) -> None:
        results = [
            _make_gpu_result(
                gpu_index=0,
                passed=False,
                retired_pages_count=3,
                details="retired pages: 3",
            ),
        ]
        process = _mock_subprocess(stdout=json.dumps(results))

        with patch("asyncio.create_subprocess_exec", return_value=process):
            diag = GpuDiagnostic()
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "retired" in result.details


class TestGpuDiagnosticPowerState:
    @pytest.mark.anyio
    async def test_abnormal_power_state(self) -> None:
        results = [
            _make_gpu_result(
                gpu_index=0,
                passed=False,
                power_state_abnormal=True,
                details="abnormal power state",
            ),
        ]
        process = _mock_subprocess(stdout=json.dumps(results))

        with patch("asyncio.create_subprocess_exec", return_value=process):
            diag = GpuDiagnostic()
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "power state" in result.details


class TestGpuDiagnosticRowRemapFailure:
    @pytest.mark.anyio
    async def test_row_remap_failure(self) -> None:
        results = [
            _make_gpu_result(
                gpu_index=3,
                passed=False,
                row_remap_failure=True,
                details="row remap failure",
            ),
        ]
        process = _mock_subprocess(stdout=json.dumps(results))

        with patch("asyncio.create_subprocess_exec", return_value=process):
            diag = GpuDiagnostic()
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "row remap" in result.details


class TestGpuDiagnosticTimeout:
    @pytest.mark.anyio
    async def test_subprocess_timeout(self) -> None:
        process = AsyncMock()
        process.communicate.side_effect = asyncio.TimeoutError()
        process.kill = AsyncMock()
        process.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=process):
            diag = GpuDiagnostic()
            result = await diag.run(node_id="node-0", timeout_seconds=1)

        assert result.passed is False
        assert "timed out" in result.details
        process.kill.assert_called_once()
        process.wait.assert_called_once()

    @pytest.mark.anyio
    async def test_timeout_kill_failure_still_returns_result(self) -> None:
        process = AsyncMock()
        process.communicate.side_effect = asyncio.TimeoutError()
        process.kill.side_effect = OSError("kill failed")
        process.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=process):
            diag = GpuDiagnostic()
            result = await diag.run(node_id="node-0", timeout_seconds=1)

        assert result.passed is False
        assert "timed out" in result.details


class TestGpuDiagnosticEmptyResults:
    @pytest.mark.anyio
    async def test_empty_gpu_list_fails(self) -> None:
        process = _mock_subprocess(stdout="[]")

        with patch("asyncio.create_subprocess_exec", return_value=process):
            diag = GpuDiagnostic()
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "no results" in result.details


class TestGpuDiagnosticProcessCrash:
    @pytest.mark.anyio
    async def test_nonzero_exit_code(self) -> None:
        process = _mock_subprocess(
            stdout="",
            returncode=1,
            stderr="Traceback: pynvml not found",
        )

        with patch("asyncio.create_subprocess_exec", return_value=process):
            diag = GpuDiagnostic()
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "gpu check process failed" in result.details
        assert "pynvml" in result.details


class TestGpuDiagnosticInvalidJson:
    @pytest.mark.anyio
    async def test_invalid_json_output(self) -> None:
        process = _mock_subprocess(stdout="not json at all")

        with patch("asyncio.create_subprocess_exec", return_value=process):
            diag = GpuDiagnostic()
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "invalid output" in result.details


class TestGpuDiagnosticMultiGpuPartialFail:
    @pytest.mark.anyio
    async def test_some_pass_some_fail(self) -> None:
        results = [
            _make_gpu_result(gpu_index=0),
            _make_gpu_result(
                gpu_index=1,
                passed=False,
                ecc_errors_uncorrectable=2,
                details="uncorrectable ECC errors: 2",
            ),
            _make_gpu_result(gpu_index=2),
            _make_gpu_result(
                gpu_index=3,
                passed=False,
                matmul_passed=False,
                details="matmul mismatch",
            ),
        ]
        process = _mock_subprocess(stdout=json.dumps(results))

        with patch("asyncio.create_subprocess_exec", return_value=process):
            diag = GpuDiagnostic()
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "GPU 1" in result.details
        assert "GPU 3" in result.details
        assert "GPU 0" not in result.details
        assert "GPU 2" not in result.details


class TestGpuDiagnosticLaunchFailure:
    @pytest.mark.anyio
    async def test_subprocess_launch_error(self) -> None:
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=OSError("exec failed"),
        ):
            diag = GpuDiagnostic()
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "failed to launch" in result.details
