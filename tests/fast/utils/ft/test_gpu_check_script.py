"""Unit tests for _gpu_check_script — pynvml + matmul checks."""
from __future__ import annotations

import json
from dataclasses import asdict
from io import StringIO
from unittest.mock import patch

import pytest

from miles.utils.ft.controller.diagnostics._gpu_check_script import (
    GpuCheckResult,
    NvmlCheckResult,
    _check_nvml,
    _check_single_gpu,
    _generate_matmul_reference,
    main,
)
from tests.fast.utils.ft.conftest import make_mock_pynvml


# ---------------------------------------------------------------------------
# _check_nvml tests
# ---------------------------------------------------------------------------

class TestCheckNvml:
    def test_healthy_gpu(self) -> None:
        mock_pynvml = make_mock_pynvml(device_count=1)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _check_nvml(handle="handle-0")

        assert result.ecc_errors_uncorrectable == 0
        assert result.retired_pages_count == 0
        assert result.power_state_abnormal is False
        assert result.row_remap_failure is False

    def test_ecc_uncorrectable_errors(self) -> None:
        mock_pynvml = make_mock_pynvml(ecc_uncorrectable=5)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _check_nvml(handle="handle-0")

        assert result.ecc_errors_uncorrectable == 5

    def test_retired_pages(self) -> None:
        mock_pynvml = make_mock_pynvml(retired_pages=["page1", "page2", "page3"])
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _check_nvml(handle="handle-0")

        assert result.retired_pages_count == 3

    def test_abnormal_power_state_8(self) -> None:
        mock_pynvml = make_mock_pynvml(power_state=8)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _check_nvml(handle="handle-0")

        assert result.power_state_abnormal is True

    def test_abnormal_power_state_15(self) -> None:
        mock_pynvml = make_mock_pynvml(power_state=15)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _check_nvml(handle="handle-0")

        assert result.power_state_abnormal is True

    def test_normal_power_state(self) -> None:
        mock_pynvml = make_mock_pynvml(power_state=0)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _check_nvml(handle="handle-0")

        assert result.power_state_abnormal is False

    def test_row_remap_failure(self) -> None:
        mock_pynvml = make_mock_pynvml(remap_info=(0, 0, 0, 1))
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _check_nvml(handle="handle-0")

        assert result.row_remap_failure is True


# ---------------------------------------------------------------------------
# _check_single_gpu tests
# ---------------------------------------------------------------------------

_MOCK_MATMUL_REF = (None, None, None)


class TestCheckSingleGpu:
    def test_all_pass(self) -> None:
        mock_pynvml = make_mock_pynvml(device_count=1)
        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._check_matmul",
                return_value=True,
            ),
        ):
            result = _check_single_gpu(
                gpu_index=0, handle="handle-0", matmul_ref=_MOCK_MATMUL_REF,
            )

        assert result.passed is True
        assert result.matmul_passed is True
        assert result.details == "all checks passed"

    def test_ecc_failure(self) -> None:
        mock_pynvml = make_mock_pynvml(ecc_uncorrectable=3)
        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._check_matmul",
                return_value=True,
            ),
        ):
            result = _check_single_gpu(
                gpu_index=0, handle="handle-0", matmul_ref=_MOCK_MATMUL_REF,
            )

        assert result.passed is False
        assert "ECC" in result.details

    def test_matmul_mismatch(self) -> None:
        mock_pynvml = make_mock_pynvml(device_count=1)
        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._check_matmul",
                return_value=False,
            ),
        ):
            result = _check_single_gpu(
                gpu_index=0, handle="handle-0", matmul_ref=_MOCK_MATMUL_REF,
            )

        assert result.passed is False
        assert result.matmul_passed is False
        assert "matmul" in result.details

    def test_multiple_failures(self) -> None:
        mock_pynvml = make_mock_pynvml(
            ecc_uncorrectable=1,
            remap_info=(0, 0, 0, 1),
        )
        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._check_matmul",
                return_value=False,
            ),
        ):
            result = _check_single_gpu(
                gpu_index=0, handle="handle-0", matmul_ref=_MOCK_MATMUL_REF,
            )

        assert result.passed is False
        assert "ECC" in result.details
        assert "row remap" in result.details
        assert "matmul" in result.details

    def test_retired_pages_failure(self) -> None:
        mock_pynvml = make_mock_pynvml(retired_pages=["p1"])
        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._check_matmul",
                return_value=True,
            ),
        ):
            result = _check_single_gpu(
                gpu_index=0, handle="handle-0", matmul_ref=_MOCK_MATMUL_REF,
            )

        assert result.passed is False
        assert "retired" in result.details

    def test_power_state_abnormal_failure(self) -> None:
        mock_pynvml = make_mock_pynvml(power_state=8)
        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._check_matmul",
                return_value=True,
            ),
        ):
            result = _check_single_gpu(
                gpu_index=0, handle="handle-0", matmul_ref=_MOCK_MATMUL_REF,
            )

        assert result.passed is False
        assert "power state" in result.details


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_healthy_output(self) -> None:
        mock_pynvml = make_mock_pynvml(device_count=2)
        stdout_capture = StringIO()

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._check_matmul",
                return_value=True,
            ),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._generate_matmul_reference",
                return_value=_MOCK_MATMUL_REF,
            ),
            patch("sys.stdout", stdout_capture),
        ):
            main()

        output = json.loads(stdout_capture.getvalue())
        assert len(output) == 2
        assert output[0]["gpu_index"] == 0
        assert output[0]["passed"] is True
        assert output[1]["gpu_index"] == 1
        assert output[1]["passed"] is True

    def test_main_mixed_results(self) -> None:
        mock_pynvml = make_mock_pynvml(device_count=2, ecc_uncorrectable=5)
        stdout_capture = StringIO()

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._check_matmul",
                return_value=True,
            ),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._generate_matmul_reference",
                return_value=_MOCK_MATMUL_REF,
            ),
            patch("sys.stdout", stdout_capture),
        ):
            main()

        output = json.loads(stdout_capture.getvalue())
        assert len(output) == 2
        assert output[0]["passed"] is False
        assert output[0]["ecc_errors_uncorrectable"] == 5

    def test_main_per_gpu_exception_produces_failed_result(self) -> None:
        mock_pynvml = make_mock_pynvml(device_count=2)
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = [
            RuntimeError("GPU 0 error"),
            "handle-1",
        ]
        stdout_capture = StringIO()

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._check_matmul",
                return_value=True,
            ),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._generate_matmul_reference",
                return_value=_MOCK_MATMUL_REF,
            ),
            patch("sys.stdout", stdout_capture),
        ):
            main()

        output = json.loads(stdout_capture.getvalue())
        assert len(output) == 2
        assert output[0]["passed"] is False
        assert "GPU 0 error" in output[0]["details"]
        assert output[1]["passed"] is True
        mock_pynvml.nvmlShutdown.assert_called_once()

    def test_main_nvml_shutdown_always_called(self) -> None:
        mock_pynvml = make_mock_pynvml(device_count=0)
        stdout_capture = StringIO()

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._generate_matmul_reference",
                return_value=_MOCK_MATMUL_REF,
            ),
            patch("sys.stdout", stdout_capture),
        ):
            main()

        mock_pynvml.nvmlShutdown.assert_called_once()
        output = json.loads(stdout_capture.getvalue())
        assert output == []


# ---------------------------------------------------------------------------
# GpuCheckResult dataclass tests
# ---------------------------------------------------------------------------

class TestGpuCheckResult:
    def test_serialization(self) -> None:
        result = GpuCheckResult(
            gpu_index=0,
            passed=True,
            ecc_errors_uncorrectable=0,
            retired_pages_count=0,
            power_state_abnormal=False,
            row_remap_failure=False,
            matmul_passed=True,
            details="all checks passed",
        )
        d = asdict(result)
        assert d["gpu_index"] == 0
        assert d["passed"] is True

        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored == d
