"""Unit tests for gpu_check_script — pynvml + compute fingerprint checks."""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict
from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

from tests.fast.utils.ft.utils import make_mock_pynvml

from miles.utils.ft.agents.diagnostics.utils.gpu_check_script import (
    GpuCheckResult,
    _check_nvml,
    _check_single_gpu,
    main,
)

_MOCK_MODEL_AND_INPUT = (None, None, None)
_MOCK_HASH = "abc123"
_GPU_SCRIPT = "miles.utils.ft.agents.diagnostics.utils.gpu_check_script"


@contextmanager
def _pynvml_and_compute(
    mock_pynvml: MagicMock,
    *,
    compute_hash: str = _MOCK_HASH,
    compute_error: Exception | None = None,
) -> Generator[None, None, None]:
    side_effect = compute_error if compute_error else None
    return_value = compute_hash if not compute_error else ""
    with (
        patch.dict("sys.modules", {"pynvml": mock_pynvml}),
        patch(
            f"{_GPU_SCRIPT}._compute_fingerprint",
            side_effect=side_effect,
            return_value=return_value,
        ),
    ):
        yield


def _run_main(
    mock_pynvml: MagicMock,
    *,
    compute_hash: str = _MOCK_HASH,
    compute_error: Exception | None = None,
) -> list[dict[str, Any]]:
    stdout_capture = StringIO()

    with (
        _pynvml_and_compute(mock_pynvml, compute_hash=compute_hash, compute_error=compute_error),
        patch(f"{_GPU_SCRIPT}._build_deterministic_model_and_input", return_value=_MOCK_MODEL_AND_INPUT),
        patch("sys.stdout", stdout_capture),
    ):
        main()

    return json.loads(stdout_capture.getvalue())


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


class TestCheckSingleGpu:
    def test_all_pass(self) -> None:
        mock_pynvml = make_mock_pynvml(device_count=1)
        with _pynvml_and_compute(mock_pynvml):
            result = _check_single_gpu(
                gpu_index=0,
                handle="handle-0",
                model_and_input=_MOCK_MODEL_AND_INPUT,
            )

        assert result.nvml_passed is True
        assert result.compute_hash == _MOCK_HASH
        assert result.compute_error == ""
        assert result.details == "all checks passed"

    def test_ecc_failure(self) -> None:
        mock_pynvml = make_mock_pynvml(ecc_uncorrectable=3)
        with _pynvml_and_compute(mock_pynvml):
            result = _check_single_gpu(
                gpu_index=0,
                handle="handle-0",
                model_and_input=_MOCK_MODEL_AND_INPUT,
            )

        assert result.nvml_passed is False
        assert "ECC" in result.details

    def test_compute_fingerprint_error(self) -> None:
        mock_pynvml = make_mock_pynvml(device_count=1)
        with _pynvml_and_compute(mock_pynvml, compute_error=RuntimeError("cuda error")):
            result = _check_single_gpu(
                gpu_index=0,
                handle="handle-0",
                model_and_input=_MOCK_MODEL_AND_INPUT,
            )

        assert result.nvml_passed is True
        assert result.compute_error != ""
        assert "compute fingerprint failed" in result.details

    def test_multiple_failures(self) -> None:
        mock_pynvml = make_mock_pynvml(
            ecc_uncorrectable=1,
            remap_info=(0, 0, 0, 1),
        )
        with _pynvml_and_compute(mock_pynvml, compute_error=RuntimeError("bad gpu")):
            result = _check_single_gpu(
                gpu_index=0,
                handle="handle-0",
                model_and_input=_MOCK_MODEL_AND_INPUT,
            )

        assert result.nvml_passed is False
        assert "ECC" in result.details
        assert "row remap" in result.details
        assert "compute fingerprint failed" in result.details

    def test_retired_pages_failure(self) -> None:
        mock_pynvml = make_mock_pynvml(retired_pages=["p1"])
        with _pynvml_and_compute(mock_pynvml):
            result = _check_single_gpu(
                gpu_index=0,
                handle="handle-0",
                model_and_input=_MOCK_MODEL_AND_INPUT,
            )

        assert result.nvml_passed is False
        assert "retired" in result.details

    def test_power_state_abnormal_failure(self) -> None:
        mock_pynvml = make_mock_pynvml(power_state=8)
        with _pynvml_and_compute(mock_pynvml):
            result = _check_single_gpu(
                gpu_index=0,
                handle="handle-0",
                model_and_input=_MOCK_MODEL_AND_INPUT,
            )

        assert result.nvml_passed is False
        assert "power state" in result.details


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_healthy_output(self) -> None:
        output = _run_main(make_mock_pynvml(device_count=2))

        assert len(output) == 2
        assert output[0]["gpu_index"] == 0
        assert output[0]["nvml_passed"] is True
        assert output[1]["gpu_index"] == 1
        assert output[1]["nvml_passed"] is True

    def test_main_mixed_results(self) -> None:
        output = _run_main(make_mock_pynvml(device_count=2, ecc_uncorrectable=5))

        assert len(output) == 2
        assert output[0]["nvml_passed"] is False
        assert output[0]["ecc_errors_uncorrectable"] == 5

    def test_main_per_gpu_exception_produces_failed_result(self) -> None:
        mock_pynvml = make_mock_pynvml(device_count=2)
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = [
            RuntimeError("GPU 0 error"),
            "handle-1",
        ]

        output = _run_main(mock_pynvml)

        assert len(output) == 2
        assert "GPU 0 error" in output[0]["details"]
        assert output[1]["nvml_passed"] is True
        mock_pynvml.nvmlShutdown.assert_called_once()

    def test_main_nvml_shutdown_always_called(self) -> None:
        mock_pynvml = make_mock_pynvml(device_count=0)
        stdout_capture = StringIO()

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(f"{_GPU_SCRIPT}._build_deterministic_model_and_input", return_value=_MOCK_MODEL_AND_INPUT),
            patch("sys.stdout", stdout_capture),
        ):
            main()

        mock_pynvml.nvmlShutdown.assert_called_once()
        output = json.loads(stdout_capture.getvalue())
        assert output == []


# ---------------------------------------------------------------------------
# GpuCheckResult dataclass tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Real GPU tests — require a GPU to run
# ---------------------------------------------------------------------------


class TestComputeFingerprintRealGpu:
    """Tests that exercise _compute_fingerprint on a real GPU.

    Validates that the deterministic computation produces consistent,
    non-empty hashes on actual hardware.
    """

    @pytest.mark.requires_gpu
    def test_compute_fingerprint_is_deterministic(self) -> None:
        """Same GPU should produce identical hash across two runs."""
        from miles.utils.ft.agents.diagnostics.utils.gpu_check_script import (
            _build_deterministic_model_and_input,
            _compute_fingerprint,
        )

        model, x, mask = _build_deterministic_model_and_input()
        hash_1 = _compute_fingerprint(0, model, x, mask)
        hash_2 = _compute_fingerprint(0, model, x, mask)

        assert hash_1 == hash_2

    @pytest.mark.requires_gpu
    def test_compute_fingerprint_produces_nonempty_hash(self) -> None:
        from miles.utils.ft.agents.diagnostics.utils.gpu_check_script import (
            _build_deterministic_model_and_input,
            _compute_fingerprint,
        )

        model, x, mask = _build_deterministic_model_and_input()
        h = _compute_fingerprint(0, model, x, mask)

        assert isinstance(h, str)
        assert len(h) == 64

    @pytest.mark.requires_gpu
    def test_main_produces_valid_json_on_real_gpu(self) -> None:
        """Run main() without mocking — output should be valid JSON with real GPU results."""
        import io
        import json as json_mod
        import sys

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main()
        finally:
            sys.stdout = old_stdout

        output = json_mod.loads(captured.getvalue())
        assert isinstance(output, list)
        assert len(output) >= 1

        for item in output:
            assert "gpu_index" in item
            assert "nvml_passed" in item
            assert "compute_hash" in item


class TestGpuCheckResult:
    def test_serialization(self) -> None:
        result = GpuCheckResult(
            gpu_index=0,
            nvml_passed=True,
            ecc_errors_uncorrectable=0,
            retired_pages_count=0,
            power_state_abnormal=False,
            row_remap_failure=False,
            compute_hash="abc123",
            compute_error="",
            details="all checks passed",
        )
        d = asdict(result)
        assert d["gpu_index"] == 0
        assert d["nvml_passed"] is True

        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored == d
