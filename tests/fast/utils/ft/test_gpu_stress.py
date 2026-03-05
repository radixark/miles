from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from miles.utils.ft.e2e.gpu_stress import _stress_loop, app

runner = CliRunner()


class TestStressLoop:
    def test_raises_when_no_cuda_devices(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.device_count.return_value = 0

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with pytest.raises(RuntimeError, match="No CUDA devices available"):
                _stress_loop(duration=1.0)

    def test_allocates_tensors_per_device(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.device.side_effect = lambda name: name

        call_count = 0
        base_time = 1000.0

        def fake_monotonic() -> float:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return base_time
            return base_time + 9999

        with (
            patch.dict("sys.modules", {"torch": mock_torch}),
            patch("time.monotonic", side_effect=fake_monotonic),
        ):
            _stress_loop(duration=1.0)

        assert mock_torch.randn.call_count == 4  # 2 tensors * 2 devices


class TestMain:
    def test_main_parses_duration(self) -> None:
        with patch("miles.utils.ft.e2e.gpu_stress._stress_loop") as mock_loop:
            result = runner.invoke(app, ["--duration", "42.0"])

            assert result.exit_code == 0
            mock_loop.assert_called_once_with(duration=42.0, matrix_size=4096)

    def test_main_parses_matrix_size(self) -> None:
        with patch("miles.utils.ft.e2e.gpu_stress._stress_loop") as mock_loop:
            result = runner.invoke(app, ["--matrix-size", "2048"])

            assert result.exit_code == 0
            mock_loop.assert_called_once_with(duration=3600.0, matrix_size=2048)
