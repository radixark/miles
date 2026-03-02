from pathlib import Path
from unittest.mock import MagicMock, patch

from miles.utils.debug_utils.run_megatron.cli.commands.args import CompareArgs
from miles.utils.debug_utils.run_megatron.cli.commands.compare import compare_impl


def _make_compare_args(**overrides: object) -> CompareArgs:
    defaults = dict(
        baseline_dir=Path("/baseline"),
        target_dir=Path("/target"),
    )
    defaults.update(overrides)
    return CompareArgs(**defaults)  # type: ignore[arg-type]


class TestCompareImpl:
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.exec_command")
    def test_calls_comparator(self, mock_exec: MagicMock) -> None:
        compare_impl(_make_compare_args())
        mock_exec.assert_called_once()
        cmd = mock_exec.call_args[0][0]
        assert "sglang.srt.debug_utils.comparator" in cmd

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.exec_command")
    def test_required_args(self, mock_exec: MagicMock) -> None:
        compare_impl(_make_compare_args())
        cmd = mock_exec.call_args[0][0]
        assert "--baseline-path" in cmd
        assert "--target-path" in cmd
        assert "/baseline" in cmd
        assert "/target" in cmd

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.exec_command")
    def test_optional_args_included(self, mock_exec: MagicMock) -> None:
        compare_impl(
            _make_compare_args(
                override_baseline_dims="b s h",
                override_target_dims="b s v",
                patch_config=Path("/patch.yaml"),
                diff_threshold=0.01,
            )
        )
        cmd = mock_exec.call_args[0][0]
        assert "--override-baseline-dims" in cmd
        assert "--override-target-dims" in cmd
        assert "--patch-config" in cmd
        assert "--diff-threshold" in cmd

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.exec_command")
    def test_optional_args_excluded(self, mock_exec: MagicMock) -> None:
        compare_impl(_make_compare_args())
        cmd = mock_exec.call_args[0][0]
        assert "--override-baseline-dims" not in cmd
        assert "--override-target-dims" not in cmd
        assert "--patch-config" not in cmd
        assert "--diff-threshold" not in cmd
