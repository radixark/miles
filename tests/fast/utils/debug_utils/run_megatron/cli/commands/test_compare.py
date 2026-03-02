from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.debug_utils.run_megatron.cli.commands.args import CompareArgs
from miles.utils.debug_utils.run_megatron.cli.commands.compare import compare_impl


def _make_compare_args(**overrides: object) -> CompareArgs:
    defaults = dict(
        baseline_dir=Path("/baseline"),
        target_dir=Path("/target"),
    )
    defaults.update(overrides)
    return CompareArgs(**defaults)  # type: ignore[arg-type]


class TestActivationComparison:
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.exec_command")
    def test_calls_comparator(self, mock_exec: MagicMock) -> None:
        compare_impl(_make_compare_args())
        mock_exec.assert_called_once()
        cmd = mock_exec.call_args[0][0]
        assert "sglang.srt.debug_utils.comparator" in cmd

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.exec_command")
    def test_required_args_in_command(self, mock_exec: MagicMock) -> None:
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


class TestActivationFailure:
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.exec_command")
    def test_activation_failure_exits(self, mock_exec: MagicMock) -> None:
        mock_exec.side_effect = subprocess.CalledProcessError(returncode=1, cmd="test")
        with pytest.raises(SystemExit) as exc_info:
            compare_impl(_make_compare_args())
        assert exc_info.value.code == 1


class TestLogprobBranch:
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.compare_logprobs")
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.exec_command")
    def test_logprob_called_when_dirs_provided(
        self, mock_exec: MagicMock, mock_logprob: MagicMock
    ) -> None:
        mock_logprob.return_value = True
        compare_impl(_make_compare_args(
            baseline_logprob_dir=Path("/bl_logprob"),
            target_logprob_dir=Path("/tg_logprob"),
            logprob_threshold=0.005,
        ))
        mock_logprob.assert_called_once_with(
            baseline_dir=Path("/bl_logprob"),
            target_dir=Path("/tg_logprob"),
            threshold=0.005,
        )

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.compare_logprobs")
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.exec_command")
    def test_logprob_not_called_when_dirs_missing(
        self, mock_exec: MagicMock, mock_logprob: MagicMock
    ) -> None:
        compare_impl(_make_compare_args())
        mock_logprob.assert_not_called()

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.compare_logprobs")
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.exec_command")
    def test_logprob_default_threshold(
        self, mock_exec: MagicMock, mock_logprob: MagicMock
    ) -> None:
        mock_logprob.return_value = True
        compare_impl(_make_compare_args(
            baseline_logprob_dir=Path("/bl"),
            target_logprob_dir=Path("/tg"),
            logprob_threshold=None,
        ))
        call_kwargs = mock_logprob.call_args[1]
        assert call_kwargs["threshold"] == 1e-3

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.compare_logprobs")
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.exec_command")
    def test_logprob_failure_exits(
        self, mock_exec: MagicMock, mock_logprob: MagicMock
    ) -> None:
        mock_logprob.return_value = False
        with pytest.raises(SystemExit) as exc_info:
            compare_impl(_make_compare_args(
                baseline_logprob_dir=Path("/bl"),
                target_logprob_dir=Path("/tg"),
            ))
        assert exc_info.value.code == 1

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.compare_logprobs")
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.exec_command")
    def test_activation_pass_logprob_fail(
        self, mock_exec: MagicMock, mock_logprob: MagicMock
    ) -> None:
        mock_logprob.return_value = False
        with pytest.raises(SystemExit):
            compare_impl(_make_compare_args(
                baseline_logprob_dir=Path("/bl"),
                target_logprob_dir=Path("/tg"),
            ))

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.compare_logprobs")
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.exec_command")
    def test_activation_fail_logprob_pass(
        self, mock_exec: MagicMock, mock_logprob: MagicMock
    ) -> None:
        mock_exec.side_effect = subprocess.CalledProcessError(returncode=1, cmd="test")
        mock_logprob.return_value = True
        with pytest.raises(SystemExit):
            compare_impl(_make_compare_args(
                baseline_logprob_dir=Path("/bl"),
                target_logprob_dir=Path("/tg"),
            ))
