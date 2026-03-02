from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.debug_utils.run_megatron.cli.commands.args import RunArgs
from miles.utils.debug_utils.run_megatron.cli.commands.run import run_impl


def _make_run_args(**overrides: object) -> RunArgs:
    defaults = dict(
        model_type="deepseek_v3",
        hf_checkpoint=Path("/fake/hf"),
        output_dir=Path("/tmp/dump"),
    )
    defaults.update(overrides)
    return RunArgs(**defaults)  # type: ignore[arg-type]


_COMMON_PATCHES = [
    patch("miles.utils.debug_utils.run_megatron.cli.commands.run.exec_command"),
    patch(
        "miles.utils.debug_utils.run_megatron.cli.commands.run.generate_token_ids",
        return_value=list(range(200)),
    ),
    patch(
        "miles.utils.debug_utils.run_megatron.cli.commands.run.write_token_ids_to_tmpfile",
        return_value=Path("/tmp/tokens.json"),
    ),
    patch(
        "miles.utils.debug_utils.run_megatron.cli.worker_executor.resolve_model_script",
        return_value=Path("/repo/scripts/models/deepseek_v3.sh"),
    ),
]


class TestRunImpl:
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.run.exec_command")
    @patch(
        "miles.utils.debug_utils.run_megatron.cli.commands.run.generate_token_ids",
        return_value=list(range(200)),
    )
    @patch(
        "miles.utils.debug_utils.run_megatron.cli.commands.run.write_token_ids_to_tmpfile",
        return_value=Path("/tmp/tokens.json"),
    )
    @patch(
        "miles.utils.debug_utils.run_megatron.cli.worker_executor.resolve_model_script",
        return_value=Path("/repo/scripts/models/deepseek_v3.sh"),
    )
    def test_calls_exec_command(
        self,
        mock_resolve: MagicMock,
        mock_write: MagicMock,
        mock_gen: MagicMock,
        mock_exec: MagicMock,
    ) -> None:
        run_impl(_make_run_args())
        mock_exec.assert_called_once()
        cmd = mock_exec.call_args[0][0]
        assert "torchrun" in cmd

    def test_routing_replay_requires_nproc1(self) -> None:
        with pytest.raises(ValueError, match="single-rank"):
            run_impl(
                _make_run_args(
                    tp=2,
                    routing_replay_dump_path=Path("/dump"),
                )
            )

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.run.exec_command")
    @patch(
        "miles.utils.debug_utils.run_megatron.cli.commands.run.generate_token_ids",
        return_value=list(range(200)),
    )
    @patch(
        "miles.utils.debug_utils.run_megatron.cli.commands.run.write_token_ids_to_tmpfile",
        return_value=Path("/tmp/tokens.json"),
    )
    @patch(
        "miles.utils.debug_utils.run_megatron.cli.worker_executor.resolve_model_script",
        return_value=Path("/repo/scripts/models/deepseek_v3.sh"),
    )
    def test_routing_replay_nproc1_ok(
        self,
        mock_resolve: MagicMock,
        mock_write: MagicMock,
        mock_gen: MagicMock,
        mock_exec: MagicMock,
    ) -> None:
        run_impl(
            _make_run_args(
                tp=1,
                routing_replay_dump_path=Path("/dump"),
            )
        )
        mock_exec.assert_called_once()
