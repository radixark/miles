from __future__ import annotations

from pathlib import Path
from typing import Generator
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


class TestRunImplValidation:
    def test_routing_replay_requires_nproc1(self) -> None:
        with pytest.raises(ValueError, match="single-rank"):
            run_impl(
                _make_run_args(
                    tp=2,
                    routing_replay_dump_path=Path("/dump"),
                )
            )

    def test_routing_replay_pp_gt1_also_fails(self) -> None:
        with pytest.raises(ValueError, match="single-rank"):
            run_impl(
                _make_run_args(
                    pp=2,
                    routing_replay_dump_path=Path("/dump"),
                )
            )


class TestRunImplExecCommand:
    """Only mock exec_command, generate_token_ids, write_token_ids_to_tmpfile,
    and resolve_model_script — let the rest (build_worker_args, build_dumper_env,
    build_torchrun_cmd, ParallelConfig, WorkerScriptArgs) run for real."""

    @pytest.fixture(autouse=True)
    def _patch_externals(self) -> Generator[None, None, None]:
        with (
            patch(
                "miles.utils.debug_utils.run_megatron.cli.commands.run.exec_command"
            ) as mock_exec,
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
        ):
            self.mock_exec = mock_exec
            yield

    def test_cmd_contains_torchrun(self) -> None:
        run_impl(_make_run_args())
        self.mock_exec.assert_called_once()
        cmd = self.mock_exec.call_args[0][0]
        assert "torchrun" in cmd

    def test_cmd_contains_parallel_config(self) -> None:
        run_impl(_make_run_args(tp=4, pp=2))
        cmd = self.mock_exec.call_args[0][0]
        assert "--tensor-model-parallel-size 4" in cmd
        assert "--pipeline-model-parallel-size 2" in cmd
        assert "--nproc-per-node 8" in cmd

    def test_cmd_contains_dumper_env(self) -> None:
        run_impl(_make_run_args(output_dir=Path("/out/test"), dumper_filter="logits"))
        cmd = self.mock_exec.call_args[0][0]
        assert "DUMPER_ENABLE=1" in cmd
        assert "DUMPER_DIR=/out/test" in cmd
        assert "DUMPER_FILTER=logits" in cmd

    def test_cmd_contains_script_args(self) -> None:
        run_impl(_make_run_args(role="critic", top_k=5))
        cmd = self.mock_exec.call_args[0][0]
        assert "--script-role critic" in cmd
        assert "--script-top-k 5" in cmd

    def test_routing_replay_nproc1_ok(self) -> None:
        run_impl(
            _make_run_args(
                tp=1,
                routing_replay_dump_path=Path("/dump"),
            )
        )
        cmd = self.mock_exec.call_args[0][0]
        assert "--use-routing-replay" in cmd

    def test_backward_enables_grad_env(self) -> None:
        run_impl(_make_run_args(run_backward=True))
        cmd = self.mock_exec.call_args[0][0]
        assert "DUMPER_ENABLE_MODEL_GRAD=1" in cmd
