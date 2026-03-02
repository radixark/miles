from pathlib import Path
from unittest.mock import patch

import pytest

from miles.utils.debug_utils.run_megatron.cli.parallel_utils import ParallelConfig
from miles.utils.debug_utils.run_megatron.cli.worker_executor import (
    _build_megatron_flags,
    build_dumper_env,
    build_torchrun_cmd,
    build_worker_args,
)
from miles.utils.debug_utils.run_megatron.worker.script_args import WorkerScriptArgs


class TestBuildDumperEnv:
    def test_basic_env_vars(self, tmp_path: Path) -> None:
        env = build_dumper_env(
            output_dir=tmp_path, run_backward=False, dumper_filter=""
        )
        assert env["DUMPER_ENABLE"] == "1"
        assert env["DUMPER_DIR"] == str(tmp_path)
        assert env["DUMPER_EXP_NAME"] == "standalone"

    def test_filter_when_non_empty(self, tmp_path: Path) -> None:
        env = build_dumper_env(
            output_dir=tmp_path, run_backward=False, dumper_filter="hidden_states"
        )
        assert env["DUMPER_FILTER"] == "hidden_states"

    def test_filter_omitted_when_empty(self, tmp_path: Path) -> None:
        env = build_dumper_env(
            output_dir=tmp_path, run_backward=False, dumper_filter=""
        )
        assert "DUMPER_FILTER" not in env

    def test_backward_enables_grad(self, tmp_path: Path) -> None:
        env = build_dumper_env(
            output_dir=tmp_path, run_backward=True, dumper_filter=""
        )
        assert env["DUMPER_ENABLE_MODEL_GRAD"] == "1"

    def test_no_grad_forward_only(self, tmp_path: Path) -> None:
        env = build_dumper_env(
            output_dir=tmp_path, run_backward=False, dumper_filter=""
        )
        assert "DUMPER_ENABLE_MODEL_GRAD" not in env


class TestBuildMegatronFlags:
    def test_parallel_sizes(self) -> None:
        flags = _build_megatron_flags(
            parallel=ParallelConfig(tp=2, pp=4, cp=2, ep=8, etp=2),
            sp=False,
            seq_length=128,
            batch_size=1,
            use_routing_replay=False,
        )
        assert "--tensor-model-parallel-size 2" in flags
        assert "--pipeline-model-parallel-size 4" in flags
        assert "--context-parallel-size 2" in flags
        assert "--expert-model-parallel-size 8" in flags
        assert "--expert-tensor-parallel-size 2" in flags

    def test_seq_length_batch_size(self) -> None:
        flags = _build_megatron_flags(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=256,
            batch_size=4,
            use_routing_replay=False,
        )
        assert "--seq-length 256" in flags
        assert "--micro-batch-size 4" in flags
        assert "--global-batch-size 4" in flags

    def test_bf16_always(self) -> None:
        flags = _build_megatron_flags(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=128,
            batch_size=1,
            use_routing_replay=False,
        )
        assert "--bf16" in flags

    def test_routing_replay_on(self) -> None:
        flags = _build_megatron_flags(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=128,
            batch_size=1,
            use_routing_replay=True,
        )
        assert "--use-routing-replay" in flags

    def test_routing_replay_off(self) -> None:
        flags = _build_megatron_flags(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=128,
            batch_size=1,
            use_routing_replay=False,
        )
        assert "--use-routing-replay" not in flags

    def test_sp_on_off(self) -> None:
        flags_on = _build_megatron_flags(
            parallel=ParallelConfig(),
            sp=True,
            seq_length=128,
            batch_size=1,
            use_routing_replay=False,
        )
        assert "--sequence-parallel" in flags_on

        flags_off = _build_megatron_flags(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=128,
            batch_size=1,
            use_routing_replay=False,
        )
        assert "--sequence-parallel" not in flags_off


class TestBuildWorkerArgs:
    def _make_script_args(self, **overrides: object) -> WorkerScriptArgs:
        defaults = dict(
            hf_checkpoint=Path("/fake/hf"),
            token_ids_file=Path("/tmp/tokens.json"),
        )
        defaults.update(overrides)
        return WorkerScriptArgs(**defaults)  # type: ignore[arg-type]

    def test_includes_parallel_flags(self) -> None:
        result = build_worker_args(
            parallel=ParallelConfig(tp=2),
            sp=False,
            seq_length=128,
            batch_size=1,
            script_args=self._make_script_args(),
            extra_args="",
        )
        assert "--tensor-model-parallel-size 2" in result

    def test_includes_script_args(self) -> None:
        result = build_worker_args(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=128,
            batch_size=1,
            script_args=self._make_script_args(hf_checkpoint=Path("/my/hf")),
            extra_args="",
        )
        assert "--script-hf-checkpoint /my/hf" in result

    def test_extra_args_appended(self) -> None:
        result = build_worker_args(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=128,
            batch_size=1,
            script_args=self._make_script_args(),
            extra_args="--custom-flag value",
        )
        assert result.endswith("--custom-flag value")

    def test_empty_extra_args(self) -> None:
        result = build_worker_args(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=128,
            batch_size=1,
            script_args=self._make_script_args(),
            extra_args="",
        )
        assert not result.endswith(" ")

    def test_routing_replay_flag(self) -> None:
        result = build_worker_args(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=128,
            batch_size=1,
            script_args=self._make_script_args(
                routing_replay_dump_path=Path("/dump")
            ),
            extra_args="",
        )
        assert "--use-routing-replay" in result

    def test_no_routing_replay(self) -> None:
        result = build_worker_args(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=128,
            batch_size=1,
            script_args=self._make_script_args(),
            extra_args="",
        )
        assert "--use-routing-replay" not in result


class TestBuildTorchrunCmd:
    @patch(
        "miles.utils.debug_utils.run_megatron.cli.worker_executor.resolve_model_script"
    )
    def test_basic_structure(self, mock_resolve: object) -> None:
        mock_resolve.return_value = Path("/repo/scripts/models/deepseek_v3.sh")  # type: ignore[union-attr]
        cmd = build_torchrun_cmd(
            model_type="deepseek_v3",
            megatron_path=Path("/megatron"),
            nproc=4,
            worker_args="--foo bar",
        )
        assert "torchrun" in cmd
        assert "source" in cmd
        assert "PYTHONPATH" in cmd

    @patch(
        "miles.utils.debug_utils.run_megatron.cli.worker_executor.resolve_model_script"
    )
    def test_nproc(self, mock_resolve: object) -> None:
        mock_resolve.return_value = Path("/repo/scripts/models/test.sh")  # type: ignore[union-attr]
        cmd = build_torchrun_cmd(
            model_type="test",
            megatron_path=Path("/megatron"),
            nproc=8,
            worker_args="",
        )
        assert "--nproc-per-node 8" in cmd

    @patch(
        "miles.utils.debug_utils.run_megatron.cli.worker_executor.resolve_model_script"
    )
    def test_worker_args_in_cmd(self, mock_resolve: object) -> None:
        mock_resolve.return_value = Path("/repo/scripts/models/test.sh")  # type: ignore[union-attr]
        cmd = build_torchrun_cmd(
            model_type="test",
            megatron_path=Path("/megatron"),
            nproc=1,
            worker_args="--my-flag 42",
        )
        assert "--my-flag 42" in cmd

    @patch(
        "miles.utils.debug_utils.run_megatron.cli.worker_executor.resolve_model_script"
    )
    def test_megatron_in_pythonpath(self, mock_resolve: object) -> None:
        mock_resolve.return_value = Path("/repo/scripts/models/test.sh")  # type: ignore[union-attr]
        cmd = build_torchrun_cmd(
            model_type="test",
            megatron_path=Path("/my/megatron"),
            nproc=1,
            worker_args="",
        )
        assert "PYTHONPATH=/my/megatron" in cmd
