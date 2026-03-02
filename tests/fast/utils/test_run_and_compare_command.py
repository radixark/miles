from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.debug_utils.run_megatron.cli.commands.args import (
    CommonRunArgs,
    CompareArgs,
    RunAndCompareArgs,
    RunArgs,
)
from miles.utils.debug_utils.run_megatron.cli.commands.run_and_compare import (
    _run_baseline_and_target,
)
from miles.utils.debug_utils.run_megatron.cli.parallel_utils import ParallelConfig


def _make_common_fields(**overrides: object) -> dict[str, object]:
    defaults = dict(
        model_type="deepseek_v3",
        hf_checkpoint=Path("/fake/hf"),
        ref_load=None,
        sp=False,
        run_backward=False,
        prompt_mode="math",
        prompt_text=None,
        prompt_file=None,
        seq_length=137,
        batch_size=1,
        apply_chat_template=False,
        role="actor",
        source_patcher_config=None,
        top_k=0,
        dumper_filter="",
        megatron_path=None,
        extra_args="",
    )
    defaults.update(overrides)
    return defaults


class TestRunAndCompare:
    @patch(
        "miles.utils.debug_utils.run_megatron.cli.commands.run_and_compare.run_impl"
    )
    def test_calls_run_twice(self, mock_run: MagicMock) -> None:
        _run_baseline_and_target(
            baseline_config=ParallelConfig(tp=1),
            target_config=ParallelConfig(tp=2),
            baseline_output=Path("/tmp/baseline"),
            target_output=Path("/tmp/target"),
            replay_dir=None,
            common_fields=_make_common_fields(),
        )
        assert mock_run.call_count == 2

    @patch(
        "miles.utils.debug_utils.run_megatron.cli.commands.run_and_compare.run_impl"
    )
    def test_output_dir_uses_dir_name(self, mock_run: MagicMock) -> None:
        baseline_config = ParallelConfig(tp=1)
        target_config = ParallelConfig(tp=2)

        _run_baseline_and_target(
            baseline_config=baseline_config,
            target_config=target_config,
            baseline_output=Path("/tmp/out") / baseline_config.dir_name(),
            target_output=Path("/tmp/out") / target_config.dir_name(),
            replay_dir=None,
            common_fields=_make_common_fields(),
        )

        baseline_args = mock_run.call_args_list[0][0][0]
        target_args = mock_run.call_args_list[1][0][0]

        assert baseline_config.dir_name() in str(baseline_args.output_dir)
        assert target_config.dir_name() in str(target_args.output_dir)

    @patch(
        "miles.utils.debug_utils.run_megatron.cli.commands.run_and_compare.run_impl"
    )
    def test_replay_baseline_nproc1_required(self, mock_run: MagicMock) -> None:
        with pytest.raises(ValueError, match="single-rank baseline"):
            _run_baseline_and_target(
                baseline_config=ParallelConfig(tp=2),
                target_config=ParallelConfig(tp=2),
                baseline_output=Path("/tmp/baseline"),
                target_output=Path("/tmp/target"),
                replay_dir=Path("/tmp/replay"),
                common_fields=_make_common_fields(),
            )

    @patch(
        "miles.utils.debug_utils.run_megatron.cli.commands.run_and_compare.run_impl"
    )
    def test_replay_paths_passed_correctly(self, mock_run: MagicMock) -> None:
        _run_baseline_and_target(
            baseline_config=ParallelConfig(tp=1),
            target_config=ParallelConfig(tp=2),
            baseline_output=Path("/tmp/baseline"),
            target_output=Path("/tmp/target"),
            replay_dir=Path("/tmp/replay"),
            common_fields=_make_common_fields(),
        )

        baseline_args = mock_run.call_args_list[0][0][0]
        target_args = mock_run.call_args_list[1][0][0]

        assert baseline_args.routing_replay_dump_path is not None
        assert baseline_args.routing_replay_load_path is None
        assert target_args.routing_replay_dump_path is None
        assert target_args.routing_replay_load_path is not None
