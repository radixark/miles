"""``run-and-compare`` CLI command."""

import dataclasses
from pathlib import Path
from typing import Annotated, TypedDict

import typer

from miles.utils.debug_utils.run_megatron.cli.commands.compare import compare
from miles.utils.debug_utils.run_megatron.cli.commands.option_types import (
    ApplyChatTemplateOpt,
    BatchSizeOpt,
    DumperFilterOpt,
    ExtraArgsOpt,
    HfCheckpointOpt,
    MegatronPathOpt,
    ModelTypeOpt,
    PromptFileOpt,
    PromptModeOpt,
    PromptTextOpt,
    RefLoadOpt,
    RoleOpt,
    RunBackwardOpt,
    SeqLengthOpt,
    SourcePatcherConfigOpt,
    SpOpt,
    TopKOpt,
)
from miles.utils.debug_utils.run_megatron.cli.commands.run import run
from miles.utils.debug_utils.run_megatron.cli.parallel_utils import ParallelConfig, parse_parallel_args


class _CommonRunKwargs(TypedDict):
    model_type: str
    hf_checkpoint: Path
    ref_load: Path | None
    sp: bool
    run_backward: bool
    prompt_mode: str
    prompt_text: str | None
    prompt_file: Path | None
    seq_length: int
    batch_size: int
    apply_chat_template: bool
    role: str
    source_patcher_config: Path | None
    top_k: int
    dumper_filter: str
    megatron_path: Path | None
    extra_args: str


def register(app: typer.Typer) -> None:
    """Register the ``run-and-compare`` command on *app*."""
    app.command(name="run-and-compare")(run_and_compare)


def run_and_compare(
    model_type: ModelTypeOpt,
    hf_checkpoint: HfCheckpointOpt,
    output_base_dir: Annotated[Path, typer.Option(help="Base output directory for dumps")],
    baseline: Annotated[str, typer.Option(help='Baseline parallel config, e.g. "--tp 1 --cp 1"')],
    target: Annotated[str, typer.Option(help='Target parallel config, e.g. "--tp 2 --cp 2"')],
    ref_load: RefLoadOpt = None,
    sp: SpOpt = False,
    run_backward: RunBackwardOpt = False,
    prompt_mode: PromptModeOpt = "math",
    prompt_text: PromptTextOpt = None,
    prompt_file: PromptFileOpt = None,
    seq_length: SeqLengthOpt = 137,  # odd + somewhat large; also the fine-structure constant
    batch_size: BatchSizeOpt = 1,
    apply_chat_template: ApplyChatTemplateOpt = False,
    role: RoleOpt = "actor",
    source_patcher_config: SourcePatcherConfigOpt = None,
    top_k: TopKOpt = 0,
    dumper_filter: DumperFilterOpt = "",
    megatron_path: MegatronPathOpt = None,
    extra_args: ExtraArgsOpt = "",
    routing_replay: Annotated[
        bool, typer.Option("--routing-replay", help="Enable routing replay (record on baseline, replay on target)")
    ] = False,
) -> None:
    """Run baseline + target configs, then compare dumps."""
    baseline_config: ParallelConfig = ParallelConfig.from_parsed_args(parse_parallel_args(baseline))
    target_config: ParallelConfig = ParallelConfig.from_parsed_args(parse_parallel_args(target))

    baseline_output: Path = output_base_dir / baseline_config.dir_name()
    target_output: Path = output_base_dir / target_config.dir_name()

    common_run_kwargs: _CommonRunKwargs = _CommonRunKwargs(
        model_type=model_type,
        hf_checkpoint=hf_checkpoint,
        ref_load=ref_load,
        sp=sp,
        run_backward=run_backward,
        prompt_mode=prompt_mode,
        prompt_text=prompt_text,
        prompt_file=prompt_file,
        seq_length=seq_length,
        batch_size=batch_size,
        apply_chat_template=apply_chat_template,
        role=role,
        source_patcher_config=source_patcher_config,
        top_k=top_k,
        dumper_filter=dumper_filter,
        megatron_path=megatron_path,
        extra_args=extra_args,
    )

    replay_dir: Path | None = output_base_dir / "routing_replay" if routing_replay else None

    _run_baseline_and_target(
        baseline_config=baseline_config,
        target_config=target_config,
        baseline_output=baseline_output,
        target_output=target_output,
        replay_dir=replay_dir,
        common_run_kwargs=common_run_kwargs,
    )

    print("[cli] Comparing baseline vs target", flush=True)
    compare(
        baseline_dir=baseline_output / "standalone",
        target_dir=target_output / "standalone",
        output_format="json",
        grouping="logical",
    )


def _run_baseline_and_target(
    *,
    baseline_config: ParallelConfig,
    target_config: ParallelConfig,
    baseline_output: Path,
    target_output: Path,
    replay_dir: Path | None,
    common_run_kwargs: _CommonRunKwargs,
) -> None:
    if replay_dir is not None:
        if baseline_config.nproc != 1:
            raise ValueError(
                f"Routing replay requires single-rank baseline (nproc=1), "
                f"got nproc={baseline_config.nproc} (tp={baseline_config.tp}, pp={baseline_config.pp}, cp={baseline_config.cp})"
            )
        print("[cli] Routing replay enabled", flush=True)

    print("[cli] Step 1/2: Baseline run", flush=True)
    run(
        **common_run_kwargs,
        **dataclasses.asdict(baseline_config),
        output_dir=baseline_output,
        routing_replay_dump_path=replay_dir,
        routing_replay_load_path=None,
    )

    print("[cli] Step 2/2: Target run", flush=True)
    run(
        **common_run_kwargs,
        **dataclasses.asdict(target_config),
        output_dir=target_output,
        routing_replay_dump_path=None,
        routing_replay_load_path=replay_dir,
    )
