"""``run-and-compare`` CLI command."""

from pathlib import Path
from typing import Annotated

import typer

from miles.utils.debug_utils.run_megatron.cli.commands.compare import compare
from miles.utils.debug_utils.run_megatron.cli.commands.run import run
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
)
from miles.utils.debug_utils.run_megatron.cli.parallel_utils import (
    build_parallel_dir_name,
    parse_parallel_args,
)


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
    dumper_filter: DumperFilterOpt = "",
    megatron_path: MegatronPathOpt = None,
    extra_args: ExtraArgsOpt = "",
    routing_replay: Annotated[
        bool, typer.Option("--routing-replay", help="Enable routing replay (record on baseline, replay on target)")
    ] = False,
) -> None:
    """Run baseline + target configs, then compare dumps."""
    baseline_p: dict[str, int] = parse_parallel_args(baseline)
    target_p: dict[str, int] = parse_parallel_args(target)

    baseline_output: Path = output_base_dir / build_parallel_dir_name(
        tp=baseline_p.get("tp", 1),
        pp=baseline_p.get("pp", 1),
        cp=baseline_p.get("cp", 1),
        ep=baseline_p.get("ep"),
        etp=baseline_p.get("etp", 1),
    )
    target_output: Path = output_base_dir / build_parallel_dir_name(
        tp=target_p.get("tp", 1),
        pp=target_p.get("pp", 1),
        cp=target_p.get("cp", 1),
        ep=target_p.get("ep"),
        etp=target_p.get("etp", 1),
    )

    common_run_kwargs: dict[str, object] = dict(
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
        dumper_filter=dumper_filter,
        megatron_path=megatron_path,
        extra_args=extra_args,
    )

    replay_dir: Path | None = output_base_dir / "routing_replay" if routing_replay else None

    _run_baseline_and_target(
        baseline_p=baseline_p,
        target_p=target_p,
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
    baseline_p: dict[str, int],
    target_p: dict[str, int],
    baseline_output: Path,
    target_output: Path,
    replay_dir: Path | None,
    common_run_kwargs: dict[str, object],
) -> None:
    if replay_dir is not None:
        print("[cli] Routing replay enabled", flush=True)

    print("[cli] Step 1/2: Baseline run", flush=True)
    run(
        **common_run_kwargs,
        **_parallel_kwargs(baseline_p),  # type: ignore[arg-type]
        output_dir=baseline_output,
        routing_replay_dump_path=replay_dir,
        routing_replay_load_path=None,
    )

    print("[cli] Step 2/2: Target run", flush=True)
    run(
        **common_run_kwargs,
        **_parallel_kwargs(target_p),  # type: ignore[arg-type]
        output_dir=target_output,
        routing_replay_dump_path=None,
        routing_replay_load_path=replay_dir,
    )


def _parallel_kwargs(parsed: dict[str, int]) -> dict[str, object]:
    """Convert parsed parallel config dict to kwargs for ``run()``."""
    return dict(
        tp=parsed.get("tp", 1),
        pp=parsed.get("pp", 1),
        cp=parsed.get("cp", 1),
        ep=parsed.get("ep"),
        etp=parsed.get("etp", 1),
    )
