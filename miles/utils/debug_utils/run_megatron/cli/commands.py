"""CLI for standalone Megatron forward/backward with dump + compare.

Usage:
    python -m miles.utils.debug_utils.run_megatron run ...
    python -m miles.utils.debug_utils.run_megatron compare ...
    python -m miles.utils.debug_utils.run_megatron run-and-compare ...
    python -m miles.utils.debug_utils.run_megatron show-model-args ...
"""

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from miles.utils.debug_utils.run_megatron.cli.comparator_utils import assert_all_passed, print_json_summary
from miles.utils.debug_utils.run_megatron.cli.option_types import (
    ApplyChatTemplateOpt,
    BatchSizeOpt,
    CpOpt,
    DumperFilterOpt,
    EpOpt,
    EtpOpt,
    ExtraArgsOpt,
    HfCheckpointOpt,
    MegatronPathOpt,
    ModelTypeOpt,
    PpOpt,
    PromptFileOpt,
    PromptModeOpt,
    PromptTextOpt,
    RefLoadOpt,
    RoleOpt,
    RoutingReplayDumpOpt,
    RoutingReplayLoadOpt,
    RunBackwardOpt,
    SeqLengthOpt,
    SourcePatcherConfigOpt,
    TpOpt,
)
from miles.utils.debug_utils.run_megatron.cli.parallel_utils import (
    build_parallel_dir_name,
    nproc,
    parse_parallel_args,
)
from miles.utils.debug_utils.run_megatron.cli.path_utils import (
    resolve_megatron_path,
    resolve_model_script,
)
from miles.utils.debug_utils.run_megatron.cli.prompt_utils import generate_token_ids, write_token_ids_to_tmpfile
from miles.utils.debug_utils.run_megatron.cli.worker_executor import (
    build_dumper_env,
    build_torchrun_cmd,
    build_worker_args,
)
from miles.utils.debug_utils.run_megatron.worker.script_args import WorkerScriptArgs
from miles.utils.misc import exec_command

app: typer.Typer = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def run(
    model_type: ModelTypeOpt,
    hf_checkpoint: HfCheckpointOpt,
    output_dir: Annotated[Path, typer.Option(help="Dump output directory")] = Path("/tmp/run_megatron_dump"),
    ref_load: RefLoadOpt = None,
    tp: TpOpt = 1,
    pp: PpOpt = 1,
    cp: CpOpt = 1,
    ep: EpOpt = None,
    etp: EtpOpt = 1,
    run_backward: RunBackwardOpt = False,
    prompt_mode: PromptModeOpt = "math",
    prompt_text: PromptTextOpt = None,
    prompt_file: PromptFileOpt = None,
    seq_length: SeqLengthOpt = 137,  # odd + somewhat large; also the fine-structure constant
    batch_size: BatchSizeOpt = 1,
    apply_chat_template: ApplyChatTemplateOpt = False,
    role: RoleOpt = "actor",
    source_patcher_config: SourcePatcherConfigOpt = None,
    routing_replay_dump_path: RoutingReplayDumpOpt = None,
    routing_replay_load_path: RoutingReplayLoadOpt = None,
    dumper_filter: DumperFilterOpt = "",
    megatron_path: MegatronPathOpt = None,
    extra_args: ExtraArgsOpt = "",
) -> None:
    """Launch torchrun to run Megatron standalone forward (or forward+backward)."""
    resolved_megatron: str = resolve_megatron_path(megatron_path)
    nproc_count: int = nproc(tp=tp, pp=pp, cp=cp)

    token_ids: list[int] = generate_token_ids(
        mode=prompt_mode,  # type: ignore[arg-type]
        seq_length=seq_length,
        tokenizer_path=hf_checkpoint,
        prompt_text=prompt_text,
        prompt_file=prompt_file,
        apply_chat_template=apply_chat_template,
    )
    token_ids_file: Path = write_token_ids_to_tmpfile(token_ids)
    print(f"[cli] Token IDs written to {token_ids_file} ({len(token_ids)} tokens)", flush=True)

    script_args: WorkerScriptArgs = WorkerScriptArgs(
        hf_checkpoint=str(hf_checkpoint),
        token_ids_file=str(token_ids_file),
        role=role,
        ref_load=str(ref_load) if ref_load is not None else None,
        run_backward=run_backward,
        source_patcher_config=str(source_patcher_config) if source_patcher_config is not None else None,
        routing_replay_dump_path=str(routing_replay_dump_path) if routing_replay_dump_path is not None else None,
        routing_replay_load_path=str(routing_replay_load_path) if routing_replay_load_path is not None else None,
    )
    worker_args_str: str = build_worker_args(
        tp=tp,
        pp=pp,
        cp=cp,
        ep=ep,
        etp=etp,
        seq_length=seq_length,
        batch_size=batch_size,
        script_args=script_args,
        extra_args=extra_args,
    )

    dumper_env: dict[str, str] = build_dumper_env(
        output_dir=output_dir,
        run_backward=run_backward,
        dumper_filter=dumper_filter,
    )
    env_prefix: str = " ".join(f"{k}={v}" for k, v in dumper_env.items())

    cmd: str = build_torchrun_cmd(
        model_type=model_type,
        megatron_path=resolved_megatron,
        nproc=nproc_count,
        worker_args=worker_args_str,
    )
    exec_command(f"{env_prefix} {cmd}")
    print(f"[cli] Run completed. Output: {output_dir}", flush=True)


@app.command()
def compare(
    baseline_dir: Annotated[Path, typer.Option(help="Baseline dump directory")],
    target_dir: Annotated[Path, typer.Option(help="Target dump directory")],
    output_format: Annotated[str, typer.Option(help="Output format: text / json")] = "text",
    grouping: Annotated[str, typer.Option(help="Grouping: logical / raw")] = "logical",
    override_baseline_dims: Annotated[str | None, typer.Option(help="Override baseline dims")] = None,
    override_target_dims: Annotated[str | None, typer.Option(help="Override target dims")] = None,
    patch_config: Annotated[Path | None, typer.Option(help="Patch config YAML path")] = None,
    diff_threshold: Annotated[float | None, typer.Option(help="Pass/fail threshold")] = None,
    strict: Annotated[bool, typer.Option(help="Assert all passed (exit 1 on failure)")] = True,
) -> None:
    """Run comparator on existing dump directories."""
    cmd_parts: list[str] = [
        sys.executable,
        "-m",
        "sglang.srt.debug_utils.comparator",
        "--baseline-path",
        str(baseline_dir),
        "--target-path",
        str(target_dir),
        "--output-format",
        output_format,
        "--grouping",
        grouping,
    ]
    if override_baseline_dims:
        cmd_parts.extend(["--override-baseline-dims", override_baseline_dims])
    if override_target_dims:
        cmd_parts.extend(["--override-target-dims", override_target_dims])
    if patch_config:
        cmd_parts.extend(["--patch-config", str(patch_config)])
    if diff_threshold is not None:
        cmd_parts.extend(["--diff-threshold", str(diff_threshold)])

    print(f"EXEC: {' '.join(cmd_parts)}", flush=True)
    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd_parts,
        capture_output=True,
        text=True,
    )

    if result.stdout.strip():
        print(f"[comparator stdout]\n{result.stdout}")
    if result.stderr.strip():
        print(f"[comparator stderr]\n{result.stderr}")
    if output_format == "json":
        print_json_summary(result.stdout)

    if strict:
        if result.returncode != 0:
            raise typer.Exit(code=1)
        if output_format == "json":
            assert_all_passed(result.stdout)

    print("[cli] Compare completed.", flush=True)


@app.command(name="run-and-compare")
def run_and_compare(
    model_type: ModelTypeOpt,
    hf_checkpoint: HfCheckpointOpt,
    output_base_dir: Annotated[Path, typer.Option(help="Base output directory for dumps")],
    baseline: Annotated[str, typer.Option(help='Baseline parallel config, e.g. "--tp 1 --cp 1"')],
    target: Annotated[str, typer.Option(help='Target parallel config, e.g. "--tp 2 --cp 2"')],
    ref_load: RefLoadOpt = None,
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


@app.command(name="show-model-args")
def show_model_args(
    model_type: ModelTypeOpt,
) -> None:
    """Show the MODEL_ARGS for a given model type (debug helper)."""
    output: str | None = exec_command(
        f'source "{resolve_model_script(model_type)}" && echo "${{MODEL_ARGS[@]}}"',
        capture_output=True,
    )
    if output:
        print(output.strip())


# ---------------------------------------------------------------------------
# run-and-compare sub-flows
# ---------------------------------------------------------------------------


def _parallel_kwargs(parsed: dict[str, int]) -> dict[str, object]:
    """Convert parsed parallel config dict to kwargs for ``run()``."""
    return dict(
        tp=parsed.get("tp", 1),
        pp=parsed.get("pp", 1),
        cp=parsed.get("cp", 1),
        ep=parsed.get("ep"),
        etp=parsed.get("etp", 1),
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
