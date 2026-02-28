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

from miles.utils.debug_utils.run_megatron.comparator_utils import assert_all_passed, model_is_moe, print_json_summary
from miles.utils.debug_utils.run_megatron.helpers import (
    build_parallel_dir_name,
    exec_command,
    nproc,
    parse_parallel_args,
    resolve_megatron_path,
    resolve_model_script,
    write_prompt_to_tmpfile,
)
from miles.utils.debug_utils.run_megatron.prompt_utils import generate_prompt
from miles.utils.debug_utils.run_megatron.torchrun import build_dumper_env, build_torchrun_cmd, build_worker_args

# ---------------------------------------------------------------------------
# Shared Annotated type aliases (metadata defined once, reused across commands)
# ---------------------------------------------------------------------------

ModelTypeOpt = Annotated[str, typer.Option(help="Model type matching scripts/models/{model_type}.sh")]
HfCheckpointOpt = Annotated[Path, typer.Option(help="HuggingFace checkpoint path")]
RefLoadOpt = Annotated[Path | None, typer.Option(help="Megatron checkpoint path")]
TpOpt = Annotated[int, typer.Option(help="Tensor parallel size")]
PpOpt = Annotated[int, typer.Option(help="Pipeline parallel size")]
CpOpt = Annotated[int, typer.Option(help="Context parallel size")]
EpOpt = Annotated[int | None, typer.Option(help="Expert parallel size (default=tp)")]
EtpOpt = Annotated[int, typer.Option(help="Expert tensor parallel size")]
RunBackwardOpt = Annotated[bool, typer.Option("--run-backward", help="Run backward pass")]
PromptModeOpt = Annotated[str, typer.Option(help="Prompt mode: math / story / text")]
PromptTextOpt = Annotated[str | None, typer.Option(help="Prompt text (for text mode)")]
PromptFileOpt = Annotated[Path | None, typer.Option(help="Prompt file (for story mode)")]
SeqLengthOpt = Annotated[int, typer.Option(help="Sequence length")]
BatchSizeOpt = Annotated[int, typer.Option(help="Micro batch size")]
ApplyChatTemplateOpt = Annotated[bool, typer.Option("--apply-chat-template", help="Apply chat template")]
RoleOpt = Annotated[str, typer.Option(help="Model role: actor / critic")]
SourcePatcherConfigOpt = Annotated[Path | None, typer.Option(help="Source patcher YAML config path")]
RoutingReplayDumpOpt = Annotated[Path | None, typer.Option(help="Routing replay dump path")]
RoutingReplayLoadOpt = Annotated[Path | None, typer.Option(help="Routing replay load path")]
IndexerReplayDumpOpt = Annotated[Path | None, typer.Option(help="Indexer replay dump path")]
IndexerReplayLoadOpt = Annotated[Path | None, typer.Option(help="Indexer replay load path")]
DumperFilterOpt = Annotated[str, typer.Option(help="Dumper filter expression")]
MegatronPathOpt = Annotated[Path | None, typer.Option(help="Path to Megatron-LM")]
ExtraArgsOpt = Annotated[str, typer.Option(help="Extra args passed to worker")]

app: typer.Typer = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def run(
    model_type: ModelTypeOpt,
    hf_checkpoint: HfCheckpointOpt,
    output_dir: Annotated[Path, typer.Option(help="Dump output directory")],
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
    seq_length: SeqLengthOpt = 128,
    batch_size: BatchSizeOpt = 1,
    apply_chat_template: ApplyChatTemplateOpt = False,
    role: RoleOpt = "actor",
    source_patcher_config: SourcePatcherConfigOpt = None,
    routing_replay_dump_path: RoutingReplayDumpOpt = None,
    routing_replay_load_path: RoutingReplayLoadOpt = None,
    indexer_replay_dump_path: IndexerReplayDumpOpt = None,
    indexer_replay_load_path: IndexerReplayLoadOpt = None,
    dumper_filter: DumperFilterOpt = "",
    megatron_path: MegatronPathOpt = None,
    extra_args: ExtraArgsOpt = "",
) -> None:
    """Launch torchrun to run Megatron standalone forward (or forward+backward)."""
    resolved_megatron: str = resolve_megatron_path(megatron_path)
    nproc_count: int = nproc(tp=tp, pp=pp, cp=cp)

    prompt: str = generate_prompt(
        mode=prompt_mode,  # type: ignore[arg-type]
        seq_length=seq_length,
        tokenizer_path=hf_checkpoint if prompt_mode != "text" else None,
        prompt_text=prompt_text,
        prompt_file=prompt_file,
        apply_chat_template=apply_chat_template,
    )
    prompt_tmpfile: Path = write_prompt_to_tmpfile(prompt)
    print(f"[cli] Prompt written to {prompt_tmpfile} ({len(prompt)} chars)", flush=True)

    worker_args_str: str = build_worker_args(
        tp=tp,
        pp=pp,
        cp=cp,
        ep=ep,
        etp=etp,
        hf_checkpoint=hf_checkpoint,
        ref_load=ref_load,
        seq_length=seq_length,
        batch_size=batch_size,
        run_backward=run_backward,
        role=role,
        prompt_file=prompt_tmpfile,
        source_patcher_config=source_patcher_config,
        routing_replay_dump_path=routing_replay_dump_path,
        routing_replay_load_path=routing_replay_load_path,
        indexer_replay_dump_path=indexer_replay_dump_path,
        indexer_replay_load_path=indexer_replay_load_path,
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
    seq_length: SeqLengthOpt = 128,
    batch_size: BatchSizeOpt = 1,
    apply_chat_template: ApplyChatTemplateOpt = False,
    role: RoleOpt = "actor",
    source_patcher_config: SourcePatcherConfigOpt = None,
    dumper_filter: DumperFilterOpt = "",
    megatron_path: MegatronPathOpt = None,
    extra_args: ExtraArgsOpt = "",
    no_replay: Annotated[bool, typer.Option("--no-replay", help="Disable automatic routing replay")] = False,
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
        indexer_replay_dump_path=None,
        indexer_replay_load_path=None,
    )

    needs_replay: bool = not no_replay and model_is_moe(model_type)
    replay_dir: Path = output_base_dir / "routing_replay"

    if needs_replay:
        _run_and_compare_with_replay(
            baseline_p=baseline_p,
            target_p=target_p,
            baseline_output=baseline_output,
            target_output=target_output,
            replay_dir=replay_dir,
            common_run_kwargs=common_run_kwargs,
        )
    else:
        _run_and_compare_simple(
            baseline_p=baseline_p,
            target_p=target_p,
            baseline_output=baseline_output,
            target_output=target_output,
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


def _run_and_compare_with_replay(
    *,
    baseline_p: dict[str, int],
    target_p: dict[str, int],
    baseline_output: Path,
    target_output: Path,
    replay_dir: Path,
    common_run_kwargs: dict[str, object],
) -> None:
    print("[cli] MOE model detected — using routing replay flow", flush=True)

    print("[cli] Step 1/3: Baseline run (record routing)", flush=True)
    run(
        **common_run_kwargs,
        **_parallel_kwargs(baseline_p),  # type: ignore[arg-type]
        output_dir=baseline_output,
        routing_replay_dump_path=replay_dir,
        routing_replay_load_path=None,
    )

    print("[cli] Step 2/3: Baseline run (replay routing)", flush=True)
    exec_command(f"rm -rf {baseline_output}")
    run(
        **common_run_kwargs,
        **_parallel_kwargs(baseline_p),  # type: ignore[arg-type]
        output_dir=baseline_output,
        routing_replay_dump_path=None,
        routing_replay_load_path=replay_dir,
    )

    print("[cli] Step 3/3: Target run (replay routing)", flush=True)
    run(
        **common_run_kwargs,
        **_parallel_kwargs(target_p),  # type: ignore[arg-type]
        output_dir=target_output,
        routing_replay_dump_path=None,
        routing_replay_load_path=replay_dir,
    )


def _run_and_compare_simple(
    *,
    baseline_p: dict[str, int],
    target_p: dict[str, int],
    baseline_output: Path,
    target_output: Path,
    common_run_kwargs: dict[str, object],
) -> None:
    print("[cli] Step 1/2: Baseline run", flush=True)
    run(
        **common_run_kwargs,
        **_parallel_kwargs(baseline_p),  # type: ignore[arg-type]
        output_dir=baseline_output,
        routing_replay_dump_path=None,
        routing_replay_load_path=None,
    )

    print("[cli] Step 2/2: Target run", flush=True)
    run(
        **common_run_kwargs,
        **_parallel_kwargs(target_p),  # type: ignore[arg-type]
        output_dir=target_output,
        routing_replay_dump_path=None,
        routing_replay_load_path=None,
    )
