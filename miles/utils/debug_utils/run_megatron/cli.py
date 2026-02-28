"""CLI for standalone Megatron forward/backward with dump + compare.

Usage:
    python -m miles.utils.debug_utils.run_megatron run ...
    python -m miles.utils.debug_utils.run_megatron compare ...
    python -m miles.utils.debug_utils.run_megatron run-and-compare ...
    python -m miles.utils.debug_utils.run_megatron show-model-args ...
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Optional

import typer

from miles.utils.debug_utils.run_megatron.prompt_utils import generate_prompt

app: typer.Typer = typer.Typer(pretty_exceptions_enable=False)

_DEFAULT_MEGATRON_PATH: str = "/root/Megatron-LM"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exec_command(cmd: str, *, capture_output: bool = False) -> Optional[str]:
    print(f"EXEC: {cmd}", flush=True)
    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            ["bash", "-c", cmd],
            check=True,
            capture_output=capture_output,
            **(dict(text=True) if capture_output else {}),
        )
    except subprocess.CalledProcessError as e:
        if capture_output:
            print(f"FAILED: stdout={e.stdout} stderr={e.stderr}")
        raise
    if capture_output:
        return result.stdout + result.stderr
    return None


def _resolve_megatron_path(megatron_path: Optional[Path]) -> str:
    if megatron_path is not None:
        return str(megatron_path)
    env_path: str | None = os.environ.get("MEGATRON_PATH")
    if env_path:
        return env_path
    return _DEFAULT_MEGATRON_PATH


def _resolve_repo_base() -> Path:
    return Path(os.path.abspath(__file__)).resolve().parents[4]


def _resolve_model_script(model_type: str) -> Path:
    repo_base: Path = _resolve_repo_base()
    script: Path = repo_base / "scripts" / "models" / f"{model_type}.sh"
    if not script.exists():
        raise typer.BadParameter(f"Model script not found: {script}")
    return script


def _build_parallel_dir_name(
    *,
    tp: int,
    pp: int,
    cp: int,
    ep: Optional[int],
    etp: int,
) -> str:
    """Build directory name from parallel config, e.g. 'tp2_cp2_ep2'."""
    parts: list[str] = [f"tp{tp}"]
    if pp > 1:
        parts.append(f"pp{pp}")
    if cp > 1:
        parts.append(f"cp{cp}")
    if ep is not None and ep != tp:
        parts.append(f"ep{ep}")
    if etp > 1:
        parts.append(f"etp{etp}")
    return "_".join(parts)


def _parse_parallel_args(args_str: str) -> dict[str, int]:
    """Parse a parallel config string like '--tp 2 --cp 2' into a dict."""
    tokens: list[str] = args_str.split()
    result: dict[str, int] = {}
    arg_map: dict[str, str] = {
        "--tp": "tp",
        "--pp": "pp",
        "--cp": "cp",
        "--ep": "ep",
        "--etp": "etp",
    }
    i: int = 0
    while i < len(tokens):
        if tokens[i] in arg_map and i + 1 < len(tokens):
            result[arg_map[tokens[i]]] = int(tokens[i + 1])
            i += 2
        else:
            i += 1
    return result


def _nproc(*, tp: int, pp: int, cp: int) -> int:
    return tp * pp * cp


def _write_prompt_to_tmpfile(prompt_text: str) -> Path:
    tmp: tempfile.NamedTemporaryFile = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="run_megatron_prompt_"
    )
    tmp.write(prompt_text)
    tmp.close()
    return Path(tmp.name)


def _build_torchrun_cmd(
    *,
    model_type: str,
    megatron_path: str,
    nproc: int,
    worker_args: str,
) -> str:
    """Build the full shell command to launch the worker via torchrun."""
    repo_base: Path = _resolve_repo_base()
    model_script: Path = _resolve_model_script(model_type)
    worker_module: str = "miles.utils.debug_utils.run_megatron.worker"

    cmd: str = (
        f'source "{model_script}" && '
        f"PYTHONPATH={megatron_path}:$PYTHONPATH "
        f"CUDA_DEVICE_MAX_CONNECTIONS=1 "
        f"torchrun --nproc-per-node {nproc} "
        f"-m {worker_module} "
        f'${{MODEL_ARGS[@]}} '
        f"--tokenizer-type HuggingFaceTokenizer "
        f"--hidden-dropout 0 --attention-dropout 0 "
        f"{worker_args}"
    )
    return cmd


def _build_worker_args(
    *,
    tp: int,
    pp: int,
    cp: int,
    ep: Optional[int],
    etp: int,
    hf_checkpoint: Path,
    ref_load: Optional[Path],
    output_dir: Path,
    seq_length: int,
    batch_size: int,
    run_backward: bool,
    role: str,
    prompt_file: Path,
    source_patcher_config: Optional[Path],
    routing_replay_dump_path: Optional[Path],
    routing_replay_load_path: Optional[Path],
    indexer_replay_dump_path: Optional[Path],
    indexer_replay_load_path: Optional[Path],
    extra_args: str,
) -> str:
    """Build the worker argument string."""
    effective_ep: int = ep if ep is not None else tp

    parts: list[str] = [
        f"--tensor-model-parallel-size {tp}",
        "--sequence-parallel" if tp > 1 else "",
        f"--pipeline-model-parallel-size {pp}",
        f"--context-parallel-size {cp}",
        f"--expert-model-parallel-size {effective_ep}",
        f"--expert-tensor-parallel-size {etp}",
        f"--hf-checkpoint {hf_checkpoint}",
        f"--seq-length {seq_length}",
        f"--micro-batch-size {batch_size}",
        f"--global-batch-size {batch_size}",
        f"--prompt-file {prompt_file}",
        f"--role {role}",
        "--bf16",
        "--no-gradient-accumulation-fusion",
        "--use-miles-router",
    ]

    if ref_load is not None:
        parts.append(f"--ref-load {ref_load}")

    if run_backward:
        parts.append("--run-backward")

    if source_patcher_config is not None:
        parts.append(f"--source-patcher-config {source_patcher_config}")

    if routing_replay_dump_path is not None:
        parts.append(f"--routing-replay-dump-path {routing_replay_dump_path}")
        parts.append("--use-routing-replay")

    if routing_replay_load_path is not None:
        parts.append(f"--routing-replay-load-path {routing_replay_load_path}")
        parts.append("--use-routing-replay")

    if indexer_replay_dump_path is not None:
        parts.append(f"--indexer-replay-dump-path {indexer_replay_dump_path}")

    if indexer_replay_load_path is not None:
        parts.append(f"--indexer-replay-load-path {indexer_replay_load_path}")

    if extra_args:
        parts.append(extra_args)

    return " ".join(p for p in parts if p)


def _build_dumper_env(
    *,
    output_dir: Path,
    run_backward: bool,
    dumper_filter: str,
) -> dict[str, str]:
    """Build DUMPER_* environment variables for the worker."""
    env: dict[str, str] = {
        "DUMPER_ENABLE": "1",
        "DUMPER_DIR": str(output_dir),
        "DUMPER_EXP_NAME": "standalone",
    }
    if dumper_filter:
        env["DUMPER_FILTER"] = dumper_filter
    if run_backward:
        env["DUMPER_DUMP_GRAD"] = "1"
    return env


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def run(
    model_type: Annotated[str, typer.Option(help="Model type matching scripts/models/{model_type}.sh")],
    hf_checkpoint: Annotated[Path, typer.Option(help="HuggingFace checkpoint path")],
    output_dir: Annotated[Path, typer.Option(help="Dump output directory")],
    ref_load: Annotated[Optional[Path], typer.Option(help="Megatron checkpoint path")] = None,
    tp: Annotated[int, typer.Option(help="Tensor parallel size")] = 1,
    pp: Annotated[int, typer.Option(help="Pipeline parallel size")] = 1,
    cp: Annotated[int, typer.Option(help="Context parallel size")] = 1,
    ep: Annotated[Optional[int], typer.Option(help="Expert parallel size (default=tp)")] = None,
    etp: Annotated[int, typer.Option(help="Expert tensor parallel size")] = 1,
    run_backward: Annotated[bool, typer.Option("--run-backward", help="Run backward pass")] = False,
    prompt_mode: Annotated[str, typer.Option(help="Prompt mode: math / story / text")] = "math",
    prompt_text: Annotated[Optional[str], typer.Option(help="Prompt text (for text mode)")] = None,
    prompt_file: Annotated[Optional[Path], typer.Option(help="Prompt file (for story mode)")] = None,
    seq_length: Annotated[int, typer.Option(help="Sequence length")] = 128,
    batch_size: Annotated[int, typer.Option(help="Micro batch size")] = 1,
    apply_chat_template: Annotated[bool, typer.Option("--apply-chat-template", help="Apply chat template")] = False,
    role: Annotated[str, typer.Option(help="Model role: actor / critic")] = "actor",
    source_patcher_config: Annotated[Optional[Path], typer.Option(help="Source patcher YAML config path")] = None,
    routing_replay_dump_path: Annotated[Optional[Path], typer.Option(help="Routing replay dump path")] = None,
    routing_replay_load_path: Annotated[Optional[Path], typer.Option(help="Routing replay load path")] = None,
    indexer_replay_dump_path: Annotated[Optional[Path], typer.Option(help="Indexer replay dump path")] = None,
    indexer_replay_load_path: Annotated[Optional[Path], typer.Option(help="Indexer replay load path")] = None,
    dumper_filter: Annotated[str, typer.Option(help="Dumper filter expression")] = "",
    megatron_path: Annotated[Optional[Path], typer.Option(help="Path to Megatron-LM")] = None,
    extra_args: Annotated[str, typer.Option(help="Extra args passed to worker")] = "",
) -> None:
    """Launch torchrun to run Megatron standalone forward (or forward+backward)."""
    resolved_megatron: str = _resolve_megatron_path(megatron_path)
    nproc_count: int = _nproc(tp=tp, pp=pp, cp=cp)

    prompt: str = generate_prompt(
        mode=prompt_mode,  # type: ignore[arg-type]
        seq_length=seq_length,
        tokenizer_path=hf_checkpoint if prompt_mode != "text" else None,
        prompt_text=prompt_text,
        prompt_file=prompt_file,
        apply_chat_template=apply_chat_template,
    )
    prompt_tmpfile: Path = _write_prompt_to_tmpfile(prompt)
    print(f"[cli] Prompt written to {prompt_tmpfile} ({len(prompt)} chars)", flush=True)

    worker_args: str = _build_worker_args(
        tp=tp, pp=pp, cp=cp, ep=ep, etp=etp,
        hf_checkpoint=hf_checkpoint,
        ref_load=ref_load,
        output_dir=output_dir,
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

    dumper_env: dict[str, str] = _build_dumper_env(
        output_dir=output_dir,
        run_backward=run_backward,
        dumper_filter=dumper_filter,
    )
    env_prefix: str = " ".join(f"{k}={v}" for k, v in dumper_env.items())

    cmd: str = _build_torchrun_cmd(
        model_type=model_type,
        megatron_path=resolved_megatron,
        nproc=nproc_count,
        worker_args=worker_args,
    )
    full_cmd: str = f"{env_prefix} {cmd}"

    _exec_command(full_cmd)
    print(f"[cli] Run completed. Output: {output_dir}", flush=True)


@app.command()
def compare(
    baseline_dir: Annotated[Path, typer.Option(help="Baseline dump directory")],
    target_dir: Annotated[Path, typer.Option(help="Target dump directory")],
    output_format: Annotated[str, typer.Option(help="Output format: text / json")] = "text",
    grouping: Annotated[str, typer.Option(help="Grouping: logical / raw")] = "logical",
    override_baseline_dims: Annotated[Optional[str], typer.Option(help="Override baseline dims")] = None,
    override_target_dims: Annotated[Optional[str], typer.Option(help="Override target dims")] = None,
    patch_config: Annotated[Optional[Path], typer.Option(help="Patch config YAML path")] = None,
    diff_threshold: Annotated[Optional[float], typer.Option(help="Pass/fail threshold")] = None,
    strict: Annotated[bool, typer.Option(help="Assert all passed (exit 1 on failure)")] = True,
) -> None:
    """Run comparator on existing dump directories."""
    cmd_parts: list[str] = [
        sys.executable, "-m", "sglang.srt.debug_utils.comparator",
        "--baseline-path", str(baseline_dir),
        "--target-path", str(target_dir),
        "--output-format", output_format,
        "--grouping", grouping,
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
        _print_json_summary(result.stdout)

    if strict:
        if result.returncode != 0:
            raise typer.Exit(code=1)
        if output_format == "json":
            _assert_all_passed(result.stdout)

    print("[cli] Compare completed.", flush=True)


@app.command(name="run-and-compare")
def run_and_compare(
    model_type: Annotated[str, typer.Option(help="Model type matching scripts/models/{model_type}.sh")],
    hf_checkpoint: Annotated[Path, typer.Option(help="HuggingFace checkpoint path")],
    output_base_dir: Annotated[Path, typer.Option(help="Base output directory for dumps")],
    baseline: Annotated[str, typer.Option(help='Baseline parallel config, e.g. "--tp 1 --cp 1"')],
    target: Annotated[str, typer.Option(help='Target parallel config, e.g. "--tp 2 --cp 2"')],
    ref_load: Annotated[Optional[Path], typer.Option(help="Megatron checkpoint path")] = None,
    run_backward: Annotated[bool, typer.Option("--run-backward", help="Run backward pass")] = False,
    prompt_mode: Annotated[str, typer.Option(help="Prompt mode: math / story / text")] = "math",
    prompt_text: Annotated[Optional[str], typer.Option(help="Prompt text (for text mode)")] = None,
    prompt_file: Annotated[Optional[Path], typer.Option(help="Prompt file (for story mode)")] = None,
    seq_length: Annotated[int, typer.Option(help="Sequence length")] = 128,
    batch_size: Annotated[int, typer.Option(help="Micro batch size")] = 1,
    apply_chat_template: Annotated[bool, typer.Option("--apply-chat-template", help="Apply chat template")] = False,
    role: Annotated[str, typer.Option(help="Model role: actor / critic")] = "actor",
    source_patcher_config: Annotated[Optional[Path], typer.Option(help="Source patcher YAML config")] = None,
    dumper_filter: Annotated[str, typer.Option(help="Dumper filter expression")] = "",
    megatron_path: Annotated[Optional[Path], typer.Option(help="Path to Megatron-LM")] = None,
    extra_args: Annotated[str, typer.Option(help="Extra args passed to worker")] = "",
    no_replay: Annotated[bool, typer.Option("--no-replay", help="Disable automatic routing replay")] = False,
) -> None:
    """Run baseline + target configs, then compare dumps."""
    baseline_parallel: dict[str, int] = _parse_parallel_args(baseline)
    target_parallel: dict[str, int] = _parse_parallel_args(target)

    baseline_tp: int = baseline_parallel.get("tp", 1)
    baseline_pp: int = baseline_parallel.get("pp", 1)
    baseline_cp: int = baseline_parallel.get("cp", 1)
    baseline_ep: Optional[int] = baseline_parallel.get("ep")
    baseline_etp: int = baseline_parallel.get("etp", 1)

    target_tp: int = target_parallel.get("tp", 1)
    target_pp: int = target_parallel.get("pp", 1)
    target_cp: int = target_parallel.get("cp", 1)
    target_ep: Optional[int] = target_parallel.get("ep")
    target_etp: int = target_parallel.get("etp", 1)

    baseline_dir_name: str = _build_parallel_dir_name(
        tp=baseline_tp, pp=baseline_pp, cp=baseline_cp, ep=baseline_ep, etp=baseline_etp,
    )
    target_dir_name: str = _build_parallel_dir_name(
        tp=target_tp, pp=target_pp, cp=target_cp, ep=target_ep, etp=target_etp,
    )

    baseline_output: Path = output_base_dir / baseline_dir_name
    target_output: Path = output_base_dir / target_dir_name

    needs_replay: bool = not no_replay and _model_is_moe(model_type)
    replay_dir: Path = output_base_dir / "routing_replay"

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

    if needs_replay:
        print("[cli] MOE model detected — using routing replay flow", flush=True)

        # Step 1: baseline run with replay RECORD
        print("[cli] Step 1/3: Baseline run (record routing)", flush=True)
        run(
            **common_run_kwargs,  # type: ignore[arg-type]
            tp=baseline_tp, pp=baseline_pp, cp=baseline_cp, ep=baseline_ep, etp=baseline_etp,
            output_dir=baseline_output,
            routing_replay_dump_path=replay_dir,
            routing_replay_load_path=None,
        )

        # Step 2: baseline run with replay REPLAY (for consistent comparison)
        print("[cli] Step 2/3: Baseline run (replay routing)", flush=True)
        _exec_command(f"rm -rf {baseline_output}")
        run(
            **common_run_kwargs,  # type: ignore[arg-type]
            tp=baseline_tp, pp=baseline_pp, cp=baseline_cp, ep=baseline_ep, etp=baseline_etp,
            output_dir=baseline_output,
            routing_replay_dump_path=None,
            routing_replay_load_path=replay_dir,
        )

        # Step 3: target run with replay REPLAY
        print("[cli] Step 3/3: Target run (replay routing)", flush=True)
        run(
            **common_run_kwargs,  # type: ignore[arg-type]
            tp=target_tp, pp=target_pp, cp=target_cp, ep=target_ep, etp=target_etp,
            output_dir=target_output,
            routing_replay_dump_path=None,
            routing_replay_load_path=replay_dir,
        )
    else:
        # Simple flow: baseline then target
        print("[cli] Step 1/2: Baseline run", flush=True)
        run(
            **common_run_kwargs,  # type: ignore[arg-type]
            tp=baseline_tp, pp=baseline_pp, cp=baseline_cp, ep=baseline_ep, etp=baseline_etp,
            output_dir=baseline_output,
            routing_replay_dump_path=None,
            routing_replay_load_path=None,
        )

        print("[cli] Step 2/2: Target run", flush=True)
        run(
            **common_run_kwargs,  # type: ignore[arg-type]
            tp=target_tp, pp=target_pp, cp=target_cp, ep=target_ep, etp=target_etp,
            output_dir=target_output,
            routing_replay_dump_path=None,
            routing_replay_load_path=None,
        )

    # Compare
    print("[cli] Comparing baseline vs target", flush=True)
    compare(
        baseline_dir=baseline_output / "standalone",
        target_dir=target_output / "standalone",
        output_format="json",
        grouping="logical",
    )


@app.command(name="show-model-args")
def show_model_args(
    model_type: Annotated[str, typer.Option(help="Model type matching scripts/models/{model_type}.sh")],
) -> None:
    """Show the MODEL_ARGS for a given model type (debug helper)."""
    model_script: Path = _resolve_model_script(model_type)
    output: Optional[str] = _exec_command(
        f'source "{model_script}" && echo "${{MODEL_ARGS[@]}}"',
        capture_output=True,
    )
    if output:
        print(output.strip())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _model_is_moe(model_type: str) -> bool:
    """Check if a model script defines num-experts > 0."""
    model_script: Path = _resolve_model_script(model_type)
    output: Optional[str] = _exec_command(
        f'source "{model_script}" && echo "${{MODEL_ARGS[@]}}"',
        capture_output=True,
    )
    if output and "--num-experts" in output:
        tokens: list[str] = output.split()
        for i, token in enumerate(tokens):
            if token == "--num-experts" and i + 1 < len(tokens):
                try:
                    return int(tokens[i + 1]) > 0
                except ValueError:
                    pass
    return False


def _print_json_summary(stdout: str) -> None:
    """Print a human-readable summary from JSON comparator output."""
    lines: list[str] = [line for line in stdout.strip().splitlines() if line.strip()]
    for line in lines:
        try:
            record: dict[str, object] = json.loads(line)
            if record.get("type") == "summary":
                print(
                    f"[summary] passed={record.get('passed')}, "
                    f"failed={record.get('failed')}, "
                    f"skipped={record.get('skipped')}",
                    flush=True,
                )
        except json.JSONDecodeError:
            pass


def _assert_all_passed(stdout: str) -> None:
    """Assert all comparisons passed in JSON output (strict mode)."""
    from sglang.srt.debug_utils.comparator.output_types import (
        ComparisonRecord,
        SummaryRecord,
        parse_record_json,
    )

    lines: list[str] = [line for line in stdout.strip().splitlines() if line.strip()]
    records = [parse_record_json(line) for line in lines]

    comparisons: list[ComparisonRecord] = [r for r in records if isinstance(r, ComparisonRecord)]
    assert len(comparisons) > 0, "No comparison records produced"

    failed: list[str] = []
    for comp in comparisons:
        if comp.diff is None or not comp.diff.passed:
            rel_diff: float = comp.diff.rel_diff if comp.diff is not None else float("nan")
            failed.append(f"{comp.name} (rel_diff={rel_diff:.6f})")

    assert len(failed) == 0, (
        f"Comparator found {len(failed)} failures out of {len(comparisons)}: "
        + ", ".join(failed[:10])
    )

    summaries: list[SummaryRecord] = [r for r in records if isinstance(r, SummaryRecord)]
    assert len(summaries) == 1, f"Expected 1 summary, got {len(summaries)}"
    summary: SummaryRecord = summaries[0]
    assert summary.passed > 0, f"Summary passed must be > 0, got {summary.passed}"
    assert summary.failed == 0, f"Summary failed must be 0, got {summary.failed}"
    assert summary.skipped == 0, f"Summary skipped must be 0, got {summary.skipped}"

    print(
        f"[cli] All passed: total={len(comparisons)}, "
        f"summary: passed={summary.passed}, failed={summary.failed}, skipped={summary.skipped}",
        flush=True,
    )
