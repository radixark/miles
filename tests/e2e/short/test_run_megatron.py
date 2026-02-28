# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.
# The comparator must report all-passed with zero failures — no exceptions.

# Usage: This is a typer CLI with 2 commands:
#   python test_run_megatron.py run --mode <mode>         Full: prepare + run baseline + run target + verify + comparator
#   python test_run_megatron.py compare --mode <mode> --dump-dir <path>
#                                                          Re-run comparator on existing dumps

import tempfile
from pathlib import Path
from typing import Annotated

import typer
from tests.e2e.conftest_dumper import check_dump_dir, clear_proxy_env, run_and_verify_comparator

import miles.utils.external_utils.command_utils as U
from miles.utils.misc import exec_command

app: typer.Typer = typer.Typer()

MODEL_NAME: str = "Qwen3-30B-A3B"
MODEL_TYPE: str = "qwen3-30B-A3B"
NUM_GPUS: int = 8

_RUN_DIR: Path = Path(tempfile.mkdtemp(prefix="test_run_megatron_"))

SOURCE_PATCHED_FIELDS: list[str] = ["layer_input", "attn_output", "pre_mlp_residual", "mlp_output"]

MEGATRON_SOURCE_PATCHER_CONFIG_YAML: str = """\
patches:
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention
    edits:
      - match: |
          inference_context = deprecate_inference_params(inference_context, inference_params)
        append: "dumper.dump('layer_input', hidden_states, dims='t(cp:zigzag,sp) 1 h')"
      - match: "nvtx_range_pop(suffix=\\"self_attention\\")"
        append: "dumper.dump('attn_output', attention_output_with_bias[0], dims='t(cp:zigzag,sp) 1 h')"
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_mlp
    edits:
      - match: "residual = hidden_states"
        append: "dumper.dump('pre_mlp_residual', residual, dims='t(cp:zigzag,sp) 1 h')"
      - match: "return self._forward_post_mlp(mlp_output_with_bias, residual)"
        prepend: "dumper.dump('mlp_output', mlp_output_with_bias[0], dims='t(cp:zigzag,sp) 1 h')"
"""

CONFIGS: dict[str, tuple[str, str]] = {
    # (baseline_parallel_args, target_parallel_args)
    "tp1_vs_tp2cp2": ("--tp 1", "--tp 2 --cp 2"),
    "tp1_vs_tp2pp2": ("--tp 1", "--tp 2 --pp 2"),
}


def _resolve_mode(mode: str) -> tuple[str, str, str]:
    if mode not in CONFIGS:
        raise typer.BadParameter(f"Unknown mode {mode!r}, valid: {list(CONFIGS.keys())}")
    baseline_args, target_args = CONFIGS[mode]
    return mode, baseline_args, target_args


def _prepare(dump_dir: Path) -> Path:
    """Download model, convert checkpoint, write source patcher config."""
    exec_command("mkdir -p /root/models")
    exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
    )
    exec_command(f"rm -rf {dump_dir}")

    source_patcher_path: Path = _RUN_DIR / "megatron_source_patcher.yaml"
    source_patcher_path.write_text(MEGATRON_SOURCE_PATCHER_CONFIG_YAML)
    return source_patcher_path


def _run_standalone(
    *,
    parallel_args: str,
    output_dir: Path,
    source_patcher_config: Path,
) -> None:
    """Run standalone Megatron forward via run_megatron CLI."""
    cmd: str = (
        f"python -m miles.utils.debug_utils.run_megatron run "
        f"--model-type {MODEL_TYPE} "
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load /root/{MODEL_NAME}_torch_dist "
        f"--output-dir {output_dir} "
        f"--seq-length 128 "
        f"--batch-size 1 "
        f"--prompt-mode math "
        f"--source-patcher-config {source_patcher_config} "
        f"--dumper-filter 'layer_id is None or layer_id < 3' "
        f"{parallel_args}"
    )
    exec_command(cmd)


def _verify_dumps(dump_dir: Path) -> None:
    """Verify dump directory structure."""
    check_dump_dir(
        phase_dir=dump_dir,
        exp_pattern="standalone",
        expected_fields=SOURCE_PATCHED_FIELDS,
    )
    print(f"Dump verification passed for {dump_dir}!")


@app.command()
def run(
    mode: Annotated[str, typer.Option(help="Config mode: " + ", ".join(CONFIGS.keys()))],
) -> None:
    """Full pipeline: prepare + run baseline + run target + verify + comparator."""
    config_name, baseline_args, target_args = _resolve_mode(mode)
    dump_dir: Path = _RUN_DIR / "dumps"
    print(f"Run directory: {_RUN_DIR}")

    source_patcher_config: Path = _prepare(dump_dir=dump_dir)
    clear_proxy_env()

    baseline_output: Path = dump_dir / "baseline"
    target_output: Path = dump_dir / "target"

    print(f"[test] Running baseline: {baseline_args}", flush=True)
    _run_standalone(
        parallel_args=baseline_args,
        output_dir=baseline_output,
        source_patcher_config=source_patcher_config,
    )

    print(f"[test] Running target: {target_args}", flush=True)
    _run_standalone(
        parallel_args=target_args,
        output_dir=target_output,
        source_patcher_config=source_patcher_config,
    )

    _verify_dumps(baseline_output)
    _verify_dumps(target_output)
    run_and_verify_comparator(
        baseline_dir=baseline_output / "standalone",
        target_dir=target_output / "standalone",
    )


@app.command()
def compare(
    mode: Annotated[str, typer.Option(help="Config mode: " + ", ".join(CONFIGS.keys()))],
    dump_dir: Annotated[str, typer.Option(help="Path to existing dump base directory")],
) -> None:
    """Re-run comparator on existing dumps (no training)."""
    _resolve_mode(mode)
    base: Path = Path(dump_dir)

    baseline_output: Path = base / "baseline"
    target_output: Path = base / "target"

    _verify_dumps(baseline_output)
    _verify_dumps(target_output)
    run_and_verify_comparator(
        baseline_dir=baseline_output / "standalone",
        target_dir=target_output / "standalone",
    )


if __name__ == "__main__":
    app()
