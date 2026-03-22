# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.
# The comparator must report all-passed with zero failures — no exceptions.

# Usage: This is a typer CLI with 2 commands:
#   python test_run_megatron.py run --mode <mode>         Full: prepare + run baseline + run target + compare
#   python test_run_megatron.py compare --mode <mode> --dump-dir <path>
#                                                          Re-run comparator on existing dumps

import dataclasses
import sys
import tempfile
from pathlib import Path
from typing import Annotated

_MILES_ROOT: Path = Path(__file__).resolve().parents[3]
if str(_MILES_ROOT) not in sys.path:
    sys.path.insert(0, str(_MILES_ROOT))

import typer
from tests.e2e.conftest_dumper import MEGATRON_PATCHER_YAMLS, clear_proxy_env

import miles.utils.external_utils.command_utils as U
from miles.utils.debug_utils.run_megatron.cli.parallel_utils import ParallelConfig, parse_parallel_args
from miles.utils.misc import exec_command

app: typer.Typer = typer.Typer()

HF_REPO: str = "fzyzcjy/Qwen3-30B-A3B-5layer"
MODEL_NAME: str = "Qwen3-30B-A3B-5layer"
MODEL_TYPE: str = "qwen3-30B-A3B-5layer"
NUM_GPUS: int = 8
NUM_LAYERS: int = 5

_RUN_DIR: Path = Path(tempfile.mkdtemp(prefix="test_run_megatron_"))


@dataclasses.dataclass(frozen=True)
class _ModeConfig:
    baseline_args: str
    target_args: str
    format: str = "bshd"
    target_extra_megatron_args: str = ""


CONFIGS: dict[str, _ModeConfig] = {
    "tp1_vs_tp2pp2cp2": _ModeConfig(
        baseline_args="--tp 1",
        target_args="--tp 2 --pp 2 --cp 2 --ep 1",
        format="bshd",
        target_extra_megatron_args="--decoder-first-pipeline-num-layers 3 --decoder-last-pipeline-num-layers 2",
    ),
    # TODO: THD mode not yet supported — standalone worker (batch.py) hardcodes
    # qkv_format="bshd" and does not construct PackedSeqParams. Enabling THD
    # requires reworking CP slicing, qkv_format passthrough, and PackedSeqParams
    # construction in the worker.
    # "tp1_vs_tp2pp2cp2_thd": _ModeConfig(
    #     baseline_args="--tp 1",
    #     target_args="--tp 2 --pp 2 --cp 2 --ep 1",
    #     format="thd",
    #     target_extra_megatron_args="--decoder-first-pipeline-num-layers 3 --decoder-last-pipeline-num-layers 2",
    # ),
}


def _resolve_mode(mode: str) -> tuple[str, _ModeConfig]:
    if mode not in CONFIGS:
        raise typer.BadParameter(f"Unknown mode {mode!r}, valid: {list(CONFIGS.keys())}")
    return mode, CONFIGS[mode]


def _prepare(dump_dir: Path, config: _ModeConfig) -> Path:
    """Download model, convert checkpoint, write source patcher config."""
    exec_command("mkdir -p /root/models")
    exec_command(f"hf download {HF_REPO} --local-dir /root/models/{MODEL_NAME}")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=min(NUM_GPUS, NUM_LAYERS),
    )
    exec_command(f"rm -rf {dump_dir}")

    source_patcher_path: Path = _RUN_DIR / "megatron_source_patcher.yaml"
    yaml_content: str = (
        MEGATRON_PATCHER_YAMLS[config.format].replace(" ep:replicated", "").replace(" etp:replicated", "")
    )
    source_patcher_path.write_text(yaml_content)
    return source_patcher_path


@app.command()
def run(
    mode: Annotated[str, typer.Option(help="Config mode: " + ", ".join(CONFIGS.keys()))],
) -> None:
    """Full pipeline: prepare + run baseline + run target + compare."""
    _config_name, config = _resolve_mode(mode)
    dump_dir: Path = _RUN_DIR / "dumps"
    print(f"Run directory: {_RUN_DIR}")

    source_patcher_config: Path = _prepare(dump_dir=dump_dir, config=config)
    clear_proxy_env()

    extra_args: str = "--attention-backend flash"

    target_extra_args_part: str = ""
    if config.target_extra_megatron_args:
        target_extra_args_part = f"--target-extra-args '{config.target_extra_megatron_args}' "

    cmd: str = (
        f"python -m miles.utils.debug_utils.run_megatron run-and-compare "
        f"--model-type {MODEL_TYPE} "
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load /root/{MODEL_NAME}_torch_dist "
        f"--output-base-dir {dump_dir} "
        f"--baseline '{config.baseline_args}' "
        f"--target '{config.target_args}' "
        f"--seq-length 128 "
        f"--batch-size 1 "
        f"--prompt-mode math "
        f"--sp "
        f"--source-patcher-config {source_patcher_config} "
        f"--dumper-filter 'layer_id is None or layer_id < 3' "
        f"--diff-threshold 0.002 "
        f"--logprob-threshold 0.06 "
        f"{target_extra_args_part}"
        f"--extra-args '{extra_args}'"
    )
    exec_command(cmd)


@app.command()
def compare(
    mode: Annotated[str, typer.Option(help="Config mode: " + ", ".join(CONFIGS.keys()))],
    dump_dir: Annotated[str, typer.Option(help="Path to existing dump base directory")],
) -> None:
    """Re-run comparator on existing dumps (no training)."""
    _config_name, config = _resolve_mode(mode)
    base: Path = Path(dump_dir)

    baseline_dir_name: str = ParallelConfig.from_parsed_args(parse_parallel_args(config.baseline_args)).dir_name()
    target_dir_name: str = ParallelConfig.from_parsed_args(parse_parallel_args(config.target_args)).dir_name()

    cmd: str = (
        f"python -m miles.utils.debug_utils.run_megatron compare "
        f"--baseline-dir {base / baseline_dir_name / 'standalone'} "
        f"--target-dir {base / target_dir_name / 'standalone'}"
    )
    exec_command(cmd)


if __name__ == "__main__":
    app()
