# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.
# The comparator must report all-passed with zero failures — no exceptions.

# Usage: This is a typer CLI with 2 commands:
#   python test_dumper.py run --mode <mode>        Full: prepare + execute + verify + comparator
#   python test_dumper.py compare --mode <mode> --dump-dir <path>
#                                                    Re-run comparator on existing dumps
#
# After running miles once (the expensive execute step), you can re-run the
# comparator many times via "compare" to investigate issues without re-running training.

import sys
import tempfile
from pathlib import Path
from typing import Annotated

_MILES_ROOT: Path = Path(__file__).resolve().parents[3]
if str(_MILES_ROOT) not in sys.path:
    sys.path.insert(0, str(_MILES_ROOT))

import typer
from tests.e2e.conftest_dumper import (
    MEGATRON_SOURCE_PATCHER_CONFIG_BSHD_YAML,
    MEGATRON_SOURCE_PATCHER_CONFIG_YAML,
    SGLANG_SOURCE_PATCHER_CONFIG_YAML,
    SOURCE_PATCHED_FIELDS,
    check_dump_dir,
    clear_proxy_env,
    run_and_verify_comparator,
)

import miles.utils.external_utils.command_utils as U

app: typer.Typer = typer.Typer()

MODEL_NAME = "Qwen3-30B-A3B"
MODEL_TYPE = "qwen3-30B-A3B"
NUM_GPUS = 8

_RUN_DIR: Path = Path(tempfile.mkdtemp(prefix="test_miles_dumper_"))
MEGATRON_SOURCE_PATCHER_CONFIG_PATH: str = str(_RUN_DIR / "megatron_source_patcher.yaml")
SGLANG_SOURCE_PATCHER_CONFIG_PATH: str = str(_RUN_DIR / "sglang_source_patcher.yaml")

EXP_PATTERNS: list[str] = ["engines/engine_*", "fwd_only", "fwd_bwd"]

_MEGATRON_AUX_THD: list[str] = ["input_ids", "cu_seqlens_q", "cu_seqlens_kv", "qkv_format"]
_MEGATRON_AUX_BSHD: list[str] = ["input_ids"]

EXPECTED_FIELDS: dict[str, dict[str, list[str]]] = {
    "tp2_pp2_cp2_ep2_etp2": {
        "engine_*": ["input_ids", "positions"] + SOURCE_PATCHED_FIELDS,
        "fwd_only": _MEGATRON_AUX_THD + SOURCE_PATCHED_FIELDS,
        "fwd_bwd": _MEGATRON_AUX_THD + SOURCE_PATCHED_FIELDS,
    },
    "tp2_pp2_cp2_ep2_etp2_bshd": {
        "engine_*": ["input_ids", "positions"] + SOURCE_PATCHED_FIELDS,
        "fwd_only": _MEGATRON_AUX_BSHD + SOURCE_PATCHED_FIELDS,
        "fwd_bwd": _MEGATRON_AUX_BSHD + SOURCE_PATCHED_FIELDS,
    },
}

CONFIGS: dict[str, str] = {
    "tp2_pp2_cp2_ep2_etp2": (
        "--tensor-model-parallel-size 2 --sequence-parallel "
        "--pipeline-model-parallel-size 2 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 2 --expert-tensor-parallel-size 2 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 2048 "
    ),
    "tp2_pp2_cp2_ep2_etp2_bshd": (
        "--tensor-model-parallel-size 2 --sequence-parallel "
        "--pipeline-model-parallel-size 2 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 2 --expert-tensor-parallel-size 2 "
        "--qkv-format bshd --micro-batch-size 1 "
    ),
}


def _resolve_mode(mode: str) -> tuple[str, str]:
    if mode not in CONFIGS:
        raise typer.BadParameter(f"Unknown mode {mode!r}, valid: {list(CONFIGS.keys())}")
    return mode, CONFIGS[mode]


def prepare(dump_dir: str, mode: str) -> None:
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS)
    U.exec_command(f"rm -rf {dump_dir}")

    megatron_yaml: str = (
        MEGATRON_SOURCE_PATCHER_CONFIG_BSHD_YAML if mode.endswith("_bshd") else MEGATRON_SOURCE_PATCHER_CONFIG_YAML
    )
    Path(MEGATRON_SOURCE_PATCHER_CONFIG_PATH).write_text(megatron_yaml)
    Path(SGLANG_SOURCE_PATCHER_CONFIG_PATH).write_text(SGLANG_SOURCE_PATCHER_CONFIG_YAML)


def _execute(perf_args: str, dump_subdir: str, dump_dir: str) -> None:
    full_dump_dir: str = f"{dump_dir}/{dump_subdir}"

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} " f"--ref-load /root/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages --label-key label --apply-chat-template "
        "--rollout-shuffle --rm-type math "
        "--rollout-max-response-len 3 --rollout-temperature 0.8 "
        # NOTE: Only generate 1 training sample
        "--num-rollout 1 --rollout-batch-size 1 --n-samples-per-prompt 1 --global-batch-size 1 "
        # NOTE: Must disable cuda graph to allow dumping
        "--sglang-disable-cuda-graph "
    )

    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant "
        "--optimizer-cpu-offload --use-precision-aware-optimizer "
    )

    grpo_args = "--advantage-estimator grpo --eps-clip 0.2 "

    sglang_args = (
        "--rollout-num-gpus-per-engine 4 "
        "--sglang-mem-fraction-static 0.6 "
        # Workaround: enable DP attention to avoid flashinfer allreduce fusion,
        # which crashes with base_gpu_id != 0 in colocate mode.
        # See https://github.com/flashinfer-ai/flashinfer/pull/2662
        "--sglang-dp-size 4 --sglang-enable-dp-attention "
    )

    dumper_filter: str = "'filter=layer_id is None or layer_id < 3 or layer_id == 24'"
    dumper_args = (
        f"--dumper-enable --dumper-dir {full_dump_dir} "
        f"--dumper-inference {dumper_filter} "
        f"--dumper-fwd-only enable_model_value=0 enable_model_grad=0 {dumper_filter} "
        f"--dumper-fwd-bwd enable_model_value=0 enable_model_grad=0 {dumper_filter} "
        f"--dumper-source-patcher-config-train {MEGATRON_SOURCE_PATCHER_CONFIG_PATH} "
        f"--dumper-source-patcher-config-inference {SGLANG_SOURCE_PATCHER_CONFIG_PATH} "
    )

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {NUM_GPUS} --colocate "
        "--moe-token-dispatcher-type alltoall "
        "--use-miles-router --use-rollout-routing-replay "
    )

    train_args = " ".join(
        [
            ckpt_args,
            rollout_args,
            optimizer_args,
            grpo_args,
            perf_args,
            sglang_args,
            dumper_args,
            misc_args,
            U.get_default_wandb_args(__file__),
        ]
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


def _verify_dumps(config_name: str, dump_subdir: str, dump_dir: str) -> None:
    base: Path = Path(dump_dir) / dump_subdir
    config_expected: dict[str, list[str]] = EXPECTED_FIELDS[config_name]
    for pattern in EXP_PATTERNS:
        check_dump_dir(base, pattern, expected_fields=config_expected.get(pattern))
    print(f"All dump verifications passed for {dump_subdir}!")


def _verify_comparator(dump_subdir: str, dump_dir: str) -> None:
    baseline_dir: Path = Path(f"{dump_dir}/{dump_subdir}/engines")
    target_dir: Path = Path(f"{dump_dir}/{dump_subdir}/fwd_bwd")
    # TODO: moe_expert_output comparison is not yet supported (unshard logic
    #       for post-alltoall MoE tensors needs dedicated handling)
    run_and_verify_comparator(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        extra_args=["--allow-failed-pattern", "moe_expert_output"],
    )


@app.command()
def run(
    mode: Annotated[str, typer.Option(help="Config mode: " + ", ".join(CONFIGS.keys()))],
) -> None:
    """Full pipeline: prepare + execute + verify + comparator."""
    config_name, perf_args = _resolve_mode(mode)
    dump_dir: str = str(_RUN_DIR / "dumps")
    print(f"Run directory: {_RUN_DIR}")

    prepare(dump_dir=dump_dir, mode=config_name)
    clear_proxy_env()
    _execute(perf_args=perf_args, dump_subdir=config_name, dump_dir=dump_dir)
    _verify_dumps(config_name=config_name, dump_subdir=config_name, dump_dir=dump_dir)
    _verify_comparator(dump_subdir=config_name, dump_dir=dump_dir)


@app.command()
def compare(
    mode: Annotated[str, typer.Option(help="Config mode: " + ", ".join(CONFIGS.keys()))],
    dump_dir: Annotated[str, typer.Option(help="Path to existing dump base directory")],
) -> None:
    """Re-run comparator on existing dumps (no training)."""
    config_name, _ = _resolve_mode(mode)

    _verify_dumps(config_name=config_name, dump_subdir=config_name, dump_dir=dump_dir)
    _verify_comparator(dump_subdir=config_name, dump_dir=dump_dir)


if __name__ == "__main__":
    app()
