import os
from pathlib import Path

import torch

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-30B-A3B"
MODEL_TYPE = "qwen3-30B-A3B"
NUM_GPUS = 8
DUMP_DIR = "/tmp/test_miles_dumper"

EXP_PATTERNS = ["engine_*", "fwd_only", "fwd_bwd"]

EXPECTED_FIELDS: dict[str, list[str]] = {
    "engine_*": ["input_ids", "positions"],
    "fwd_only": ["input_ids", "cu_seqlens_q", "cu_seqlens_kv", "qkv_format"],
    "fwd_bwd": ["input_ids", "cu_seqlens_q", "cu_seqlens_kv", "qkv_format"],
}

# Two configs that together cover all parallelism dimensions:
#   Config A: TP=2, SP, PP=2, EP=2, DP=2            → covers DP
#   Config B: TP=2, SP, PP=2, CP=2, EP=2, eTP=2     → covers CP, expert_TP
CONFIGS: dict[str, str] = {
    "tp2_pp2_ep2_dp2": (
        "--tensor-model-parallel-size 2 --sequence-parallel "
        "--pipeline-model-parallel-size 2 "
        "--expert-model-parallel-size 2 --expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 2048 "
    ),
    "tp2_pp2_cp2_ep2_etp2": (
        "--tensor-model-parallel-size 2 --sequence-parallel "
        "--pipeline-model-parallel-size 2 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 2 --expert-tensor-parallel-size 2 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 2048 "
    ),
}


def prepare() -> None:
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS)
    U.exec_command(f"rm -rf {DUMP_DIR}")


def _execute(perf_args: str, dump_subdir: str) -> None:
    dump_dir = f"{DUMP_DIR}/{dump_subdir}"

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} " f"--ref-load /root/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages --label-key label --apply-chat-template "
        "--rollout-shuffle --rm-type math "
        "--num-rollout 1 --rollout-batch-size 4 --n-samples-per-prompt 2 "
        "--rollout-max-response-len 20 --rollout-temperature 0.8 "
        "--global-batch-size 8 "
    )

    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant "
        "--optimizer-cpu-offload --use-precision-aware-optimizer "
    )

    grpo_args = "--advantage-estimator grpo --eps-clip 0.2 "

    sglang_args = "--rollout-num-gpus-per-engine 8 " "--sglang-mem-fraction-static 0.6 "

    dumper_args = (
        f"--dumper-enable --dumper-dir {dump_dir} "
        "--dumper-fwd-only enable_model_value=0 enable_model_grad=0 "
        "--dumper-fwd-bwd enable_model_value=0 enable_model_grad=0 "
    )

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {NUM_GPUS} --colocate "
        "--moe-token-dispatcher-type alltoall "
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


def _check_dump_dir(phase_dir: Path, exp_pattern: str, expected_fields: list[str] | None = None) -> None:
    assert phase_dir.exists(), f"Missing dump dir: {phase_dir}"
    dump_subdirs = list(phase_dir.glob(exp_pattern))
    assert len(dump_subdirs) > 0, f"No {exp_pattern} subdirs in {phase_dir}"
    dump_files = list(dump_subdirs[0].glob("*.pt"))
    assert len(dump_files) > 0, f"No .pt files in {dump_subdirs[0]}"
    sample = torch.load(dump_files[0], weights_only=False)
    assert isinstance(sample, dict), f"Unexpected type: {type(sample)}"
    assert "value" in sample and "meta" in sample, f"Missing keys: {sample.keys()}"

    if expected_fields:
        for field in expected_fields:
            matches = list(phase_dir.rglob(f"*name={field}.pt"))
            assert len(matches) > 0, f"Expected field '{field}' not found under {phase_dir}"


def verify(dump_subdir: str) -> None:
    base = Path(f"{DUMP_DIR}/{dump_subdir}")
    for pattern in EXP_PATTERNS:
        _check_dump_dir(base, pattern, expected_fields=EXPECTED_FIELDS.get(pattern))
    print(f"All dump verifications passed for {dump_subdir}!")


def _select_configs() -> dict[str, str]:
    selected = os.environ["MILES_TEST_DUMPER_CONFIG"]
    if selected not in CONFIGS:
        raise ValueError(f"Unknown MILES_TEST_DUMPER_CONFIG={selected!r}, " f"valid values: {list(CONFIGS.keys())}")
    return {selected: CONFIGS[selected]}


if __name__ == "__main__":
    configs = _select_configs()
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    for config_name, perf_args in configs.items():
        _execute(perf_args=perf_args, dump_subdir=config_name)
        verify(config_name)
