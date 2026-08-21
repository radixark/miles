"""Resume LoRA weights, scheduler position, rollout ID, and dataset cursor.

Optimizer parameter-state persistence is covered separately. Stage B deliberately
uses ``--no-load-optim`` so this progress-resume layer is independently testable.

Requires: 4 GPUs, Qwen2.5-0.5B-Instruct model, GSM8K dataset.
Triggered by label: run-ci-lora
"""

import glob
import os

import torch

from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(est_time=900, suite="stage-c-4-gpu-h200", labels=["lora"])

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4

STAGE_A = "/root/checkpoints/lora-progress-resume-a"
STAGE_B = "/root/checkpoints/lora-progress-resume-b"


def prepare():
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command_cpu("hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/datasets/gsm8k")
    U.exec_command_cpu(f"rm -rf {STAGE_A} {STAGE_B}")


def _train_args(
    save_dir: str,
    adapter_path: str | None,
    *,
    debug_exit_after_rollout: int | None = None,
    no_load_optim: bool = False,
) -> str:
    resume = f"--lora-adapter-path {adapter_path} " if adapter_path else ""
    debug_exit = f"--debug-exit-after-rollout {debug_exit_after_rollout} " if debug_exit_after_rollout else ""
    optimizer_load = "--no-load-optim " if no_load_optim else ""
    return (
        f"--hf-checkpoint /root/models/{MODEL_NAME}/ "
        "--megatron-to-hf-mode bridge "
        "--lora-rank 32 --lora-alpha 32 --lora-dropout 0.0 "
        '--target-modules "all-linear" '
        f"{resume}"
        f"{optimizer_load}"
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages --label-key label --apply-chat-template --rollout-shuffle "
        "--rm-type math "
        "--num-rollout 4 --rollout-batch-size 8 --n-samples-per-prompt 8 "
        "--rollout-max-response-len 512 --rollout-temperature 1.0 --global-batch-size 32 "
        "--advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type low_var_kl "
        "--kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
        "--optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
        "--tensor-model-parallel-size 1 --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 4096 "
        "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.4 "
        f"{U.get_default_wandb_args(__file__)} "
        f"--save-interval 1 --save {save_dir} "
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
        "--ci-test "
        f"{debug_exit}"
        "--attention-backend flash --calculate-per-token-loss --use-miles-router "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {NUM_GPUS} --colocate "
    )


def _latest_adapter(save_dir: str) -> str:
    iters = sorted(glob.glob(os.path.join(save_dir, "iter_*", "adapter")))
    assert iters, f"no adapter checkpoint under {save_dir}"
    return iters[-1]


def _assert_adapter_is_finite(adapter_dir: str) -> None:
    shards = sorted(glob.glob(os.path.join(adapter_dir, "adapter_megatron_*.pt")))
    assert shards, f"no adapter shards in {adapter_dir}"
    for shard in shards:
        tensors = torch.load(shard, map_location="cpu", weights_only=True)
        assert tensors and all(torch.isfinite(value).all() for value in tensors.values())


def _training_state(adapter_dir: str) -> dict:
    return torch.load(
        os.path.join(adapter_dir, "training_state_rank0.pt"),
        map_location="cpu",
        weights_only=False,
    )


def _scheduler_step(adapter_dir: str) -> int:
    return _training_state(adapter_dir)["opt_param_scheduler"]["num_steps"]


def _dataset_offset(save_dir: str, rollout_id: int) -> int:
    state = torch.load(
        os.path.join(save_dir, "rollout", f"global_dataset_state_dict_{rollout_id}.pt"),
        map_location="cpu",
        weights_only=False,
    )
    return state["sample_offset"]


def _execute_stages():
    U.execute_train(
        train_args=_train_args(STAGE_A, adapter_path=None, debug_exit_after_rollout=2),
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )
    stage_a_adapter = _latest_adapter(STAGE_A)
    _assert_adapter_is_finite(stage_a_adapter)
    stage_a_dataset_offset = _dataset_offset(STAGE_A, rollout_id=1)

    U.execute_train(
        train_args=_train_args(STAGE_B, adapter_path=stage_a_adapter, no_load_optim=True),
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )

    resumed = sorted(glob.glob(os.path.join(STAGE_B, "iter_*", "adapter")))
    assert [os.path.basename(os.path.dirname(path)) for path in resumed] == ["iter_0000002", "iter_0000003"]
    for adapter_dir in resumed:
        _assert_adapter_is_finite(adapter_dir)
    first_scheduler_step = _scheduler_step(resumed[0])
    assert first_scheduler_step > 0
    assert _scheduler_step(resumed[-1]) == first_scheduler_step + 64
    assert _dataset_offset(STAGE_B, rollout_id=2) == stage_a_dataset_offset + 8
    assert _dataset_offset(STAGE_B, rollout_id=3) == stage_a_dataset_offset + 16


def execute():
    external_ray = os.environ.get("MILES_SCRIPT_EXTERNAL_RAY")
    if external_ray is None:
        U.exec_command_cpu(f"ray start --head --node-ip-address 127.0.0.1 --num-gpus {NUM_GPUS} --disable-usage-stats")
        os.environ["MILES_SCRIPT_EXTERNAL_RAY"] = "1"
    try:
        _execute_stages()
    finally:
        if external_ray is None:
            os.environ.pop("MILES_SCRIPT_EXTERNAL_RAY")
            U.exec_command_cpu("ray stop --force || true")


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
