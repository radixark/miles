"""Warm-starting a LoRA run from its own checkpoint has to continue it, not start over.

No test in the tree passes ``--lora-adapter-path``, so the save/resume pair -- the whole point
of multi-stage LoRA RL -- has no coverage. Both ways it can fail are silent: the adapter loads
cleanly and the job keeps running either way, only with a policy that is garbage or is the
initialization again.

Train briefly with ``--save``, resume from what that wrote, then assert the resumed adapter is
finite, still moving, and starts where the first run stopped.

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

STAGE_A = "/root/checkpoints/lora-resume-a"
STAGE_B = "/root/checkpoints/lora-resume-b"


def prepare():
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command_cpu("hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/datasets/gsm8k")
    U.exec_command_cpu(f"rm -rf {STAGE_A} {STAGE_B}")


def _train_args(save_dir: str, adapter_path: str | None) -> str:
    resume = f"--lora-adapter-path {adapter_path} " if adapter_path else ""
    return (
        f"--hf-checkpoint /root/models/{MODEL_NAME}/ "
        "--megatron-to-hf-mode bridge "
        "--lora-rank 32 --lora-alpha 32 --lora-dropout 0.0 "
        '--target-modules "all-linear" '
        f"{resume}"
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages --label-key label --apply-chat-template --rollout-shuffle "
        "--rm-type math "
        "--num-rollout 2 --rollout-batch-size 8 --n-samples-per-prompt 8 "
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
        "--attention-backend flash --calculate-per-token-loss --use-miles-router "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {NUM_GPUS} --colocate "
    )


def _latest_adapter(save_dir: str) -> str:
    iters = sorted(glob.glob(os.path.join(save_dir, "iter_*", "adapter")))
    assert iters, f"no adapter checkpoint under {save_dir}"
    return iters[-1]


def _assert_adapter_is_trainable(adapter_dir: str, *, stage: str) -> dict:
    shards = sorted(glob.glob(os.path.join(adapter_dir, "adapter_megatron_*.pt")))
    assert shards, f"{stage}: no adapter shards in {adapter_dir}"

    merged = {}
    for shard in shards:
        tensors = torch.load(shard, map_location="cpu", weights_only=True)
        nan = [k for k, v in tensors.items() if not torch.isfinite(v).all()]
        assert not nan, (
            f"{stage}: {len(nan)}/{len(tensors)} adapter tensors are non-finite in "
            f"{os.path.basename(shard)}, e.g. {nan[:3]}"
        )
        merged.update({f"{os.path.basename(shard)}:{k}": v for k, v in tensors.items()})
    return merged


def _trained_component_norm(adapter: dict) -> float:
    """Norm over the adapter's zero-initialized side.

    Megatron names the two factors ``adapter.linear_in`` (LoRA A) and ``adapter.linear_out``
    (LoRA B). ``linear_out`` starts at exactly zero, so its norm is everything training
    produced. The whole-adapter norm is not usable for this: ``linear_in``'s random init
    dominates it (~8 against ~0.01) and barely moves however training goes.
    """
    keys = [k for k in adapter if "linear_out" in k]
    assert keys, f"no linear_out tensors among {list(adapter)[:3]}; the adapter naming changed"
    return sum(float(adapter[k].float().norm()) ** 2 for k in keys) ** 0.5


def _optimizer_step(adapter_dir: str) -> int:
    state = torch.load(
        os.path.join(adapter_dir, "training_state_rank0.pt"),
        map_location="cpu",
        weights_only=False,
    )
    return state["optimizer"]["optimizer"]["param_groups"][0]["step"]


def _execute_stages():
    U.execute_train(
        train_args=_train_args(STAGE_A, adapter_path=None),
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )
    stage_a_adapter = _latest_adapter(STAGE_A)
    stage_a = _assert_adapter_is_trainable(stage_a_adapter, stage="stage A")
    stage_a_step = _optimizer_step(stage_a_adapter)

    # A resume needs the moments too, and DistributedOptimizer keeps those out of
    # state_dict() by design.
    assert glob.glob(os.path.join(stage_a_adapter, "optimizer_param_state*.pt")), (
        f"stage A wrote no optimizer parameter state into {stage_a_adapter}; a resume from it "
        f"would rebuild Adam with zero moments"
    )

    U.execute_train(
        train_args=_train_args(STAGE_B, adapter_path=stage_a_adapter),
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )

    resumed = sorted(glob.glob(os.path.join(STAGE_B, "iter_*", "adapter")))
    assert len(resumed) >= 2, f"expected at least two checkpoints under {STAGE_B}, got {resumed}"
    first = _assert_adapter_is_trainable(resumed[0], stage="stage B first")
    last = _assert_adapter_is_trainable(resumed[-1], stage="stage B last")

    moved = sum(1 for k in first if k in last and not torch.equal(first[k], last[k]))
    assert moved, (
        "stage B produced no adapter change across its checkpoints, so the resumed run is not "
        "training; a warm start that silently stops learning is the failure this test exists for"
    )
    assert _optimizer_step(resumed[0]) == stage_a_step + 2
    assert _optimizer_step(resumed[-1]) == stage_a_step + 4

    # A fresh restart also stays finite and moves. The zero-initialized factor must
    # remain near stage A's final norm rather than collapse toward initialization.
    before = _trained_component_norm(stage_a)
    after = _trained_component_norm(first)
    assert after > 0.9 * before, (
        f"stage B started from a trained-component norm of {after:.5f} against stage A's final "
        f"{before:.5f}: a resume continues from the saved adapter, so a collapse back toward the "
        f"zero initialization means the checkpoint was loaded and then discarded"
    )


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
