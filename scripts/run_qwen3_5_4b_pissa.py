"""Matched Qwen3.5-4B LoRA/PiSSA GSM8K pilot on one eight-GPU node.

Requires the original HF checkpoint and its converted torch_dist checkpoint in
--model-dir, plus a PiSSA residual base, adapter, and converted residual checkpoint.
Create the latter with tools.prepare_pissa and the normal Miles checkpoint converter.
No optimizer state is imported. Both arms use the original frozen reference.
The native Miles audit events, rollout dumps, and checkpoints retain the results.

Args:
  --model-dir: Qwen3.5-4B, Qwen3.5-4B_torch_dist, Qwen3.5-4B-pissa-r32
    (residual_base/ and adapter/), and Qwen3.5-4B-pissa-r32_torch_dist.
  --data-dir: train.parquet and test.parquet, with messages and label columns.
    The pilot uses all GSM8K train tasks and the first 128 fixed test tasks.
    Append "Please reason step by step, and put your final answer within \\boxed{}."
    to each question; label is the answer after ####. Thinking is disabled.
  --arm: both (sequential, default), lora, or pissa.
  --objective: grpo (matched pilot), tis, gspo, score_centering, or score_centering_tis.
    The other objectives are separate experiments, not the matched pilot's results.
  --num-rollout / --lr / --seed / --response-tokens: shared by both arms.

    python -m scripts.run_qwen3_5_4b_pissa --model-dir /root/models \
        --data-dir /root/datasets/gsm8k --output-dir /root/shared_data
"""

import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import typer

from miles.utils.external_utils import command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=lambda: U.create_run_id())
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets/gsm8k"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8
    arm: Literal["both", "lora", "pissa"] = "both"
    objective: Literal["grpo", "tis", "gspo", "score_centering", "score_centering_tis"] = "grpo"
    num_rollout: int = 100
    lr: float = 3e-6
    seed: int = 42
    response_tokens: int = 2048
    eval_interval: int = 20

    def __post_init__(self):
        if self.num_nodes != 1 or self.num_gpus_per_node != 8:
            raise ValueError("This matched recipe uses exactly one eight-GPU node")
        if min(self.num_rollout, self.response_tokens, self.eval_interval) <= 0 or self.lr <= 0:
            raise ValueError("Training budgets, evaluation interval, and learning rate must be positive")


def _algorithm_args(objective: str) -> str:
    if objective == "gspo":
        return "--advantage-estimator gspo"
    if objective == "tis":
        return "--advantage-estimator grpo --use-tis --tis-clip 2.0"
    if objective in ("score_centering", "score_centering_tis"):
        mode = "tis" if objective == "score_centering_tis" else "none"
        return (
            "--advantage-estimator grpo --loss-type score_centering "
            "--rollout-top-logprobs-num 128 --rollout-top-p 1.0 --rollout-top-k -1 "
            "--use-rollout-logprobs --disable-grpo-std-normalization --calculate-per-token-loss "
            f"--score-centering-is {mode} --score-centering-tis-clip 2.0"
        )
    if objective != "grpo":
        raise ValueError(f"Unknown objective: {objective}")
    return "--advantage-estimator grpo"


def _train_args(args: ScriptArgs, arm: str) -> str:
    original = Path(args.model_dir) / "Qwen3.5-4B"
    pissa = Path(args.model_dir) / "Qwen3.5-4B-pissa-r32"
    out = Path(args.output_dir) / args.run_id / arm
    checkpoint = (
        f"--hf-checkpoint {shlex.quote(str(original if arm == 'lora' else pissa / 'residual_base'))} "
        f"--load {shlex.quote(str(original if arm == 'lora' else pissa) + '_torch_dist')} "
        f"--ref-load {shlex.quote(str(original) + '_torch_dist')} "
        f"--save {shlex.quote(str(out / 'checkpoints'))} --save-interval {args.eval_interval} "
        "--megatron-to-hf-mode raw --no-load-optim --no-load-rng --finetune "
        "--lora-rank 32 --lora-alpha 32 --lora-dropout 0 --lora-type canonical_lora "
        "--lora-A-init-method kaiming --target-modules attn,mlp --no-gradient-accumulation-fusion "
    )
    if arm == "pissa":
        checkpoint += f"--lora-adapter-path {shlex.quote(str(pissa / 'adapter'))} "
    rollout = (
        f"--prompt-data {shlex.quote(str(Path(args.data_dir) / 'train.parquet'))} "
        "--input-key messages --label-key label --apply-chat-template "
        "--apply-chat-template-kwargs '{\"enable_thinking\": false}' "
        f"--rollout-shuffle --rollout-seed {args.seed} --seed {args.seed} --rm-type math "
        f"--num-rollout {args.num_rollout} --rollout-batch-size 32 --n-samples-per-prompt 8 "
        f"--rollout-max-response-len {args.response_tokens} --rollout-temperature 1 "
        "--global-batch-size 256 --balance-data "
    )
    evaluation = (
        f"--eval-interval {args.eval_interval} --eval-prompt-data gsm8k "
        f"{shlex.quote(str(Path(args.data_dir) / 'test.parquet'))} "
        "--eval-input-key messages --eval-label-key label --n-samples-per-eval-prompt 1 "
        f"--eval-temperature 0 --eval-max-response-len {args.response_tokens} "
    )
    algorithm = (
        f"{_algorithm_args(args.objective)} --use-kl-loss --kl-loss-coef 0 --kl-loss-type low_var_kl "
        "--entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28 "
    )
    optimizer = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
    )
    perf = (
        "--tensor-model-parallel-size 2 --sequence-parallel --expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 8192 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
    )
    sglang = (
        "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.4 --sglang-dtype bfloat16 "
        "--sglang-max-lora-rank 32 --sglang-lora-backend triton --sglang-decode-log-interval 1000 "
    )
    misc = (
        "--attention-dropout 0 --hidden-dropout 0 --update-weight-buffer-size 536870912 "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} --colocate "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash "
        f"--check-lora-weight-equal --dump-details {shlex.quote(str(out / 'details'))} "
        f"--use-tensorboard --tb-project-name {shlex.quote(str(out.parent / 'tensorboard'))} "
        f"--tb-experiment-name {arm} "
    )
    return checkpoint + rollout + evaluation + algorithm + optimizer + perf + sglang + misc


def execute(args: ScriptArgs):
    backend = args.create_backend()
    for arm in ("lora", "pissa") if args.arm == "both" else (args.arm,):
        if (Path(args.output_dir) / args.run_id / arm / "checkpoints").exists():
            raise ValueError("Use a new run ID; this recipe starts from initialization, not a saved optimizer")
        config = replace(args, ray_submission_id=f"pissa-{args.run_id}-{arm}")
        backend.execute_train(
            train_args=_train_args(args, arm)
            + " "
            + U.get_default_wandb_args(__file__, run_id=f"{args.run_id}-{arm}"),
            config=config,
            num_gpus_per_node=args.num_gpus_per_node,
            megatron_model_type="qwen3.5-4B",
            megatron_path=args.megatron_path,
            extra_env_vars={"PYTHONPATH": f"{U.repo_base_dir}:{args.megatron_path}"},
            job_lifetime="launcher",
        )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
