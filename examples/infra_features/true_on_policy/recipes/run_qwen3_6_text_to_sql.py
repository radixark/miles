"""Qwen3.6-35B-A3B, fully async multi-turn text-to-SQL (BIRD-Platinum) with weights up to 8 versions stale: the five arms
of the blog's async text-to-SQL figures.

All arms: the text-to-SQL episode and reward of `examples/text_to_sql`, 64 questions x 16 attempts per step, 3,072
tokens per turn, 29,696 per episode, temperature 1.0; fully async with retract-mode weight updates, at most 8 versions
of staleness, the radix (prefix) cache on; decoupled PPO with the old-policy pass skipped, so the importance weight
w = pi_old / mu is taken against the rollout log-probs; lr 1e-6 with 10-step warmup and cosine decay pinned to 160
steps; eval every 5 steps on BIRD-Platinum validation and Arcwise at temperature 1e-5 (TOP engines refuse greedy).
4 trainer nodes (EP8, DP32) and 4 rollout nodes (16 two-GPU engines), 64 H200.

| `--arm` | kernels | R3 | IS weight cap |
|---|---|---|---|
| `top-r3-nocap` | TOP | yes | none |
| `top-cap2` | TOP | no | 2 |
| `std-cap2` | standard | yes | 2 |
| `top-nocap` | TOP | no | none |
| `std-r3-nocap` | standard | yes | none |

Under TOP the rollout log-probs equal the trainer's at the sampling weights, so w carries staleness only; the standard
arms' w also carries the kernel mismatch.

This recipe is a translation, not the runs' exact command: the runs held a group older than 8 versions back until it
fit, with a staleness policy this tree does not have; here `--async-submit-within-staleness` bounds what the producer
submits instead. See the recipes README for every difference.

Requires: `python examples/text_to_sql/run_qwen3_6_35b_a3b_text_to_sql.py prepare` (recipe package, checkpoint and
its torch_dist conversion, data); an 8-node Ray cluster already joined (MILES_SCRIPT_EXTERNAL_RAY=1) whose head is a
rollout node, since the rollout manager and the grader's SQLite scratch run there.

Args:
    --arm                 one of the five arms above
    --num-rollout         training steps (the figures plot steps 0-59)
    --model-dir / --data-dir / --output-dir   as for examples/text_to_sql
    --megatron-path       the Megatron-LM checkout

Usage:
    MILES_SCRIPT_EXTERNAL_RAY=1 python examples/infra_features/true_on_policy/recipes/run_qwen3_6_text_to_sql.py \
        --arm top-r3-nocap
"""

from dataclasses import dataclass
from typing import Literal

import typer

from miles.true_on_policy.launch import top_launch
from miles.utils.external_utils import command_utils

_EPISODE = "examples.text_to_sql.sql_episode"
# kernels TOP, rollout routing replay, IS weight cap
_ARMS = {
    "top-r3-nocap": (True, True, "inf"),
    "top-cap2": (True, False, "2"),
    "std-cap2": (False, True, "2"),
    "top-nocap": (True, False, "inf"),
    "std-r3-nocap": (False, True, "inf"),
}


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    arm: Literal["top-r3-nocap", "top-cap2", "std-cap2", "top-nocap", "std-r3-nocap"] = "top-r3-nocap"
    num_rollout: int = 60
    model_name: str = "Qwen3.6-35B-A3B"
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8
    # 4 trainer nodes and 4 rollout nodes.
    num_nodes: int = 8

    @property
    def top(self) -> bool:
        return _ARMS[self.arm][0]

    @property
    def hf_checkpoint(self) -> str:
        return f"{self.model_dir}/{self.model_name}"

    @property
    def megatron_model_type(self) -> str:
        return "qwen3.6-35B-A3B-top" if self.top else "qwen3.6-35B-A3B"

    @property
    def sqlite_scratch_dir(self) -> str:
        return f"{self.output_dir}/text-to-sql-sqlite-{self.arm}"


def _ckpt_args(args: ScriptArgs) -> str:
    if args.top:
        # The TOP trainer loads the HF checkpoint directly.
        return f"--hf-checkpoint {args.hf_checkpoint} --load {args.hf_checkpoint} --ref-load {args.hf_checkpoint} "
    return f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.hf_checkpoint}_torch_dist "


def _rollout_args(args: ScriptArgs) -> str:
    prompts = f"{args.data_dir}/text-to-sql"
    return (
        f"--prompt-data {prompts}/train.jsonl --input-key prompt --label-key label --tool-key tools "
        "--apply-chat-template --rollout-shuffle "
        f"--custom-generate-function-path {_EPISODE}.generate --custom-rm-path {_EPISODE}.reward_func "
        f"--reward-key score --num-rollout {args.num_rollout} --rollout-batch-size 64 --n-samples-per-prompt 16 "
        "--global-batch-size 1024 --rollout-max-response-len 3072 --max-seq-len 29696 --rollout-temperature 1.0 "
        "--fully-async --pause-generation-mode retract --update-weights-interval 1 "
        "--max-weight-staleness 8 --async-submit-within-staleness --async-data-buffer-capacity-factor 9 "
        "--async-unused-samples-handler retry --async-max-consecutive-aborted-groups 64 "
        "--async-max-concurrent-samples 2048 --sglang-enable-prefill-weight-versions "
        f"--eval-interval 5 --eval-prompt-data validation {prompts}/validation.jsonl arcwise {prompts}/arcwise.jsonl "
        "--eval-reward-key correct --n-samples-per-eval-prompt 1 --eval-temperature 1e-05 --eval-max-response-len 3072 "
    )


def _algo_optimizer_args(args: ScriptArgs) -> str:
    _, routing_replay, cap = _ARMS[args.arm]
    return (
        "--advantage-estimator grpo --disable-grpo-std-normalization --entropy-coef 0.00 "
        "--eps-clip 0.2 --eps-clip-high 0.28 "
        f"--use-tis --skip-actor-forward-only --tis-clip {cap} "
        + ("--use-rollout-routing-replay " if routing_replay else "")
        + "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--optimizer adam --lr 1e-06 --lr-decay-style cosine --min-lr 1e-07 --lr-warmup-iters 10 --lr-decay-iters 160 "
        "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )


def _layout_args(args: ScriptArgs) -> str:
    out = (
        "--tensor-model-parallel-size 1 --sequence-parallel --pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 --expert-model-parallel-size 8 --expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 32768 "
        "--rollout-num-gpus-per-engine 2 --sglang-mem-fraction-static 0.7 --sglang-max-running-requests 128 "
        "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash "
        "--actor-num-nodes 4 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 --rollout-num-gpus 32 "
        "--pin-rollout-manager-to-head --log-passrate "
    )
    if args.top:
        return out + "--sglang-cuda-graph-max-bs-decode 64 " + " ".join(_top().train_args) + " "
    return out + "--sglang-ep-size 2 "


def _top():
    return top_launch(model_type="qwen3_5_moe", engine_gpus=2)


def execute(args: ScriptArgs):
    """Run one arm of the async text-to-SQL comparison."""
    U = args.create_backend()
    # The grader refuses a snapshot or temp root that other users can read (mode 0700 only).
    U.exec_command_cpu(f"mkdir -p -m 700 {args.sqlite_scratch_dir}/snapshots {args.sqlite_scratch_dir}/tmp")
    U.exec_command_cpu(f"chmod 700 {args.sqlite_scratch_dir}/snapshots {args.sqlite_scratch_dir}/tmp")
    env = {
        "TEXT_TO_SQL_SQLITE_SNAPSHOT_ROOT": f"{args.sqlite_scratch_dir}/snapshots",
        "TMPDIR": f"{args.sqlite_scratch_dir}/tmp",
    }
    if args.top:
        env = _top().env_vars | env
    U.execute_train(
        train_args=f"{_ckpt_args(args)} {_rollout_args(args)} {_algo_optimizer_args(args)} {_layout_args(args)}",
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=env,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
