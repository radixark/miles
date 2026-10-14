"""FlashREINFORCE on Qwen2.5-Math-1.5B: critic-free, one rollout per prompt, fully async.

Follows the paper's Qwen2.5-Math-1.5B setting (Tab. 13): 128 prompts x 1 rollout per
update, one optimizer step per batch, 4k responses, lr 1e-6, trust region delta 3e-3.
One 8-GPU node: 2 training GPUs and 6 single-GPU rollout engines.

`prepare` downloads the model and DAPO-Math-17k/AIME-2024 and converts the checkpoint
to Megatron torch_dist; `execute` submits the run.

Args:
    --num-train-gpus: GPUs given to Megatron; the rest of the node serves rollout.
    --extra-args: appended last, so any flag here can be overridden, e.g.
        "--tis-binary-kl-threshold inf" for importance-weighted PG without the trust region.

Example:
    python examples/flash_reinforce/run_qwen2_5_math_1_5b_flash_reinforce.py
"""

from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    model_name: str = "Qwen2.5-Math-1.5B"
    megatron_model_type: str = "qwen2.5-1.5B"
    num_gpus_per_node: int = 8
    num_train_gpus: int = 2
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    extra_args: str = ""


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    U = args.create_backend()
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 100 "
    )

    rollout_args = (
        "--fully-async "
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        # 0/1 reward on the last \\boxed{} answer, as in the paper. A batch where every rollout scores
        # the same has zero advantage, so a strict format check that fails everything never starts learning.
        "--rm-type math "
        "--num-rollout 2000 "
        # 128 single-rollout prompts per update, consumed by exactly one optimizer step.
        "--rollout-batch-size 128 "
        "--n-samples-per-prompt 1 "
        "--num-steps-per-rollout 1 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 1.0 "
        "--rollout-top-p 1.0 "
        # 512 trajectories in flight against 128 per update: a policy lag of about four updates.
        "--async-max-concurrent-samples 512 "
        # In-flight requests continue on the new weights; the per-token IS weights correct the mix.
        "--pause-generation-mode in_place "
        "--balance-data "
    )

    eval_args = (
        "--eval-interval 50 "
        f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 16 "
        "--eval-max-response-len 4096 "
        "--eval-temperature 0.7 "
        "--eval-top-p 0.7 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    flash_reinforce_args = (
        # A = R - mean(R) over the batch's rollouts: no prompt groups, no std, no whitening.
        "--advantage-estimator flash_reinforce "
        # The training forward doubles as the old policy, so the PPO ratio is exactly 1 and
        # the surrogate is plain REINFORCE.
        "--skip-actor-forward-only "
        # Unclipped token IS against the sampler's log-probs, gated per sequence by the mean
        # sampled-token binary KL.
        "--use-tis "
        "--custom-tis-function-path miles.backends.training_utils.loss.hub.corrections.binary_kl_trust_region_function "
        "--tis-binary-kl-threshold 3e-3 "
        # The default per-sample loss is the paper's sample-mean reduction; keep
        # --calculate-per-token-loss off.
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--clip-grad 1.0 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.8 "
        # The paper's 2k prompt + 4k response budget exceeds the 4k max_position_embeddings in the
        # model config; SGLang only accepts the longer context with SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.
        "--sglang-context-length 6144 "
    )

    misc_args = (
        # --skip-actor-forward-only requires deterministic forwards.
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_train_gpus} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.num_gpus_per_node - args.num_train_gpus} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{eval_args} "
        f"{optimizer_args} "
        f"{flash_reinforce_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        train_script="train_async.py",
        megatron_path=args.megatron_path,
        extra_env_vars={"PYTHONPATH": args.megatron_path, "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"},
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
