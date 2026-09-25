"""Train Qwen3-1.7B with example-local privileged-context forward KL.

Requires Megatron, SGLang, math_verify, and a single node with at least four GPUs.
Preparation downloads data and converts the base checkpoint before training.

Args:
    model_dir: Download and converted-checkpoint directory.
    data_dir: Download and rendered-dataset directory.
    output_dir: Training checkpoint and teacher-log directory.
    num_gpus_per_node: Two student GPUs, one teacher GPU, and the rest for rollout.
    megatron_path: Megatron checkout.
    teacher_port: Local frozen-teacher HTTP port.
    extra_args: Additional train.py arguments.

Example:
    python -m examples.on_policy_distillation.qwen3_1_7b_opsd.run_qwen3_1_7b_opsd
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    num_gpus_per_node: int = 8
    megatron_path: str = "/root/Megatron-LM"
    teacher_port: int = 13141
    extra_args: str = ""

    def __post_init__(self):
        if self.num_nodes != 1 or self.num_gpus_per_node < 4:
            raise ValueError("OPSD requires one node with at least four GPUs.")
        if U.get_bool_env_var("MILES_SCRIPT_EXTERNAL_RAY") or any(
            os.environ.get(key) for key in ("RAY_ADDRESS", "CUDA_VISIBLE_DEVICES")
        ):
            raise ValueError("Use a dedicated node with no external Ray cluster or CUDA_VISIBLE_DEVICES override.")


def prepare(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {quote(args.model_dir)} {quote(args.data_dir)}")
    U.exec_command_cpu(f"hf download Qwen/Qwen3-1.7B --local-dir {quote(args.model_dir + '/Qwen3-1.7B')}")
    U.hf_download_dataset("open-r1/OpenThoughts-114k-math", data_dir=args.data_dir)
    U.hf_download_dataset("HuggingFaceH4/aime_2024", data_dir=args.data_dir)
    U.convert_checkpoint(
        model_name="Qwen3-1.7B",
        megatron_model_type="qwen3-1.7B",
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/Qwen3-1.7B",
        megatron_path=args.megatron_path,
    )
    U.exec_command_cpu(
        "python -m examples.on_policy_distillation.qwen3_1_7b_opsd.prepare_data "
        + " ".join(
            quote(path)
            for path in (
                f"{args.model_dir}/Qwen3-1.7B",
                f"{args.data_dir}/OpenThoughts-114k-math",
                f"{args.data_dir}/aime_2024",
                f"{args.data_dir}/opsd-train.jsonl",
                f"{args.data_dir}/opsd-aime24.jsonl",
            )
        )
    )


def execute(args: ScriptArgs):
    module = "examples.on_policy_distillation.qwen3_1_7b_opsd.opsd"
    checkpoint = f"{args.output_dir}/{args.run_id}/checkpoints"
    config = Path(__file__).with_name("config.yaml")
    ckpt_args = (
        f"--hf-checkpoint {quote(args.model_dir + '/Qwen3-1.7B')} "
        f"--ref-load {quote(args.model_dir + '/Qwen3-1.7B_torch_dist')} "
        f"--load {quote(checkpoint)} --save {quote(checkpoint)} --save-interval 25 "
    )
    rollout_args = (
        f"--prompt-data {quote(args.data_dir + '/opsd-train.jsonl')} --input-key prompt --metadata-key metadata "
        "--rollout-shuffle --num-rollout 100 --rollout-batch-size 32 --n-samples-per-prompt 1 "
        "--rollout-max-response-len 1024 --rollout-temperature 1.1 --rollout-top-k 20 "
        "--global-batch-size 32 --balance-data "
        f"--eval-prompt-data aime24 {quote(args.data_dir + '/opsd-aime24.jsonl')} "
        "--eval-interval 25 --eval-label-key label --n-samples-per-eval-prompt 12 "
        "--eval-max-response-len 38912 --eval-temperature 1.0 --eval-top-p 0.95 --eval-top-k -1 "
    )
    algorithm_args = (
        f"--custom-rm-path {module}.reward_func --rm-url http://127.0.0.1:{args.teacher_port}/generate "
        f"--custom-convert-samples-to-train-data-path {module}.convert_samples "
        f"--loss-type custom_loss --custom-loss-function-path {module}.loss_function "
        f"--custom-config-path {quote(str(config))} --disable-compute-advantages-and-returns "
        "--object-store-backend ray --lora-rank 64 --lora-alpha 128 --lora-dropout 0.0 "
        "--target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --megatron-to-hf-mode bridge "
    )
    perf_args = (
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 16384 "
    )
    optimizer_args = (
        "--optimizer adam --lr 5e-6 --lr-decay-style constant --clip-grad 0.1 "
        "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )
    misc_args = (
        f"--actor-num-nodes 1 --actor-num-gpus-per-node 2 --num-gpus-per-node {args.num_gpus_per_node - 1} "
        f"--rollout-num-gpus {args.num_gpus_per_node - 3} --rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.7 --attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash "
    )
    teacher_pid = None

    def start_teacher():
        nonlocal teacher_pid
        U.exec_command_cpu(f"mkdir -p {quote(args.output_dir + '/' + args.run_id)}")
        teacher_pid = int(
            U.exec_command_cpu(
                f"CUDA_VISIBLE_DEVICES={args.num_gpus_per_node - 1} python -m sglang.launch_server "
                f"--model-path {quote(args.model_dir + '/Qwen3-1.7B')} --host 127.0.0.1 --port {args.teacher_port} "
                "--tp 1 --chunked-prefill-size 4096 --mem-fraction-static 0.25 "
                f"> {quote(args.output_dir + '/' + args.run_id + '/teacher.log')} 2>&1 & echo $!",
                capture_output=True,
            )
        )
        if teacher_pid <= 0:
            raise ValueError("Invalid teacher process ID.")
        U.exec_command_cpu(
            f"for attempt in $(seq 1 120); do kill -0 {teacher_pid} || exit 1; "
            f"curl -sf http://127.0.0.1:{args.teacher_port}/health_generate >/dev/null && exit 0; "
            "sleep 5; done; exit 1"
        )

    try:
        U.execute_train(
            train_args=ckpt_args
            + rollout_args
            + algorithm_args
            + perf_args
            + optimizer_args
            + misc_args
            + U.get_default_wandb_args(__file__, run_id=args.run_id)
            + " "
            + args.extra_args,
            num_gpus_per_node=args.num_gpus_per_node - 1,
            megatron_model_type="qwen3-1.7B",
            megatron_path=args.megatron_path,
            config=args,
            before_ray_job_submit=start_teacher,
        )
    finally:
        if teacher_pid is not None and teacher_pid > 0:
            U.exec_command_cpu(f"kill {teacher_pid} 2>/dev/null || true")


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
