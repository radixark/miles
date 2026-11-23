"""What the Trainium launchers share: GRPO on one trn3.48xlarge, torchtitan trainer + sglang-neuron rollout.

The host's 16 Neuron devices hold 64 logical NeuronCores (LNC 2). The torchtitan trainer runs one rank per core on
cores 0..num_trainer_cores-1; one sglang-neuron engine serves the policy with TP 4 on the four cores starting at
--engine-first-core and drives as an external rollout engine. Weights reach the engine through disk-delta: the
trainer publishes deltas, the engine's /pull_weights patches its local checkpoint, and update_weights_from_disk
reloads it into the compiled graphs in place. scripts/trainium/README.md describes the environment both expect.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from miles.utils.external_utils import command_utils

ENGINE_TP_SIZE = 4
# NeuronPlatform.weight_transfer_modes; the launcher does not import torch_neuronx to read it
WEIGHT_TRANSFER_MODE = "disk-delta"


@dataclass(frozen=True)
class TitanNeuronModel:
    """What a launcher fixes about its model; the rest is the shared recipe."""

    # the checkpoint's directory under --model-dir
    name: str
    titan_model_name: str
    titan_model_flavor: str
    # the NKI MoE kernels take a multiple of 8192 tokens a call, and every microbatch pads to --seq-length
    seq_length: int
    num_trainer_cores: int
    sglang_args: str = ""
    train_args: str = ""


@dataclass(frozen=True)
class _Recipe:
    num_rollout: int
    rollout_batch_size: int
    n_samples_per_prompt: int
    max_response_len: int
    global_batch_size: int
    enable_eval: bool


_RECIPES: dict[str, _Recipe] = {
    "smoke": _Recipe(3, 8, 4, 2048, 32, enable_eval=False),
    "full": _Recipe(100, 16, 8, 4096, 128, enable_eval=True),
}


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    recipe: Literal["smoke", "full"] = "full"
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    # host-local directory the engine keeps its pulled checkpoint in, and the trainer publishes deltas under
    weight_sync_dir: str = "/root/weight_sync"
    sglang_port: int = 30000
    engine_first_core: int = 32
    # trainer only: miles parses sglang-miles' server args; the engine keeps the image's SGLang
    sglang_miles_pythonpath: str = "/opt/sglang-miles/python"
    # sglang-neuron refuses a NEFF cache it did not stamp, e.g. the trainer's
    engine_compile_cache_dir: str = "/root/.cache/sglang-neuron"
    extra_args: str = ""
    # the pkill sweep would kill the engine, which runs outside ray
    graceful_ray_stop: bool = True


def serve_sglang(args: ScriptArgs, model: TitanNeuronModel) -> None:
    """Serve the policy with TP 4 on cores engine_first_core..+3 in the foreground."""
    U = args.create_backend()
    last_core = args.engine_first_core + ENGINE_TP_SIZE - 1
    U.exec_command_cpu(
        "SGLANG_PLATFORM=neuron NEURON_LOGICAL_NC_CONFIG=2 "
        f"NEURON_RT_VISIBLE_CORES={args.engine_first_core}-{last_core} "
        f"TORCH_NEURONX_NEFF_CACHE_DIR={args.engine_compile_cache_dir}/neff "
        f"TORCH_NEURONX_HLO_CACHE_DIR={args.engine_compile_cache_dir}/hlo "
        # the default sampler is greedy-only
        "SGLANG_NEURON_SAMPLER=temperature "
        "python3 -m sglang.launch_server "
        f"--model-path {args.model_dir}/{model.name} "
        f"--tp-size {ENGINE_TP_SIZE} "
        f"--context-length {model.seq_length} "
        "--chunked-prefill-size 2048 "
        "--max-running-requests 64 "
        # headroom for graphs first loaded mid-run (rare decode buckets); 0.8 hung Qwen3-30B-A3B's engine
        "--mem-fraction-static 0.7 "
        f"{model.sglang_args}"
        f"--host 0.0.0.0 --port {args.sglang_port}"
    )


def prepare_model_and_data(args: ScriptArgs, model_commands: Sequence[str]) -> None:
    """Make the checkpoint with `model_commands` and fetch the datasets; each command skips what is already there."""
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    for command in model_commands:
        U.exec_command_cpu(command)
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


def _trainer_env(args: ScriptArgs, model: TitanNeuronModel) -> dict[str, str]:
    return {
        "MILES_ACCELERATOR": "neuron",
        "NEURON_LOGICAL_NC_CONFIG": "2",
        "NEURON_RT_VISIBLE_CORES": f"0-{model.num_trainer_cores - 1}",
        "PYTHONPATH": args.sglang_miles_pythonpath,
    }


def execute_train(args: ScriptArgs, model: TitanNeuronModel, *, launcher_file: str) -> None:
    """Train on cores 0..num_trainer_cores-1 against the sglang server."""
    U = args.create_backend()
    recipe = _RECIPES[args.recipe]
    # exported rather than passed as extra_env_vars so execute_train skips its nvidia-smi probe
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        # thinking off: a thinking response often runs out of tokens before its answer, so every reward is 0
        """--apply-chat-template-kwargs '{"enable_thinking":false}' """
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {recipe.num_rollout} "
        f"--rollout-batch-size {recipe.rollout_batch_size} "
        f"--n-samples-per-prompt {recipe.n_samples_per_prompt} "
        f"--rollout-max-response-len {recipe.max_response_len} "
        # sglang refuses a prompt that leaves less than the response budget in its context
        f"--rollout-max-prompt-len {model.seq_length - recipe.max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {recipe.global_batch_size} "
    )

    eval_args = ""
    if recipe.enable_eval:
        eval_args = (
            "--eval-interval 10 "
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 4 "
            f"--eval-max-response-len {recipe.max_response_len} "
            "--eval-top-p 1 "
        )

    grpo_args = "--advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        f"--rollout-external-engine-addrs http://localhost:{args.sglang_port} "
        f"--rollout-num-gpus {ENGINE_TP_SIZE} "
        f"--rollout-num-gpus-per-engine {ENGINE_TP_SIZE} "
        f"--update-weight-transfer-mode {WEIGHT_TRANSFER_MODE} "
        f"--update-weight-disk-dir {args.weight_sync_dir}/{args.run_id}/delta "
        # the engine creates and manages this directory; it must not exist yet
        f"--update-weight-local-checkpoint-dir {args.weight_sync_dir}/{args.run_id}/local "
    )

    train_backend_args = (
        "--train-backend torchtitan "
        f"--titan-model-name {model.titan_model_name} "
        f"--titan-model-flavor {model.titan_model_flavor} "
        f"--seq-length {model.seq_length} "
        "--micro-batch-size 1 "
        "--gradient-checkpointing "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {model.seq_length} "
        f"{model.train_args}"
    )

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {model.num_trainer_cores} "
        f"--num-gpus-per-node {model.num_trainer_cores} "
    )

    train_args = (
        f"--hf-checkpoint {args.model_dir}/{model.name} "
        f"{rollout_args} "
        f"{eval_args} "
        f"{grpo_args} "
        f"{optimizer_args} "
        f"{command_utils.get_default_wandb_args(launcher_file, run_id=args.run_id)} "
        f"{sglang_args} "
        f"{train_backend_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=model.num_trainer_cores,
        megatron_model_type=None,
        extra_env_vars=_trainer_env(args, model),
    )
