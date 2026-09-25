"""Nemotron 3.5 Lightning Workplace Assistant GRPO with separate rollout and training nodes.

Requires two joined eight-GPU Ray nodes, matching model and dataset paths on both,
and the NeMo Gym Workplace Assistant verifier. The model starts from original HF
weights. The native resource service owns tool state and binary reward; all policy
turns use Miles TITO sessions.

Args:
    --config: JSON object containing the ScriptArgs fields below.

Example:
    python examples/experimental/nemo-gym-workspace-assistant/run_nemotron35_workplace.py \
        --config /path/to/launcher_config.json
"""

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from tap import Tap

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    num_nodes: int = 2
    num_gpus_per_node: int = 8
    model_dir: str = "/root/models"
    model_name: str = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    data_dir: str = "/root/datasets"
    data_file: str = "workplace_train.jsonl"
    megatron_path: str = "/root/Megatron-LM"
    verifier_url: str = "http://127.0.0.1:8211"
    hf_cache_dir: str = "/root/.cache/huggingface"
    learning_rate: float = 3e-7
    rollout_batch_size: int = 8
    group_size: int = 16
    global_batch_size: int = 128
    num_rollout: int = 1000
    save_interval: int = 100
    response_length: int = 65536
    context_length: int = 81920
    max_tokens_per_gpu: int = 16384
    pause_generation_mode: str = "retract"

    def __post_init__(self) -> None:
        if (self.num_nodes, self.num_gpus_per_node) != (2, 8):
            raise ValueError("This example requires two nodes with eight GPUs each")
        if self.global_batch_size != self.rollout_batch_size * self.group_size:
            raise ValueError("global_batch_size must equal rollout_batch_size * group_size")
        if not 0 < self.response_length < self.context_length:
            raise ValueError("response_length must be positive and smaller than context_length")
        if (
            min(
                self.global_batch_size,
                self.rollout_batch_size,
                self.group_size,
                self.num_rollout,
                self.save_interval,
                self.max_tokens_per_gpu,
            )
            <= 0
        ):
            raise ValueError("Batch sizes, update counts and token budgets must be positive")
        if self.pause_generation_mode != "retract":
            raise ValueError("This routing-replay recipe requires pause_generation_mode=retract")

    @property
    def run_name(self) -> str:
        return f"{self.run_id}-nemotron35-lightning-workplace-async-2n-bs{self.global_batch_size}-g{self.group_size}-{self.pause_generation_mode}"


def _flags(values: dict[str, object]) -> str:
    tokens = []
    for key, value in values.items():
        if value is False or value is None:
            continue
        tokens.append("--" + key.replace("_", "-"))
        if value is not True:
            tokens.append(str(value))
    return shlex.join(tokens)


def _learning_args(args: ScriptArgs) -> str:
    performance_args = _flags(
        {
            "tensor_model_parallel_size": 2,
            "sequence_parallel": True,
            "pipeline_model_parallel_size": 2,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 2,
            "expert_tensor_parallel_size": 1,
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
            "use_dynamic_batch_size": True,
            "max_tokens_per_gpu": args.max_tokens_per_gpu,
            "log_probs_chunk_size": 128,
            "seq_length": args.context_length,
            "optimizer_cpu_offload": True,
            "overlap_cpu_optimizer_d2h_h2d": True,
            "use_precision_aware_optimizer": True,
        }
    )
    algorithm_args = _flags(
        {
            "advantage_estimator": "grpo",
            "use_kl_loss": True,
            "kl_loss_coef": 0,
            "kl_loss_type": "low_var_kl",
            "entropy_coef": 0,
            "eps_clip": 0.2,
            "eps_clip_high": 0.28,
            "use_rollout_logprobs": True,
            "pause_generation_mode": args.pause_generation_mode,
        }
    )
    optimizer_args = _flags(
        {
            "optimizer": "adam",
            "lr": args.learning_rate,
            "lr_decay_style": "constant",
            "weight_decay": 0.1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.98,
        }
    )
    return " ".join([performance_args, algorithm_args, optimizer_args])


def _checkpoint_args(args: ScriptArgs) -> str:
    checkpoint = str(Path(args.model_dir) / args.model_name)
    return _flags(
        {
            "hf_checkpoint": checkpoint,
            "ref_load": checkpoint,
            "save": str(Path(args.output_dir) / "checkpoints"),
            "save_interval": args.save_interval,
            "megatron_to_hf_mode": "bridge",
            "no_load_optim": True,
            "no_load_rng": True,
            "finetune": True,
        }
    )


def _rollout_args(args: ScriptArgs) -> str:
    return _flags(
        {
            "prompt_data": str(Path(args.data_dir) / args.data_file),
            "input_key": "prompt",
            "metadata_key": "metadata",
            "label_key": "label",
            "apply_chat_template": False,
            "custom_generate_function_path": "workplace_generate.generate",
            "custom_agent_function_path": "workplace_agent.run",
            "max_seq_len": args.context_length,
            "dynamic_sampling_filter_path": "miles.rollout.filter_hub.common_filters.apply_aborted_filter",
            "rollout_shuffle": True,
            "rm_type": "deepscaler",
            "custom_rm_path": "workplace_generate.reward_func",
            "num_rollout": args.num_rollout,
            "rollout_batch_size": args.rollout_batch_size,
            "n_samples_per_prompt": args.group_size,
            "global_batch_size": args.global_batch_size,
            "rollout_max_response_len": args.response_length,
            "rollout_temperature": 1,
            "balance_data": True,
        }
    )


def _serving_args(args: ScriptArgs) -> str:
    return _flags(
        {
            "rollout_num_gpus_per_engine": 1,
            "sglang_mem_fraction_static": 0.7,
            "sglang_context_length": args.context_length,
            "use_rollout_routing_replay": True,
            "sglang_router_policy": "round_robin",
            "sglang_enable_metrics": True,
            "use_session_server": "v1",
            "session_server_workers": 8,
            "pin_rollout_manager_to_head": True,
            "session_server_ip": "0.0.0.0",
            "session_server_port": 31000,
            "tito_model": "nemotron3",
            "sglang_reasoning_parser": "nemotron_3",
            "sglang_tool_call_parser": "qwen3_coder",
        }
    )


def _miscellaneous_args(args: ScriptArgs) -> str:
    return _flags(
        {
            "attention_dropout": 0,
            "hidden_dropout": 0,
            "accumulate_allreduce_grads_in_fp32": True,
            "attention_softmax_in_fp32": True,
            "attention_backend": "auto",
            "actor_num_nodes": 1,
            "actor_num_gpus_per_node": args.num_gpus_per_node,
            "num_gpus_per_node": args.num_gpus_per_node,
            "rollout_num_gpus": args.num_gpus_per_node,
            "mtp_loss_scaling_factor": 0,
            "custom_megatron_before_train_step_hook_path": "workplace_training_checks.before_train_step",
            "dump_details": str(Path(args.output_dir) / "traces"),
            "use_miles_dashboard": True,
            "observe_training_entropy": True,
            "use_rollout_entropy": True,
            "use_prometheus": True,
            "prometheus_port": 9090,
            "dashboard_forward_prometheus": True,
        }
    )


def execute(args: ScriptArgs) -> None:
    U.execute_train(
        train_args=" ".join(
            [
                _checkpoint_args(args),
                _rollout_args(args),
                _learning_args(args),
                _serving_args(args),
                _miscellaneous_args(args),
                U.get_default_wandb_args(__file__, run_id=args.run_name),
            ]
        ),
        config=args,
        train_script="train_async.py",
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="nemotron-3-nano-30b-a3b",
        megatron_path=args.megatron_path,
        extra_env_vars={
            "MILES_NEMOTRONH_KEEP_MTP": "",
            "WORKPLACE_RESOURCE_URL": args.verifier_url,
            "HF_HOME": args.hf_cache_dir,
            "HUGGINGFACE_HUB_CACHE": str(Path(args.hf_cache_dir) / "hub"),
            "PYTHONPATH": str(Path(U.repo_base_dir) / "examples/experimental/nemo-gym-workspace-assistant"),
        },
    )


class _CLI(Tap):
    config: Path


def main() -> None:
    cli = _CLI().parse_args()
    execute(ScriptArgs(**json.loads(cli.config.read_text())))


if __name__ == "__main__":
    main()
