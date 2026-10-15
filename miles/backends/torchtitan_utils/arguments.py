from dataclasses import dataclass

import torch

from miles.backends.fsdp_utils.arguments import FSDPArgs, build_dataclass_parser, load_args_from_parser
from miles.utils.lora.utils import is_lora_enabled, lora_rollout_enabled


@dataclass
class TorchtitanArgs(FSDPArgs):
    # torchtitan.models.<name>.model_registry(<flavor>)
    titan_model_name: str = "qwen3"
    titan_model_flavor: str = "0.6B"

    seq_length: int = 4096

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1


def build_torchtitan_parser(extra_args_provider=None):
    parser = build_dataclass_parser(TorchtitanArgs, "torchtitan Training (miles)", extra_args_provider)
    # torchtitan's LoRA gives each projection its own lora_A, initialized kaiming_uniform_(a=sqrt(5))
    parser.set_defaults(lora_type="canonical_lora", lora_A_init_method="kaiming")
    return parser


def load_torchtitan_args(extra_args_provider=None):
    return load_args_from_parser(build_torchtitan_parser(extra_args_provider))


def validate_torchtitan_args(args) -> None:
    torch_version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    if torch_version < (2, 13):
        raise ValueError(f"the torchtitan backend needs torch>=2.13; this environment runs {torch.__version__}")
    if args.context_parallel_size != 1 and args.titan_model_name == "qwen3_5":
        raise ValueError("torchtitan does not support context parallelism for qwen3_5")

    if args.rollout_max_context_len is not None:
        if args.seq_length < args.rollout_max_context_len:
            raise ValueError(
                f"--seq-length {args.seq_length} is shorter than "
                f"--rollout-max-context-len {args.rollout_max_context_len}: torchtitan builds its "
                "rotary embeddings for the former, so a longer sequence would index past them"
            )
    elif args.seq_length <= args.rollout_max_response_len:
        raise ValueError(
            f"--seq-length {args.seq_length} leaves no room for a prompt ahead of "
            f"--rollout-max-response-len {args.rollout_max_response_len}: torchtitan builds its "
            "rotary embeddings for the former, and a prompt-plus-response beyond them asserts "
            "inside the rope kernel"
        )

    if args.ref_update_interval is not None:
        raise ValueError("--ref-update-interval is not supported by the torchtitan backend")
    if args.fp16:
        raise ValueError("the torchtitan backend trains in bf16 mixed precision; --fp16 is not supported")
    if args.lr_decay_style not in ("constant", "linear", "cosine"):
        raise ValueError(
            f"torchtitan's LR schedule has no {args.lr_decay_style!r} decay; use constant, linear or cosine"
        )
    if args.lr_warmup_fraction is not None or args.lr_wsd_decay_iters is not None or args.lr_decay_iters is not None:
        raise ValueError(
            "torchtitan's LR schedule takes --lr-warmup-iters only; fraction, WSD and decay-iters are unsupported"
        )
    if args.save_debug_train_data is not None:
        raise ValueError("--save-debug-train-data is not wired up for the torchtitan backend")
    if args.stream_optimizer_state_to_disk:
        if not args.optimizer_cpu_offload:
            raise ValueError(
                "--stream-optimizer-state-to-disk on torchtitan backs the host-resident Adam moments; "
                "it needs --optimizer-cpu-offload"
            )
        if args.stream_optimizer_state_moment_dtype != "fp32":
            raise ValueError(
                "torchtitan keeps the moments in the dtype AdamW steps them in; "
                f"--stream-optimizer-state-moment-dtype {args.stream_optimizer_state_moment_dtype} is unsupported"
            )
    if is_lora_enabled(args):
        _validate_lora_args(args)


def _validate_lora_args(args) -> None:
    if args.titan_model_name != "qwen3":
        raise ValueError(
            f"torchtitan LoRA is validated on --titan-model-name qwen3 only; {args.titan_model_name} builds "
            "projections whose module paths or HF names the LoRA target mapping does not handle yet"
        )
    if args.lora_type != "canonical_lora":
        raise ValueError(
            "torchtitan gives q/k/v and gate/up their own lora_A, which is --lora-type canonical_lora; "
            f"--lora-type {args.lora_type} shares one lora_A across each fused projection on megatron"
        )
    if args.lora_A_init_method != "kaiming":
        raise ValueError(
            "torchtitan initializes lora_A as --lora-A-init-method kaiming; "
            f"got --lora-A-init-method {args.lora_A_init_method}"
        )
    if args.lora_dropout != 0:
        raise ValueError(f"torchtitan's LoRA has no dropout; got --lora-dropout {args.lora_dropout}")
    if args.lora_adapter_path is not None:
        raise ValueError("--lora-adapter-path is not wired up for the torchtitan backend; resume with --load")
    if lora_rollout_enabled(args) and args.colocate_memory_peak_device == "gpu":
        raise ValueError(
            "--colocate-memory-peak-device gpu pauses the LoRA grad buffers of the megatron backend, "
            "which the torchtitan backend does not have; use the default cpu"
        )
