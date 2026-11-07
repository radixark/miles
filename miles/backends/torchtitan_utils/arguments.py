from dataclasses import dataclass

import torch

from miles.backends.fsdp_utils.arguments import FSDPArgs, build_dataclass_parser, load_args_from_parser
from miles.backends.training_utils.loss.hub.response_layout import validate_response_layout_args
from miles.kernels.tpu import FLASH_ATTENTION_BLOCK
from miles.platforms import current_platform, tpu_shared_slice
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
    # torchtitan's LoRA initializes lora_A kaiming_uniform_(a=sqrt(5)); --lora-type defaults in validate_lora_args
    parser.set_defaults(lora_A_init_method="kaiming")
    return parser


def load_torchtitan_args(extra_args_provider=None):
    return load_args_from_parser(build_torchtitan_parser(extra_args_provider))


def validate_torchtitan_args(args) -> None:
    torch_version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    if torch_version < (2, 14):
        raise ValueError(f"the torchtitan backend needs torch>=2.14; this environment runs {torch.__version__}")
    if args.context_parallel_size != 1 and args.titan_model_name == "qwen3_5":
        raise ValueError("torchtitan does not support context parallelism for qwen3_5")

    if args.max_seq_len is not None:
        if args.seq_length < args.max_seq_len:
            raise ValueError(
                f"--seq-length {args.seq_length} is shorter than "
                f"--max-seq-len {args.max_seq_len}: torchtitan builds its "
                "rotary embeddings for the former, so a longer sequence would index past them"
            )
    elif args.seq_length <= args.rollout_max_response_len:
        raise ValueError(
            f"--seq-length {args.seq_length} leaves no room for a prompt ahead of "
            f"--rollout-max-response-len {args.rollout_max_response_len}: torchtitan builds its "
            "rotary embeddings for the former, and a prompt-plus-response beyond them asserts "
            "inside the rope kernel"
        )

    _validate_platform_args(args)
    if args.update_weight_transfer_mode == "ici" and tpu_shared_slice.spec() is None:
        raise ValueError(
            f"--update-weight-transfer-mode ici needs {tpu_shared_slice.ENV_VAR} on the trainer, naming the slice "
            "it shares with the engine"
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
    if args.use_routing_replay and not args.use_rollout_routing_replay and not current_platform.dynamic_shapes:
        raise ValueError(
            "--use-routing-replay without --use-rollout-routing-replay records the ids inside a compiled block, which "
            "recompiles every microbatch on a static-shape platform. Use --use-rollout-routing-replay."
        )
    if args.ci_test and args.use_routing_replay and not current_platform.dynamic_shapes:
        raise ValueError(
            "--ci-test with --use-routing-replay: the replay checker syncs to the host from inside a block, which a "
            "static-shape platform compiles fullgraph. Drop --ci-test or run the check on a dynamic-shape platform."
        )
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
    validate_response_layout_args(args)


def _validate_lora_args(args) -> None:
    if args.titan_model_name != "qwen3":
        raise ValueError(
            f"torchtitan LoRA is validated on --titan-model-name qwen3 only; {args.titan_model_name} builds "
            "projections whose module paths or HF names the LoRA target mapping does not handle yet"
        )
    if args.lora_type != "lora":
        raise ValueError(
            "torchtitan fuses q/k/v and gate/up and shares one lora_A across each, which is --lora-type lora; "
            f"--lora-type {args.lora_type} gives each projection its own lora_A"
        )
    if args.lora_A_init_method != "kaiming":
        raise ValueError(
            "torchtitan initializes lora_A as --lora-A-init-method kaiming; "
            f"got --lora-A-init-method {args.lora_A_init_method}"
        )
    if args.lora_dropout != 0:
        raise ValueError(f"torchtitan's LoRA has no dropout; got --lora-dropout {args.lora_dropout}")
    if lora_rollout_enabled(args) and args.colocate_memory_peak_device == "gpu":
        raise ValueError(
            "--colocate-memory-peak-device gpu pauses the LoRA grad buffers of the megatron backend, "
            "which the torchtitan backend does not have; use the default cpu"
        )


def _validate_platform_args(args) -> None:
    platform = current_platform
    if args.log_probs_backend == "fused" and not platform.triton_kernels:
        raise ValueError(
            f"--log-probs-backend fused is a Triton kernel; on {platform.device_type} use --log-probs-backend torch"
        )
    # colocate syncs through CUDA IPC, outside the transfer modes
    modes = platform.weight_transfer_modes
    if modes is not None and (args.colocate or args.update_weight_transfer_mode not in modes):
        raise ValueError(
            f"on {platform.device_type} the trainer syncs weights with --update-weight-transfer-mode "
            f"{' or '.join(sorted(modes))} to disaggregated engines; got "
            f"{args.update_weight_transfer_mode!r}{' with --colocate' if args.colocate else ''}"
        )
    if platform.dynamic_shapes:
        return
    # without dynamic shapes, packed microbatches are padded to a power-of-two bucket capped at --seq-length
    if args.seq_length % FLASH_ATTENTION_BLOCK != 0:
        raise ValueError(
            f"--seq-length {args.seq_length} must be a multiple of {FLASH_ATTENTION_BLOCK} on {platform.device_type}: "
            f"the splash attention kernel works in {FLASH_ATTENTION_BLOCK}-token blocks and the largest length bucket "
            "is --seq-length"
        )
    if args.context_parallel_size != 1:
        raise ValueError(
            f"--context-parallel-size {args.context_parallel_size} is not supported on {platform.device_type}: "
            "its length buckets do not align microbatches to the CP shards"
        )
    if args.expert_model_parallel_size != 1:
        raise ValueError(
            f"--expert-model-parallel-size {args.expert_model_parallel_size} is not supported on "
            f"{platform.device_type}: the all-to-all token dispatcher's split sizes depend on the routing, which a "
            "fullgraph static-shape compile cannot trace; keep every expert on each rank"
        )
    if args.use_dynamic_batch_size and args.max_tokens_per_gpu > args.seq_length:
        raise ValueError(
            f"--max-tokens-per-gpu {args.max_tokens_per_gpu} exceeds --seq-length {args.seq_length}, "
            "the largest length bucket"
        )
    if not args.use_dynamic_batch_size and args.micro_batch_size > 1:
        raise ValueError(
            f"--micro-batch-size {args.micro_batch_size} can pack more than --seq-length {args.seq_length} "
            "tokens, the largest length bucket; use --use-dynamic-batch-size with --max-tokens-per-gpu"
        )
