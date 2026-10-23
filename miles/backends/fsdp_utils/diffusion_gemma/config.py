"""Constraints of the first, text-only fixed-data SFT integration."""

import math


def is_diffusion_gemma(config) -> bool:
    return getattr(config, "model_type", None) == "diffusion_gemma"


def validate_training_args(args) -> None:
    requirements = {
        "loss_type": "sft_loss",
        "debug_train_only": True,
        "qkv_format": "bshd",
        "attn_implementation": "sdpa",
        "kernel_backend": "native",
        "rollout_function_path": "miles.rollout.diffusion_gemma_sft.generate_rollout",
        "n_samples_per_prompt": 1,
        "apply_chat_template": False,
    }
    for name, expected in requirements.items():
        if getattr(args, name, None) != expected:
            raise ValueError(f"DiffusionGemma offline SFT requires --{name.replace('_', '-')} {expected}")
    if getattr(args, "compute_advantages_and_returns", None) is not False:
        raise ValueError("DiffusionGemma offline SFT requires --disable-compute-advantages-and-returns")
    unsupported = (
        "use_dynamic_batch_size",
        "use_dynamic_global_batch_size",
        "use_kl_loss",
        "kl_coef",
        "use_opd",
        "true_on_policy_mode",
        "use_sampling_support_replay",
        "use_routing_replay",
        "eval_num_gpus",
        "use_lora",
        "lora_rank",
    )
    for name in unsupported:
        if getattr(args, name, False):
            raise ValueError(f"DiffusionGemma offline SFT does not support --{name.replace('_', '-')}")
    dp_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    if min(dp_size, args.global_batch_size, args.micro_batch_size, args.rollout_batch_size) < 1:
        raise ValueError("GPU and batch counts must be positive")
    if args.global_batch_size % (dp_size * args.micro_batch_size):
        raise ValueError("global_batch_size must be divisible by DP size * micro_batch_size")
    if args.rollout_batch_size % args.global_batch_size:
        raise ValueError("rollout_batch_size must be divisible by global_batch_size")
    if not 0 < args.diffusion_noise_epsilon <= 1:
        raise ValueError("diffusion_noise_epsilon must be in (0, 1]")
    if not 0 <= args.diffusion_self_conditioning_probability <= 1:
        raise ValueError("diffusion_self_conditioning_probability must be in [0, 1]")
    if not math.isfinite(args.diffusion_encoder_loss_weight) or args.diffusion_encoder_loss_weight < 0:
        raise ValueError("diffusion_encoder_loss_weight must be finite and nonnegative")
