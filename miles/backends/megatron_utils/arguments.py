import logging
import os

import torch
from megatron.training.arguments import parse_args
from megatron.training.arguments import validate_args as _megatron_validate_args
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding

__all__ = ["validate_args", "parse_args", "set_default_megatron_args"]

logger = logging.getLogger(__name__)


def validate_args(args):
    args = _megatron_validate_args(args)

    optimizer_state_dtypes = (args.main_params_dtype, args.exp_avg_dtype, args.exp_avg_sq_dtype)
    if args.optimizer_cpu_offload and any(dtype != torch.float32 for dtype in optimizer_state_dtypes):
        # HybridDeviceOptimizer currently creates FP32 master parameters and Adam states
        # regardless of the precision-aware optimizer dtype settings.
        raise ValueError(
            "--optimizer-cpu-offload does not honor lower-precision --main-params-dtype, "
            "--exp-avg-dtype, or --exp-avg-sq-dtype; remove these dtype overrides"
        )

    return args


def set_default_megatron_args(args):
    # Muon currently owns its sharding path, and Megatron's distributed optimizer
    # only supports Adam-family optimizers.
    args.use_distributed_optimizer = (args.optimizer is None or args.optimizer.lower() == "adam") and not getattr(
        args, "debug_disable_optimizer", False
    )
    # Multi-LoRA: per-slot LayerWise optimizers require plain DDP all-reduce.
    if getattr(args, "multi_lora_n_adapters", 0) > 0:
        args.use_distributed_optimizer = False
    # TODO: maybe change this after megatron has good fp8 support
    args.bf16 = not args.fp16
    # placeholders
    if args.seq_length is None:
        args.seq_length = 4096
    args.max_position_embeddings = args.seq_length
    # Notice(Jiajun): new megatron has removed this argument and use dp_reshardable instead of fully_shard
    if os.getenv("DEPRECATED_MEGATRON_COMPATIBLE", "0") == "1":
        args.dist_ckpt_save_pre_mcore_014 = True
    # compatible for megatron
    if hasattr(args, "rope_type") and args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"

    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

    if not args.tokenizer_model and not args.tokenizer_type:
        logger.info("--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"

    if not hasattr(args, "miles_dsa_topk_backend"):
        args.miles_dsa_topk_backend = "torch"

    return args
