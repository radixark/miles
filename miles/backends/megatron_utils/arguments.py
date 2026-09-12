import logging
import os

from megatron.core.tokenizers.utils.build_tokenizer import vocab_size_with_padding as _vocab_size_with_padding
from megatron.training.arguments import parse_args, validate_args

from miles.utils.hf_config import load_hf_config
from miles_plugins.models.deepseek_v4.arguments import apply_dsv4_model_impl

__all__ = ["validate_args", "parse_args", "set_default_megatron_args"]

logger = logging.getLogger(__name__)


def set_default_megatron_args(args):
    if getattr(args, "true_on_policy_mode", False):
        raise NotImplementedError(
            "--true-on-policy-mode is not supported on the megatron backend with this Megatron "
            "version; support lands in a follow-up PR. Use --train-backend fsdp for true-on-policy."
        )
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
    # Before 20260819, radixark/Megatron-LM pick torch gemm for router, which is fp32 x fp32 ->
    # fp32, and 20260819 convert the default to TE gemm which is bf16 x bf16 -> fp32. Result show
    # the TE one increase log prob diff so manually set back
    args.moe_router_use_torch_mm = True
    # compatible for megatron
    if hasattr(args, "rope_type") and args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"

    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

    if not args.tokenizer_model and not args.tokenizer_type:
        logger.info("--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"

    args.trust_remote_code = True

    if not hasattr(args, "miles_dsa_topk_backend"):
        args.miles_dsa_topk_backend = "torch"

    hf_config = load_hf_config(args.hf_checkpoint) if args.hf_checkpoint else None
    args.model_family = getattr(hf_config, "model_type", None)
    apply_model_impl(args)

    return args


def apply_model_impl(args):
    # the conversion and replay tools parse without the serving flags
    if getattr(args, "megatron_to_hf_mode", "raw") == "bridge" and args.model_impl != "megatron":
        raise ValueError(
            "--model-impl miles cannot run under --megatron-to-hf-mode bridge: "
            "Megatron-Bridge builds megatron-native modules"
        )
    if args.model_family == "deepseek_v4":
        apply_dsv4_model_impl(args)
    elif args.model_impl != "megatron":
        raise ValueError(
            f"--model-impl miles: {args.model_family or 'this model'} has only the megatron implementation"
        )
