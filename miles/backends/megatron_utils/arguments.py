import logging
import os

from megatron.core.tokenizers.utils.build_tokenizer import vocab_size_with_padding as _vocab_size_with_padding
from megatron.training.arguments import parse_args, validate_args

from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.models.deepseek_v4.arguments import DSV4_SPEC_MODULE, is_dsv4_model, normalize_dsv4_args
from miles_plugins.models.glm5.arguments import MILES_DSA_SPEC, normalize_dsa_args

__all__ = ["validate_args", "parse_args", "set_default_megatron_args"]

logger = logging.getLogger(__name__)


def set_default_megatron_args(args):
    if args.true_on_policy_mode:
        raise NotImplementedError(
            "--true-on-policy-mode is not supported on the megatron backend with this Megatron "
            "version; support lands in a follow-up PR. Use --train-backend fsdp for true-on-policy."
        )
    # Muon currently owns its sharding path, and Megatron's distributed optimizer
    # only supports Adam-family optimizers.
    args.use_distributed_optimizer = (
        args.optimizer is None or args.optimizer.lower() == "adam"
    ) and not args.debug_disable_optimizer
    # Multi-LoRA: per-slot LayerWise optimizers require plain DDP all-reduce.
    if args.multi_lora_n_adapters > 0:
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
    if args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"

    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

    if not args.tokenizer_model and not args.tokenizer_type:
        logger.info("--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"

    args.trust_remote_code = True

    _validate_miles_dsa_kernel_backend(args)

    if is_dsv4_model(args):
        normalize_dsv4_args(args)

    if args.dsa_impl == "megatron":
        normalize_dsa_args(args, load_hf_config(args.hf_checkpoint))
    elif args.cp_comm_type is None:
        args.cp_comm_type = ["p2p"]

    return args


def _validate_miles_dsa_kernel_backend(args) -> None:
    """--miles-dsa-kernel-backend cake only reaches the kernels through the miles plugin specs; fail early
    where it would otherwise be ignored (bridge-built models, native Megatron DSA / DeepSeek-V4, other specs)."""
    if args.miles_dsa_kernel_backend != "cake":
        return
    if args.megatron_to_hf_mode != "raw" or args.multi_lora_n_adapters > 0:
        raise ValueError(
            "--miles-dsa-kernel-backend cake requires --megatron-to-hf-mode raw without multi-LoRA: those runs "
            "build the model from the Megatron-Bridge provider, which does not use the miles DSA plugin kernels."
        )
    spec_module = (args.spec or [None])[0]
    if spec_module == MILES_DSA_SPEC[0] and args.dsa_impl == "miles":
        return
    if spec_module == DSV4_SPEC_MODULE and args.dsv4_impl == "miles":
        return
    raise ValueError(
        "--miles-dsa-kernel-backend cake is only wired into the miles DSA plugin specs: "
        f"--spec {' '.join(MILES_DSA_SPEC)} with --dsa-impl miles, or --spec {DSV4_SPEC_MODULE} get_dsv4_spec "
        f"with --dsv4-impl miles; got --spec {args.spec}, --dsa-impl {args.dsa_impl}."
    )
