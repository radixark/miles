import logging
import os

from megatron.core.tokenizers.utils.build_tokenizer import vocab_size_with_padding as _vocab_size_with_padding
from megatron.training.arguments import parse_args, validate_args

from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.models.deepseek_v4.arguments import is_dsv4_model, normalize_dsv4_args
from miles_plugins.models.glm5.arguments import normalize_dsa_args

__all__ = ["validate_args", "parse_args", "set_default_megatron_args", "admit_true_on_policy_trainer"]

logger = logging.getLogger(__name__)


def set_default_megatron_args(args):
    if args.true_on_policy_mode or args.true_on_policy_contract or _is_top_spec(args):
        # TOP modules load SGLang's true-on-policy package, which ordinary runs do not require.
        from miles.true_on_policy.contracts import select_true_on_policy

        select_true_on_policy(args)
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

    if is_dsv4_model(args):
        normalize_dsv4_args(args)
    elif _is_top_dsv4(args):
        # The TOP spec builds DS4 / DS4.1 layers itself and shares only Megatron's mHC switch with the plugin spec.
        args.enable_hyper_connections = True
        if args.context_parallel_size > 1 and args.cp_comm_type is None:
            # TOP DS4 attention all-gathers the CP chunks itself; Megatron's local layers then require all_gather.
            args.cp_comm_type = ["all_gather"]
    elif _is_top_spec(args) and args.context_parallel_size > 1 and args.cp_comm_type is None:
        # TOP attention gathers keys over CP itself; the TOP spec's local (non-TE) layers then require all_gather.
        args.cp_comm_type = ["all_gather"]

    if args.dsa_impl == "megatron":
        normalize_dsa_args(args, load_hf_config(args.hf_checkpoint))
    elif args.cp_comm_type is None:
        args.cp_comm_type = ["p2p"]

    return args


def _is_top_spec(args) -> bool:
    spec = args.spec
    return bool(spec) and spec[0] == "miles_plugins.top.spec"


def _refuse_top_gdn_paths(args) -> None:
    """Paths Qwen GDN TOP has no program for: each would fail or train wrong after launch."""
    from miles.backends.megatron_utils.model import _top_deferred_lm_head

    if args.recompute_loss_function and _top_deferred_lm_head(args):
        raise NotImplementedError(
            "[top] --recompute-loss-function checkpoints the loss on tensor inputs, but GDN TOP hands it the "
            "deferred LM head's rows (TopDeferredLogits), so the policy loss would get no gradient; drop the flag"
        )
    if args.megatron_to_hf_mode == "bridge":
        raise NotImplementedError(
            "[top] Qwen GDN TOP converts its local-spec weights with the raw converter only (weight sync and "
            "--save-hf); use --megatron-to-hf-mode raw"
        )


def admit_true_on_policy_trainer(args) -> None:
    """Refuse an unadmitted TOP model or trainer layout before any process starts; get_top_spec re-checks the build."""
    if not args.true_on_policy_mode:
        return
    from miles.true_on_policy.args import refuse_unreplayable_routing
    from miles.true_on_policy.gdn import default_serving_context_length, is_top_gdn, require_gdn_program
    from miles_plugins.top.fp8_config import checkpoint_is_block_fp8
    from miles_plugins.top.registry import TrainerLayout, admit_trainer

    if not args.hf_checkpoint:
        raise ValueError("--true-on-policy-mode needs --hf-checkpoint, whose model_type names the TOP family")
    if args.use_session_server and is_top_gdn(args):
        raise NotImplementedError("[top] no Qwen GDN TOP gate runs session rollouts (--use-session-server) yet")
    if is_top_gdn(args):
        require_gdn_program()
        _refuse_top_gdn_paths(args)
        default_serving_context_length(args)
    family = admit_trainer(
        load_hf_config(args.hf_checkpoint).model_type,
        TrainerLayout.from_args(args),
        fp8=checkpoint_is_block_fp8(args.hf_checkpoint),
        vision=args.multimodal_keys is not None,
    )
    refuse_unreplayable_routing(args, family=family)


def _is_top_dsv4(args) -> bool:
    """A TOP DeepSeek-V4 or V4.1 run: both build their mHC layers on Megatron's hyper-connection switch."""
    if not _is_top_spec(args):
        return False
    model_type = getattr(
        load_hf_config(args.hf_checkpoint), "model_type", None
    )  # config-access-exempt: third-party HF configurations may omit model_type
    return model_type in ("deepseek_v4", "deepseek_v41")
