import logging
from typing import Any

from miles.utils.hf_utils.config import load_hf_config

logger = logging.getLogger(__name__)


def add_lora_arguments(parser):
    """Add LoRA-related arguments for Megatron backend."""
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=0,
        help="LoRA rank. Set to 0 to disable LoRA (default: 0)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha for scaling (default: 16)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout rate (default: 0.0)",
    )
    parser.add_argument(
        "--lora-type",
        type=str,
        default="lora",
        choices=["lora", "canonical_lora"],
        help="LoRA variant to use: 'lora' (standard) or 'canonical_lora' (split Q/K/V) (default: lora)",
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        default=None,
        help="Target modules for LoRA. Use 'all-linear' or comma-separated module names "
        "(e.g., 'q_proj,k_proj,v_proj,o_proj' for HF naming or 'linear_qkv,linear_proj' for Megatron naming)",
    )
    parser.add_argument(
        "--exclude-modules",
        type=str,
        default=None,
        help="Modules to exclude from LoRA (comma-separated)",
    )
    parser.add_argument(
        "--lora-adapter-path",
        type=str,
        default=None,
        help="Path to load pre-trained LoRA adapter weights (default: None)",
    )
    parser.add_argument(
        "--lora-sync-from-tensor",
        action="store_true",
        default=False,
        help="Sync LoRA weights via tensor instead of file (more efficient)",
    )
    parser.add_argument(
        "--lora-base-cpu-backup",
        action="store_true",
        default=False,
        help=(
            "LoRA + colocate: keep SGLang-side CPU mirror of base weights "
            "and skip per-step base sync. Trades host RAM for faster "
            "onload/offload. Ignored unless --colocate and LoRA are both on. "
            "Also needs 'weight' in --offload-rollout-level: SGLang populates "
            "the mirror during release_weights_occupation, so with the weights "
            "never released the mirror is never built and the flag does nothing."
        ),
    )
    parser.add_argument(
        "--lora-train-only",
        action="store_true",
        default=False,
        help=(
            "Train LoRA adapters in Megatron but keep rollout engines on the frozen "
            "base policy: SGLang LoRA serving and adapter weight sync are disabled "
            "(only the base weights are synced). For models without SGLang LoRA "
            "support (e.g. Inkling native LoRA)."
        ),
    )
    parser.add_argument(
        "--experts-shared-outer-loras",
        action="store_true",
        default=False,
        help="Enable shared-outer grouped-expert LoRA (gate_up lora_A and "
        "down lora_B shared across experts, expert_dim=1). Matches SGLang "
        "PR #21466's experts_shared_outer_loras=True serving contract.",
    )
    parser.add_argument(
        "--multi-lora-n-adapters",
        type=int,
        default=0,
        help="Maximum number of concurrent adapter slots for multi-LoRA. Set to 0 to disable multi-LoRA (default: 0)",
    )
    parser.add_argument(
        "--check-lora-weight-equal",
        action="store_true",
        default=False,
        help=(
            "Verify the megatron->sglang LoRA adapter weight-sync on the colocated "
            "(from_tensors) path: on every sync the trainer ships a per-tensor sha256 "
            "manifest of the adapter it sends, and each rollout engine hashes the "
            "tensors it received and fails the load on any mismatch/missing/extra "
            "name. The LoRA analogue of --check-weight-update-equal, which only "
            "covers base weights."
        ),
    )
    # required whenever expert projections are LoRA targets, inert otherwise
    # (sglang's own default is False)
    parser.set_defaults(sglang_lora_use_virtual_experts=True)
    parser.add_argument(
        "--no-sglang-lora-use-virtual-experts",
        dest="sglang_lora_use_virtual_experts",
        action="store_false",
        help="Serve MoE-expert LoRA through sglang's fused_moe_lora alignment path instead "
        "of the virtual-experts path.",
    )
    return parser


def validate_lora_args(args):
    # Parse LoRA target modules
    if args.lora_rank > 0:
        assert args.target_modules is not None, "'--target-modules' is required when LoRA is enabled."

        if args.target_modules == "all-linear":
            # MLA projections are HF-config-gated (SGLang sizes LoRA buffers per module name;
            # listing them on a dense model crashes the engine). The DSA indexer stays excluded.
            modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            hf_config = load_hf_config(args.hf_checkpoint)
            if getattr(hf_config, "kv_lora_rank", None):
                modules += ["kv_a_proj_with_mqa", "kv_b_proj"]
                if getattr(hf_config, "q_lora_rank", None):
                    modules += ["q_a_proj", "q_b_proj"]
        elif "," in args.target_modules:
            modules = [m.strip() for m in args.target_modules.split(",")]
        else:
            modules = [args.target_modules]

        if args.exclude_modules:
            exclude_set = (
                set(m.strip() for m in args.exclude_modules.split(","))
                if "," in args.exclude_modules
                else {args.exclude_modules}
            )
            modules = [m for m in modules if m not in exclude_set]

        args.target_modules = modules

        # Training and serving must agree on shared-outer grouped-expert LoRA
        # (expert_dim=1 buffers in SGLang).
        if args.experts_shared_outer_loras and hasattr(args, "sglang_experts_shared_outer_loras"):
            args.sglang_experts_shared_outer_loras = True
        assert args.experts_shared_outer_loras == bool(
            getattr(args, "sglang_experts_shared_outer_loras", args.experts_shared_outer_loras)
        ), "experts_shared_outer_loras and sglang_experts_shared_outer_loras must agree"

        # the two MoE-expert adapter layouts are not checkpoint-compatible; say which one runs
        _expert_leaves = ("linear_fc1", "linear_fc2", "gate_proj", "up_proj", "down_proj")
        if any(leaf in str(tm) for tm in modules for leaf in _expert_leaves):
            logger.warning(
                "MoE-expert LoRA layout: %s (--experts-shared-outer-loras).",
                "shared-outer" if args.experts_shared_outer_loras else "per-expert",
            )

    # Sets args.multi_lora, then validates/defaults the multi-LoRA arg surface
    # (adapter configs themselves are loaded later by the controller).
    validate_multi_lora_args(args)


def validate_multi_lora_args(args: Any) -> None:
    """Set ``args.multi_lora``, then validate the trainer-side constraints of
    the slot machinery. A no-op for normal runs."""
    args.multi_lora = getattr(args, "multi_lora_n_adapters", 0) > 0
    if not args.multi_lora:
        return

    assert args.lora_rank > 0, "--lora-rank must be set when --multi-lora-n-adapters > 0"
    assert args.target_modules is not None, "--target-modules must be set when --multi-lora-n-adapters > 0"
    assert args.train_backend == "megatron", "Multi-LoRA currently requires --train-backend megatron"
    # Adapter routing is only recompute-safe without pipelining; enforce at launch.
    assert getattr(args, "context_parallel_size", 1) == 1, (
        "multi-LoRA requires --context-parallel-size 1: the Tinker losses zip "
        "full-length per-datum vectors against log_probs, which CP would shard"
    )
    assert getattr(args, "pipeline_model_parallel_size", 1) == 1, (
        "Multi-LoRA requires --pipeline-model-parallel-size 1: a pipelined schedule would "
        "recompute activations against a later micro-batch's adapter routing."
    )
    # Per-slot token spans assume sequence-major contiguous sample packing, which only 'thd' provides.
    assert getattr(args, "qkv_format", "thd") == "thd", (
        "Multi-LoRA requires --qkv-format thd: per-adapter token spans assume the "
        f"micro-batch packs samples contiguously, which bshd does not (got {args.qkv_format!r})."
    )
    assert not getattr(args, "experts_shared_outer_loras", False), (
        "Multi-LoRA does not support --experts-shared-outer-loras; MoE expert adapters "
        "use the per-expert layout. Drop the flag (and --sglang-experts-shared-outer-loras)."
    )
    assert "muon" not in str(getattr(args, "optimizer", "")).lower(), (
        "Multi-LoRA does not support Muon: per-adapter decoupled stepping is only "
        "implemented for Adam-family per-slot optimizers"
    )
    assert not args.colocate, "Multi-LoRA requires separate training and sampling GPUs to retain accumulated gradients"
    assert (
        not getattr(args, "indep_dp", False) and "train" not in args.ft_components
    ), "Multi-LoRA does not support independent-DP training; remove 'train' from --ft-components"
    assert not args.offload_train, (
        "Multi-LoRA retains per-adapter gradient accumulation in GPU buffers between "
        "train calls; --offload-train would destroy it. Disable offload for multi-LoRA."
    )
    assert not getattr(args, "enable_witness", False), (
        "Multi-LoRA runs without the distributed optimizer (per-slot LayerWise "
        "optimizers); the witness module assumes use_distributed_optimizer"
    )
    assert getattr(args, "sglang_tokenizer_worker_num", 1) == 1, (
        "Multi-LoRA requires --sglang-tokenizer-worker-num 1: dynamic adapter loading "
        "requires a single tokenizer-side LoRA registry."
    )
    assert not args.calculate_per_token_loss, (
        "Multi-LoRA normalizes each sample by its adapter batch "
        "(sample-mean); per-token loss normalization would make adapter batch weights "
        "depend on batch contents. Drop --calculate-per-token-loss."
    )
    assert (getattr(args, "optimizer", "adam") or "adam").lower() == "adam", (
        "Multi-LoRA requires --optimizer adam: the per-slot SlotOptimizer only "
        f"implements Adam semantics; got --optimizer {args.optimizer}"
    )
    args.megatron_to_hf_mode = "bridge"
