"""Select the raw DSA implementation before Megatron validates its config."""

from argparse import ArgumentParser, Namespace

MILES_DSA_SPEC = ("miles_plugins.models.glm5.glm5", "get_glm5_spec")
MEGATRON_DSA_SPEC = ("miles_plugins.models.glm5.megatron_spec", "get_dsa_spec")


def add_dsa_arguments(parser: ArgumentParser) -> ArgumentParser:
    # Distinguish an omitted Megatron CP setting from an explicit incompatible choice.
    parser.set_defaults(cp_comm_type=None)
    group = parser.add_argument_group(title="deepseek-sparse-attention")
    group.add_argument(
        "--dsa-impl",
        choices=["miles", "megatron"],
        default="miles",
        help=(
            "Raw-mode DSA implementation for DeepSeek-V3.2 and GLM-5/5.1/5.2. "
            "'miles' uses the existing TileLang plugin; 'megatron' uses native DSA "
            "with --dsa-kernel-backend. Convert torch_dist checkpoints with the same selection. "
            "DeepSeek-V4 uses --dsv4-impl instead; bridge mode uses --dsa-attention-backend."
        ),
    )
    group.add_argument(
        "--miles-dsa-topk-backend",
        choices=["torch", "flashinfer"],
        default="torch",
        help="DSA indexer top-k backend for both raw DSA implementations.",
    )
    return parser


def normalize_dsa_args(args: Namespace, hf_config) -> None:
    """Map the shared Miles DSA spec and HF indexer conventions to native Megatron."""
    if args.dsa_impl != "megatron":
        return
    if getattr(args, "megatron_to_hf_mode", "raw") != "raw":
        raise ValueError("--dsa-impl megatron requires --megatron-to-hf-mode raw")
    if tuple(getattr(args, "spec", None) or ()) not in (MILES_DSA_SPEC, MEGATRON_DSA_SPEC):
        raise ValueError("--dsa-impl megatron requires the shared DeepSeek-V3.2/GLM DSA spec")
    if hf_config.model_type not in ("deepseek_v32", "glm_moe_dsa"):
        raise ValueError(f"--dsa-impl megatron does not support model_type={hf_config.model_type!r}")
    if getattr(args, "context_parallel_size", 1) > 1 and getattr(args, "allgather_cp", False):
        raise ValueError(
            "--dsa-impl megatron uses zigzag CP token partitioning; remove --allgather-cp. "
            "Megatron's attention communication still uses --cp-comm-type allgather."
        )
    if args.cp_comm_type is None:
        args.cp_comm_type = ["allgather"]
    elif any(comm_type != "allgather" for comm_type in args.cp_comm_type):
        raise ValueError("--dsa-impl megatron requires --cp-comm-type allgather")

    args.spec = list(MEGATRON_DSA_SPEC)
    args.experimental_attention_variant = "dsa"
    args.enable_experimental = True
    args.dsa_indexer_topk_backend = args.miles_dsa_topk_backend
    args.dsa_indexer_n_heads = hf_config.index_n_heads
    args.dsa_indexer_head_dim = hf_config.index_head_dim
    args.dsa_indexer_topk = hf_config.index_topk
    args.dsa_indexer_rope_interleaved = bool(getattr(hf_config, "indexer_rope_interleave", False))
    args.indexer_rope_interleave = args.dsa_indexer_rope_interleaved
    args.dsa_indexer_topk_freq = getattr(hf_config, "index_topk_freq", 1) or 1
    args.dsa_indexer_skip_topk_offset = getattr(hf_config, "index_skip_topk_offset", 0) or 0
    # Match the existing raw indexer: no Hadamard rotation, fp32 key LayerNorm,
    # and no auxiliary indexer objective unless the user explicitly requests one.
    args.dsa_indexer_rotate_activation = False
    args.dsa_indexer_k_norm_epsilon = 1e-6
    args.dsa_indexer_k_norm_fp32 = True
    if getattr(args, "dsa_indexer_loss_coeff", None) is None:
        args.dsa_indexer_loss_coeff = 0.0
    if args.dsa_indexer_loss_coeff < 0:
        raise ValueError("--dsa-indexer-loss-coeff must be non-negative")
