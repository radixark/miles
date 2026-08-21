"""DeepSeek-V4 command-line arguments and their translation to Megatron config.

Imports nothing from megatron or the plugin's kernels: argument parsing runs long
before tilelang can be loaded.
"""

from argparse import ArgumentParser, Namespace

DSV4_SPEC_MODULE = "miles_plugins.models.deepseek_v4.deepseek_v4"

# DeepSeek-V4 model shape, expressed as Megatron config fields. The CLI keeps the
# --dsv4-* spelling the recipes already use.
_ARG_TO_MEGATRON_FIELD = {
    "dsv4_window_size": "csa_window_size",
    "dsv4_compress_ratios": "csa_compress_ratios",
    "dsv4_compress_rope_theta": "csa_compress_rotary_base",
    "dsv4_o_groups": "o_groups",
    "dsv4_o_lora_rank": "o_lora_rank",
    "dsv4_n_hash_layers": "moe_n_hash_layers",
    "dsv4_hc_mult": "num_residual_streams",
    "dsv4_hc_sinkhorn_iters": "mhc_sinkhorn_iterations",
}


def is_dsv4_model(args: Namespace) -> bool:
    """Whether this run builds its layers from the DeepSeek-V4 plugin spec."""
    spec = getattr(args, "spec", None)
    return bool(spec) and spec[0] == DSV4_SPEC_MODULE


def add_dsv4_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Declare the DeepSeek-V4 arguments."""
    group = parser.add_argument_group(title="deepseek-v4")
    group.add_argument(
        "--dsv4-impl",
        type=str,
        choices=["miles", "megatron"],
        default="miles",
        help=(
            "Which DeepSeek-V4 attention implementation to train with. 'miles' is the plugin path "
            "(BSHD, sparse context parallelism, tilelang kernels, miles' hyper-connections) and is "
            "the only one that supports tensor parallelism. 'megatron' is Megatron's native "
            "dsv4_hybrid path (THD, cuDNN or unfused kernels, native hyper-connections). The two "
            "read the same HuggingFace checkpoint but their torch_dist checkpoints are not "
            "interchangeable."
        ),
    )
    group.add_argument("--dsv4-window-size", type=int, default=128)
    group.add_argument("--dsv4-compress-ratios", type=int, nargs="+", default=None)
    group.add_argument("--dsv4-compress-rope-theta", type=float, default=160000)
    group.add_argument("--dsv4-o-groups", type=int, default=8)
    group.add_argument("--dsv4-o-lora-rank", type=int, default=1024)
    group.add_argument("--dsv4-n-hash-layers", type=int, default=3)
    group.add_argument("--dsv4-hc-mult", type=int, default=4)
    group.add_argument("--dsv4-hc-sinkhorn-iters", type=int, default=20)
    return parser


def normalize_dsv4_args(args: Namespace) -> None:
    """Feed the --dsv4-* arguments into the Megatron fields the model is built from.

    Must run before ``core_transformer_config_from_args``: the attention variant decides
    which post-init contract Megatron enforces on the config.
    """
    _validate_impl(args)
    for arg, field in _ARG_TO_MEGATRON_FIELD.items():
        setattr(args, field, getattr(args, arg))
    # Both implementations take their hyper-connections from Megatron's own module.
    args.enable_hyper_connections = True
    args.experimental_attention_variant = "dsv4_hybrid" if args.dsv4_impl == "megatron" else "dsv4"


def _validate_impl(args: Namespace) -> None:
    kernel_backend = getattr(args, "dsa_kernel_backend", None)
    if args.dsv4_impl == "megatron":
        if args.tensor_model_parallel_size > 1:
            raise ValueError(
                f"--dsv4-impl megatron requires tensor-model-parallel-size 1, got "
                f"{args.tensor_model_parallel_size}. Use --dsv4-impl miles for tensor parallelism."
            )
        if kernel_backend == "tilelang":
            raise ValueError(
                "--dsv4-impl megatron does not support --dsa-kernel-backend tilelang; "
                "use 'cudnn' for fused kernels or 'none' for the PyTorch fallback."
            )
    elif kernel_backend == "cudnn":
        raise ValueError(
            "--dsv4-impl miles runs its own tilelang kernels and ignores cuDNN; "
            "drop --dsa-kernel-backend or switch to --dsv4-impl megatron."
        )


# DeepSeek-V4 weights follow Megatron's names, so one checkpoint serves both --dsv4-impl
# values. Checkpoints written before that switch used the plugin's own spellings and would
# load into a silently half-initialized model, so refuse them by name.
_SUPERSEDED_WEIGHT_NAMES = (
    "self_attention.wq_a.weight",
    "self_attention.wo_a.weight",
    "hc_attn_fn",
    "hc_head_params.hc_head_fn",
)


def assert_checkpoint_is_current(load_dir: str) -> None:
    """Reject DeepSeek-V4 checkpoints written before the attention weights were renamed."""
    from pathlib import Path

    from torch.distributed.checkpoint import FileSystemReader

    iteration_file = Path(load_dir) / "latest_checkpointed_iteration.txt"
    step = iteration_file.read_text().strip() if iteration_file.is_file() else None
    directory = Path(load_dir) / step if step else Path(load_dir)
    if not directory.is_dir():
        return

    metadata = FileSystemReader(directory).read_metadata()
    stale = [key for key in metadata.state_dict_metadata if any(name in key for name in _SUPERSEDED_WEIGHT_NAMES)]
    if stale:
        raise ValueError(
            f"{load_dir} is a DeepSeek-V4 checkpoint from before the attention weights were renamed "
            f"to Megatron's names (found {stale[0]!r}); loading it would leave the attention layers "
            f"uninitialized. Re-convert from the HuggingFace checkpoint, or ask your coding agent: "
            f'"remap the DeepSeek-V4 checkpoint at {load_dir} from the old attention weight names '
            f"to the current ones (see _RENAMED_ATTENTION_WEIGHT in "
            f'miles_plugins/models/deepseek_v4/arguments.py)."'
        )
