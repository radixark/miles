"""DeepSeek-V4 command-line arguments.

The model shape is passed through Megatron's own flags (--csa-window-size,
--o-groups, --num-residual-streams, ...); the plugin resolves --model-impl into
the fields that select which of the two implementations trains the model.

Imports nothing from megatron or the plugin's kernels: argument parsing runs long
before tilelang can be loaded.
"""

from argparse import Namespace


def is_dsv4_model(args: Namespace) -> bool:
    """Whether the checkpoint is a DeepSeek-V4 (HF model_type)."""
    return args.model_family == "deepseek_v4"


def apply_dsv4_model_impl(args: Namespace) -> None:
    """Resolve --model-impl into the Megatron fields that select the implementation.

    Must run before ``core_transformer_config_from_args``: the attention variant decides
    which post-init contract Megatron enforces on the config.
    """
    if args.model_impl is None:
        args.model_impl = "megatron"
    _validate_impl(args)
    # Both implementations take their hyper-connections from Megatron's own module.
    args.enable_hyper_connections = True
    args.experimental_attention_variant = "dsv4_hybrid" if args.model_impl == "megatron" else "dsv4"


def _validate_impl(args: Namespace) -> None:
    kernel_backend = getattr(args, "dsa_kernel_backend", None)
    if args.model_impl == "megatron":
        if args.tensor_model_parallel_size > 1:
            raise ValueError(
                f"--model-impl megatron requires tensor-model-parallel-size 1, got "
                f"{args.tensor_model_parallel_size}. Use --model-impl miles for tensor parallelism."
            )
        if kernel_backend == "tilelang":
            raise ValueError(
                "--model-impl megatron does not support --dsa-kernel-backend tilelang; "
                "use 'cudnn' for fused kernels or 'none' for the PyTorch fallback."
            )
    elif kernel_backend == "cudnn":
        raise ValueError(
            "--model-impl miles runs its own tilelang kernels and ignores cuDNN; "
            "drop --dsa-kernel-backend or switch to --model-impl megatron."
        )


def assert_checkpoint_is_current(load_dir: str) -> None:
    """Reject DeepSeek-V4 checkpoints from before the weights took Megatron's names."""
    from pathlib import Path

    from torch.distributed.checkpoint import FileSystemReader

    path = Path(load_dir)
    step_file = path / "latest_checkpointed_iteration.txt"
    if step_file.is_file():
        path = path / step_file.read_text().strip()
    if not path.is_dir():
        return

    if any("self_attention.wq_a." in key for key in FileSystemReader(path).read_metadata().state_dict_metadata):
        raise ValueError(
            f"{load_dir} was written before the DeepSeek-V4 weights took Megatron's names and "
            "would load half-initialized. Re-convert it from the HuggingFace checkpoint."
        )
