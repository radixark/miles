"""--dsa-attention-backend cake: the Cake DSA kernels of the miles-ext-kernels submodule (miles_ext_kernels.dsa).

The sparse-attention and indexer dispatchers reach them through the opt-in rows CAKE_SPARSE_ATTENTION_CONFIG and
CAKE_INDEXER_CONFIG, which call :func:`ops`. A plugin spec that builds DSA layers with the cake backend calls
:func:`setup` once: it hands the kernels the values of their own arguments (``--cake-dsa-variant`` and the rest, in
``args.miles_ext_kernels_args``) and fails before the first step on a device or an indexer geometry they do not run.

The sparse attention runs the variant ``--cake-dsa-variant`` selects: ``det`` (bit-identical forward and backward) or
``fast`` (FlashInfer's kernels for 64 local query heads without a sink; any other call runs ``det``). The indexer
scores are always ``det``. CUDA-only (NVIDIA SM100a / SM103a): only the cake rows of the DSA dispatchers reach it, and
:func:`setup` rejects other devices when the model is built.
"""

import argparse

# The indexer geometries the Cake score kernels serve
INDEXER_HEADS = (8, 16, 32, 64)
INDEXER_HEAD_DIM = 128


def ops():
    """``miles_ext_kernels.dsa.ops``, with a clear error when the submodule is not importable."""
    try:
        from miles_ext_kernels.dsa import ops
    except ImportError as exc:
        raise ImportError(
            "DSA kernel backend 'cake' runs miles_ext_kernels.dsa from the miles-ext-kernels submodule: pass "
            "--use-miles-ext-kernels and check the submodule out with "
            "'git submodule update --init 3rdparty/miles-ext-kernels'."
        ) from exc
    return ops


def check_indexer_geometry(index_heads: int, index_head_dim: int) -> None:
    """Raise when the Cake indexer score kernels do not serve ``index_heads`` x ``index_head_dim``."""
    if index_heads not in INDEXER_HEADS or index_head_dim != INDEXER_HEAD_DIM:
        raise ValueError(
            "--dsa-attention-backend cake scores the DSA indexer with "
            f"{'/'.join(map(str, INDEXER_HEADS))} heads of dimension {INDEXER_HEAD_DIM}; this model has "
            f"{index_heads} heads of dimension {index_head_dim}. Use the default --dsa-attention-backend tilelang."
        )


def setup(args, *, index_heads: int, index_head_dim: int) -> None:
    """Prepare the Cake DSA kernels for a model whose spec builds its DSA layers with the cake backend.

    ``args`` is the miles configuration the spec receives; ``index_heads`` / ``index_head_dim`` are the model's
    indexer geometry from its HF config. Idempotent: every spec build (one per virtual pipeline stage) may call it.
    """
    check_indexer_geometry(index_heads, index_head_dim)
    kernels = ops()
    from miles_ext_kernels.dsa import configure

    configure(argparse.Namespace(**(args.miles_ext_kernels_args or {})))
    kernels.check_device()
