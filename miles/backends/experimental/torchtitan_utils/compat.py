"""torch-2.11 compat for torchtitan (pip-installed, pinned). Import before any torchtitan module.

torchtitan main tracks torch nightly. On torch 2.11 there are two gaps found so far,
both shimmed here:

1. Import-time: a missing ``DataParallelMeshDims`` symbol that
   ``torchtitan.distributed.fsdp`` (and, transitively, every model's ``parallelize``
   module) imports. A placeholder unblocks the whole chain and is never instantiated on
   the paths we use (it only reaches ``fully_shard`` when ``dp_mesh_dims``/
   ``edp_mesh_dims`` are explicitly set — the "full_dtensor" spmd backend, unused here).

2. Runtime, on the packed-document flex-attention mask path (the one RL training
   actually needs): ``Decoder._create_flex_attention_mask`` passes nightly-only
   ``separate_full_blocks=`` to ``create_block_mask``, which 2.11 doesn't accept
   (``TypeError``). It's a pure kernel-block-iteration-order perf knob — torchtitan's
   own comment says it's disabled under batch-invariant mode anyway — so dropping it on
   2.11 changes performance, not correctness. Shimmed by wrapping ``create_block_mask``.

Also asserts the pinned commit (main tracks nightly; any bump can silently break 2.11)
and guards against ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments`` (set as an import
side effect by ``torchtitan.experiments.rl``, which we must never import — it also
hard-imports vllm; that side effect breaks the CUDA-IPC weight-sync transport).
"""

import dataclasses
import logging
import os

logger = logging.getLogger(__name__)

TORCHTITAN_PINNED_COMMIT = "7e3f2ebc"


def _assert_no_expandable_segments() -> None:
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" in conf:
        raise RuntimeError(
            "PYTORCH_CUDA_ALLOC_CONF contains 'expandable_segments', which is "
            "incompatible with CUDA-IPC weight sync. Never import "
            "torchtitan.experiments.rl (it sets this at import time)."
        )


def _shim_data_parallel_mesh_dims() -> None:
    import torch.distributed.fsdp as _fsdp

    if hasattr(_fsdp, "DataParallelMeshDims"):
        return

    @dataclasses.dataclass
    class DataParallelMeshDims:  # placeholder; never instantiated on our paths
        shard: object = None
        replicate: object = None

    _fsdp.DataParallelMeshDims = DataParallelMeshDims


def _shim_create_block_mask_kwargs() -> None:
    import inspect

    from torch.nn.attention import flex_attention

    accepted = inspect.signature(flex_attention.create_block_mask).parameters
    if "separate_full_blocks" in accepted:
        return

    _orig = flex_attention.create_block_mask

    def create_block_mask(*args, separate_full_blocks=None, **kwargs):
        return _orig(*args, **kwargs)

    flex_attention.create_block_mask = create_block_mask
    # torchtitan's decoder module does `from torch.nn.attention.flex_attention import
    # create_block_mask` at call time (inside the method), so patching the flex_attention
    # module attribute above is sufficient — no separate re-patch needed there.


def _check_pinned_commit() -> None:
    """Best-effort pin check. A git-URL pip install exposes the commit via
    importlib.metadata's direct_url.json; source installs (e.g. -e .) don't, so this
    only warns rather than raising in that case."""
    try:
        from importlib.metadata import distribution

        direct_url = distribution("torchtitan").read_text("direct_url.json")
    except Exception:
        direct_url = None
    if direct_url is None:
        logger.warning("torchtitan install metadata unavailable; cannot verify pinned commit")
        return
    if TORCHTITAN_PINNED_COMMIT not in direct_url:
        logger.warning(
            f"torchtitan direct_url.json does not mention pinned commit "
            f"{TORCHTITAN_PINNED_COMMIT!r}: {direct_url.strip()}"
        )


def _probe_fla() -> bool:
    try:
        import fla  # noqa: F401
    except ImportError:
        return False
    return True


_assert_no_expandable_segments()
_shim_data_parallel_mesh_dims()
_shim_create_block_mask_kwargs()
_check_pinned_commit()

HAS_FLA = _probe_fla()
