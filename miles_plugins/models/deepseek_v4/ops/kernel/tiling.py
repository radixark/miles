"""Choose a kernel tiling the compiler's target can actually host.

The DeepSeek-V4 tilelang kernels ship with tilings tuned for one GPU; on a smaller shared-memory
budget they do not compile at all. Rather than tabulate replacements per architecture, this derives
candidates from the target's own limits and asks tilelang how much shared memory each one needs.

Both numbers come from tilelang. The budget is in the target description it builds for every
compilation; the requirement is the ``dyn_shared_memory_buf`` attribute on the lowered device
function, the same figure it formats into "Requested dynamic shared memory N exceeds device limit
M". Reading it needs lowering but not code generation, roughly 4x cheaper than a build.

This depends on three tilelang internals -- ``JITImpl.get_tir``, ``engine.lower`` with device
compilation disabled, and the attribute name -- all pinned by tests. It is a deliberate trade
against matching error strings, which would fail silently rather than loudly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Entry points that have already warned about an unreadable requirement, so each says it once.
_warned_unreadable: set[str] = set()

# tilelang sets this only on functions that use dynamic shared memory, so "absent" is ambiguous
# between "needs none" and "renamed". Named here so a rename fails in one place.
SHARED_MEMORY_ATTR = "dyn_shared_memory_buf"

# Diagnostic only: nothing branches on these. A rejection starts the search whether or not it is
# recognised, so a tilelang reword costs one unhelpful log line rather than the mechanism.
SHARED_MEMORY_EXCEEDED = "exceeds device limit"  # the shared-memory check at codegen
WARP_PARTITION_DEGENERATE = "Divide by zero"  # warp partitioning inside T.gemm
VECTORIZE_FAILED = "is_scalar"  # the vectorizer on the atomic store loop

RETRYABLE_BUILD_ERRORS = (SHARED_MEMORY_EXCEEDED, WARP_PARTITION_DEGENERATE, VECTORIZE_FAILED)

# A block_H that does not divide the head count makes a candidate inapplicable. Its sibling asserts
# -- a dim that is not a power of two, a topk that does not divide block_I -- are caller errors.
RETRYABLE_ASSERTIONS = ("block_H",)


def is_retryable_build_error(exc: BaseException) -> bool:
    """Is this a rejection this module already understands? Diagnostic only."""
    message = str(exc)
    if isinstance(exc, AssertionError):
        return any(marker in message for marker in RETRYABLE_ASSERTIONS)
    return any(marker in message for marker in RETRYABLE_BUILD_ERRORS)


# The smallest N one matrix-core instruction covers, which bounds how many warps a gemm's output
# can split across; below it tilelang's warp partitioning divides by zero. The one limit the target
# does not carry. MFMA is 16x16x16, Tensor Core MMA is m16n8k16. An unlisted target gets the
# smaller value, which only offers candidates that then fail to lower.
_MIN_GEMM_N_PER_WARP = {"hip": 16, "cuda": 8}
_MIN_GEMM_N_PER_WARP_DEFAULT = 8

# The kernels' own floor, from `padded_H = max(next_power_of_2(H), 16)`. Not derivable from
# min_gemm_n_per_warp: the head block is the QK^T gemm's M, that limit constrains its N.
MIN_HEAD_BLOCK = 16


@dataclass(frozen=True)
class DeviceLimits:
    """The subset of a tilelang target that constrains a tiling."""

    shared_memory_per_block: int
    warp_size: int
    max_threads_per_block: int
    min_gemm_n_per_warp: int

    @classmethod
    def from_target(cls, target: Any) -> DeviceLimits:
        attrs = target.attrs
        return cls(
            shared_memory_per_block=int(attrs["max_shared_memory_per_block"]),
            warp_size=int(attrs["thread_warp_size"]),
            max_threads_per_block=int(attrs.get("max_num_threads", attrs["max_threads_per_block"])),
            min_gemm_n_per_warp=_MIN_GEMM_N_PER_WARP.get(target.kind.name, _MIN_GEMM_N_PER_WARP_DEFAULT),
        )

    def threads_for(self, gemm_n: int) -> int:
        """The widest thread block whose warps each still get a full matrix-core tile.

        The shipped tilings used exactly this expression -- block_I 64, 32, 16 give 256, 128, 64 --
        so those thread counts were never free parameters.
        """
        warps = max(1, gemm_n // self.min_gemm_n_per_warp)
        return min(warps * self.warp_size, self.max_threads_per_block)


def current_target() -> Any:
    from tilelang.utils.target import determine_target

    return determine_target("auto", return_object=True)


def shared_memory_required(prim_func: Any, target: Any) -> int | None:
    """How much dynamic shared memory tilelang's lowered kernel asks for, in bytes.

    Lowers without generating device code, so it is much cheaper than a build while giving the
    exact figure the build would have checked. None when it cannot be read -- see
    SHARED_MEMORY_ATTR.
    """
    from tilelang import engine

    with target:
        lowered = engine.lower(prim_func, target=target, enable_host_codegen=False, enable_device_compile=False)
    return largest_declared_shared_memory(lowered)


def largest_declared_shared_memory(lowered: Any) -> int | None:
    """Read the requirement off a lowered module. Separate so it is testable without a GPU.

    None when no function declares the attribute. Returning 0 would make every candidate look like
    it fits, so the search would take the largest and fail with the error it exists to route around.
    """
    device_mod = getattr(lowered, "device_mod", lowered)
    declared = [
        int(func.attrs[SHARED_MEMORY_ATTR])
        for _, func in device_mod.functions.items()
        if func.attrs is not None and SHARED_MEMORY_ATTR in func.attrs
    ]
    return max(declared) if declared else None


def halvings(start: int, floor: int) -> Iterator[int]:
    """``start``, then repeated halvings, down to and including ``floor``."""
    value = start
    while value >= floor:
        yield value
        value //= 2


# ---------------------------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------------------------


def build_with_largest_fitting_tiling(
    *,
    requested: T,
    derived: Iterable[T],
    compile_tiling: Callable[[T], Any],
    required_bytes: Callable[[T], int | None],
    budget: int,
    describe: Callable[[T], str],
    what: str,
) -> tuple[Any, T]:
    """Compile the requested tiling; if it will not build, compile the largest one that fits.

    The requested tiling is attempted optimistically, so a device that fits it pays nothing here --
    not a wasted compilation and not even a lowering. Only a rejection starts the search, and the
    search compiles exactly one candidate.

    Any rejection starts it, whether or not this module recognises the message: gating on the text
    would let a tilelang reword disable the fallback silently. Starting unconditionally cannot give
    a wrong answer, because a caller error fails every candidate alike and ends at the ``raise``
    below with the caller's own message -- which names the requested shape and is the useful one.
    """
    try:
        return compile_tiling(requested), requested
    except Exception as exc:
        # Carrying the exception out of its `except` block would keep __traceback__, and with it
        # every caller's locals -- during a forward, its activation tensors, freeable only by the
        # cyclic collector. The message is all the re-raise needs.
        requested_error = exc.with_traceback(None)
        if not is_retryable_build_error(exc):
            logger.debug(
                "[%s] the requested tiling was refused for an unrecognised reason (%s: %s); "
                "searching for a smaller one anyway",
                what,
                type(exc).__name__,
                str(exc).strip().splitlines()[0] if str(exc).strip() else "",
            )

    for candidate in derived:
        try:
            required = required_bytes(candidate)
        except Exception:
            continue  # cannot be lowered at all; the requested tiling's error is the one to report
        if required is None:
            # Unplannable, so fall back to compiling it and letting that be the answer: a build per
            # candidate instead of a lowering, which is the price of not knowing.
            if what not in _warned_unreadable:
                logger.warning(
                    "[%s] tilelang did not report %s on the lowered kernel, so tilings are being "
                    "compiled to find out whether they fit. Check whether tilelang renamed it.",
                    what,
                    SHARED_MEMORY_ATTR,
                )
                _warned_unreadable.add(what)
            try:
                kernel = compile_tiling(candidate)
            except Exception:
                continue
            # Not the message below: with no requirement to attribute the rejection to, all this
            # says is that the candidate is the first that built.
            logger.warning(
                "[%s] the requested tiling did not build; compiled the largest candidate that did: %s",
                what,
                describe(candidate),
            )
            return kernel, candidate
        if required > budget:
            continue
        logger.warning(
            "[%s] shared memory forced a smaller tiling on this GPU: %s (needs %d B of %d)",
            what,
            describe(candidate),
            required,
            budget,
        )
        return compile_tiling(candidate), candidate

    raise requested_error


# Candidate orders. The shrink order is a statement about cost, not about any one GPU: give up
# pipeline depth first (latency hiding only), then the KV/index block (more inner iterations), and
# only then the head block (more grid blocks, and less output one workgroup can accumulate).


@dataclass(frozen=True)
class ForwardTiling:
    num_stages: int
    block_I: int
    threads: int
    block_H: int | None  # None keeps the kernel's own "one block of padded_H"

    def describe(self) -> str:
        return f"num_stages={self.num_stages} block_I={self.block_I} threads={self.threads} block_H={self.block_H}"


def sparse_mla_forward_tilings(
    *, padded_heads: int, topk: int, block_I: int, limits: DeviceLimits
) -> Iterator[ForwardTiling]:
    """Progressively smaller forward tilings, largest first. Excludes the requested one."""
    for head_block in halvings(padded_heads, MIN_HEAD_BLOCK):
        for index_block in halvings(block_I, limits.min_gemm_n_per_warp):
            if topk % index_block:
                continue  # would trip sparse_mqa_fwd's own divisibility assert
            yield ForwardTiling(
                num_stages=1,
                block_I=index_block,
                threads=limits.threads_for(index_block),
                # None is the kernel's own "one block of padded_H", so a device that needs no
                # shrinking stays on the exact code path it had before this module existed.
                block_H=None if head_block == padded_heads else head_block,
            )


@dataclass(frozen=True)
class BackwardTiling:
    block_size: int
    threads: int
    split_store: int
    stage_dq_through_shared: bool
    max_block_H: int | None  # None keeps the kernel's own choice

    def describe(self) -> str:
        return (
            f"block_size={self.block_size} threads={self.threads} split_store={self.split_store} "
            f"stage_dq_through_shared={self.stage_dq_through_shared} max_block_H={self.max_block_H}"
        )


def sparse_mla_backward_tilings(
    *, padded_heads: int, topk: int, block_size: int, limits: DeviceLimits
) -> Iterator[BackwardTiling]:
    """Progressively smaller backward tilings, largest first. Excludes the requested one.

    Two levers here are not tile sizes. ``split_store`` sets the height of ``acc_dkv_shared``, an
    fp32 buffer, so raising it to ``block_size`` cuts that buffer to one row at the cost of an
    atomic store pass per row. ``stage_dq_through_shared`` drops a whole ``[block_H, D]`` buffer by
    writing dQ straight to global memory. Both come last, because both cost bandwidth rather than
    parallelism, and staging dQ is kept for as long as possible: tilelang reuses dQ_shared, so it
    costs 480 B rather than the 16 KiB the buffer suggests.
    """
    for head_block in halvings(padded_heads, MIN_HEAD_BLOCK):
        for kv_block in halvings(block_size, limits.min_gemm_n_per_warp):
            if topk % kv_block:
                continue
            for stage_dq in (True, False):
                for split_store in dict.fromkeys((2, kv_block)):  # deduped, order preserved
                    if kv_block % split_store:
                        continue
                    yield BackwardTiling(
                        block_size=kv_block,
                        threads=limits.threads_for(kv_block),
                        split_store=split_store,
                        stage_dq_through_shared=stage_dq,
                        max_block_H=head_block,
                    )


def indexer_forward_block_ns(*, block_N: int, limits: DeviceLimits) -> Iterator[int]:
    """Halvings of the requested block_N. Excludes the requested one.

    block_N only sets how much of the KV axis one workgroup sweeps per iteration, so shrinking it
    costs performance and nothing else.
    """
    yield from halvings(block_N // 2, limits.min_gemm_n_per_warp)
