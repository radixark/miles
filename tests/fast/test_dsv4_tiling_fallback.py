"""The shared-memory tiling search: what it asks tilelang, what it retries, and what it caches.

The shared-memory query is stubbed out, so no GPU and no compilation are involved; the stubs stand
in for a lowering of the candidate kernel, which is where the real implementation gets its numbers.
"""

import inspect

import pytest

tilelang = pytest.importorskip("tilelang", reason="the DeepSeek-V4 kernels are tilelang modules")

from miles_plugins.models.deepseek_v4.ops.kernel import tilelang_indexer_fwd as indexer_fwd  # noqa: E402
from miles_plugins.models.deepseek_v4.ops.kernel import tilelang_sparse_mla_bwd as sparse_mla_bwd  # noqa: E402
from miles_plugins.models.deepseek_v4.ops.kernel import tilelang_sparse_mla_fwd as sparse_mla_fwd  # noqa: E402
from miles_plugins.models.deepseek_v4.ops.kernel import tiling  # noqa: E402

# gfx942 / MI300X-MI308X, as tilelang describes it. Every measured number in this file is from it.
GFX942 = tiling.DeviceLimits(
    shared_memory_per_block=65536, warp_size=64, max_threads_per_block=256, min_gemm_n_per_warp=16
)
# An Ampere-class CUDA target, for contrast: more shared memory, half the warp, and a Tensor Core
# minimum of 8 instead of MFMA's 16.
SM80 = tiling.DeviceLimits(
    shared_memory_per_block=166912, warp_size=32, max_threads_per_block=1024, min_gemm_n_per_warp=8
)

SHARED_MEM_ERROR = RuntimeError("Requested dynamic shared memory 85488 exceeds device limit 65536")


# ---------------------------------------------------------------------------------------------
# Device limits come from the compiler's target, not from the framework
# ---------------------------------------------------------------------------------------------


def test_limits_are_read_from_the_tilelang_target():
    """No torch device API and no architecture string: the target carries all four numbers."""
    from tvm.target import Target

    target = Target(
        "hip -keys=hip,gpu -max_num_threads=256 -max_shared_memory_per_block=65536"
        " -max_threads_per_block=256 -mcpu=gfx942 -thread_warp_size=64"
    )
    assert tiling.DeviceLimits.from_target(target) == GFX942


def test_an_unknown_target_kind_gets_the_permissive_gemm_minimum():
    """A target nobody has characterised must not be refused, only planned for conservatively."""
    from tvm.target import Target

    target = Target("cuda -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32")
    assert tiling.DeviceLimits.from_target(target).min_gemm_n_per_warp == 8


@pytest.mark.parametrize(
    ("gemm_n", "expected"),
    [(64, 256), (32, 128), (16, 64), (8, 64)],  # 8 clamps to one warp: there is no half-warp
)
def test_thread_width_is_derived_from_the_warp_and_the_matrix_core(gemm_n, expected):
    """The shipped tilings used exactly these thread counts, so they were never free parameters."""
    assert GFX942.threads_for(gemm_n) == expected


def test_a_narrower_matrix_core_allows_more_warps():
    """Tensor Core's minimum N is 8, so the same gemm width supports twice as many warps."""
    assert SM80.threads_for(64) == 8 * 32


def test_the_head_block_floor_is_not_the_gemm_width_floor():
    """Different axes: the head block is the gemm's M, min_gemm_n_per_warp is its N.

    Both are 16 on gfx942, which is a coincidence; deriving one from the other would make the head
    block floor move with the matrix-core shape.
    """
    assert tiling.MIN_HEAD_BLOCK == 16
    assert list(tiling.halvings(64, tiling.MIN_HEAD_BLOCK)) == [64, 32, 16]
    assert tiling.MIN_HEAD_BLOCK != SM80.min_gemm_n_per_warp


# ---------------------------------------------------------------------------------------------
# The shared-memory requirement comes from tilelang, exactly
# ---------------------------------------------------------------------------------------------


def test_the_attribute_the_requirement_is_read_from_is_the_one_tilelang_sets():
    """A canary: tilelang's wrapper reads the same attribute to build its error message."""
    from tilelang.jit.adapter import wrapper

    assert tiling.SHARED_MEMORY_ATTR in inspect.getsource(wrapper)


def test_lowering_without_device_compilation_is_a_supported_call():
    """The other internal coupling: engine.lower must still accept being told not to compile."""
    from tilelang import engine

    parameters = inspect.signature(engine.lower).parameters
    assert "enable_device_compile" in parameters
    assert "enable_host_codegen" in parameters


def test_the_kernel_builders_still_expose_their_tir():
    """get_tir is what lets a candidate be measured without being built."""
    for builder in (sparse_mla_fwd.sparse_mqa_fwd, sparse_mla_bwd.bwd, indexer_fwd.tl_indexer_fwd_impl):
        assert hasattr(builder, "get_tir"), builder


def test_the_requirement_is_the_largest_any_device_function_declares():
    """A lowered module can hold several functions; the block budget has to cover the biggest."""

    class _Func:
        def __init__(self, attrs):
            self.attrs = attrs

    class _Mod:
        functions = {
            "a": _Func({tiling.SHARED_MEMORY_ATTR: 51184}),
            "b": _Func({tiling.SHARED_MEMORY_ATTR: 68608}),
            "c": _Func(None),
            "d": _Func({"unrelated": 1}),
        }

    assert tiling.largest_declared_shared_memory(_Mod()) == 68608


def test_an_unreadable_requirement_is_none_not_zero():
    """Zero would be the worst possible answer: every candidate would look like it fits."""

    class _Empty:
        functions = {}

    class _Renamed:
        functions = {"main": type("F", (), {"attrs": {"dyn_shared_memory_buffer": 141312}})()}

    assert tiling.largest_declared_shared_memory(_Empty()) is None
    assert tiling.largest_declared_shared_memory(_Renamed()) is None


def test_an_unreadable_requirement_falls_back_to_compiling_rather_than_assuming_it_fits():
    """A tilelang rename must degrade to try-and-retry, not silently undo the whole fallback."""
    too_big = RuntimeError("Requested dynamic shared memory 141312 exceeds device limit 65536")
    compiled = []

    def compile_tiling(candidate):
        compiled.append(candidate)
        if candidate in ("requested", "biggest"):
            raise too_big
        return f"kernel({candidate})"

    kernel, chosen = tiling.build_with_largest_fitting_tiling(
        requested="requested",
        derived=["biggest", "fits"],
        compile_tiling=compile_tiling,
        required_bytes=lambda _: None,
        budget=65536,
        describe=str,
        what="probe",
    )

    assert chosen == "fits", "the search must keep going instead of taking the first candidate"
    assert compiled == ["requested", "biggest", "fits"]
    assert kernel == "kernel(fits)"


def test_the_requirement_is_read_through_device_mod_when_there_is_one():
    class _Inner:
        functions = {"k": type("F", (), {"attrs": {tiling.SHARED_MEMORY_ATTR: 34032}})()}

    class _Outer:
        device_mod = _Inner()
        functions = {"host": type("F", (), {"attrs": {tiling.SHARED_MEMORY_ATTR: 999999}})()}

    assert tiling.largest_declared_shared_memory(_Outer()) == 34032


# ---------------------------------------------------------------------------------------------
# The requested tiling is compiled first, and nothing else is even measured
# ---------------------------------------------------------------------------------------------


def _driver(compiled, requirements, derived, *, budget=65536):
    """Run the driver with scripted outcomes; returns (result, compiled_order, measured_order)."""
    compiled_order, measured_order = [], []

    def compile_tiling(t):
        compiled_order.append(t)
        outcome = compiled.get(t, f"kernel({t})")
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def required_bytes(t):
        measured_order.append(t)
        outcome = requirements[t]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    result = tiling.build_with_largest_fitting_tiling(
        requested="requested",
        derived=derived,
        compile_tiling=compile_tiling,
        required_bytes=required_bytes,
        budget=budget,
        describe=str,
        what="probe",
    )
    return result, compiled_order, measured_order


def test_a_fitting_request_is_compiled_once_and_nothing_is_measured():
    """The whole point: on a device that fits the shipped tiling this costs nothing."""
    (kernel, chosen), compiled, measured = _driver({}, {}, ["never-reached"])

    assert kernel == "kernel(requested)"
    assert chosen == "requested"
    assert compiled == ["requested"]
    assert measured == [], "a fitting request must not trigger a single lowering"


def test_a_refused_request_falls_through_to_the_first_candidate_that_fits():
    (kernel, chosen), compiled, measured = _driver(
        {"requested": SHARED_MEM_ERROR},
        {"too-big": 70000, "fits": 51184, "smaller": 33000},
        ["too-big", "fits", "smaller"],
    )

    assert chosen == "fits"
    assert compiled == ["requested", "fits"], "exactly one candidate is compiled"
    assert measured == ["too-big", "fits"], "measuring stops at the first that fits"


def test_a_candidate_that_cannot_be_lowered_is_skipped():
    """The two non-size rejections surface while lowering, so they are caught there."""
    (_, chosen), _, measured = _driver(
        {"requested": SHARED_MEM_ERROR},
        {"degenerate": RuntimeError("... Divide by zero ..."), "fits": 51184},
        ["degenerate", "fits"],
    )

    assert chosen == "fits"
    assert measured == ["degenerate", "fits"]


def test_the_request_s_own_rejection_is_raised_when_nothing_fits():
    """It names the shape the caller asked for, which the last candidate's error does not."""
    with pytest.raises(RuntimeError, match="85488"):
        _driver(
            {"requested": SHARED_MEM_ERROR},
            {"a": 90000, "b": 80000},
            ["a", "b"],
        )


def test_an_empty_candidate_set_raises_the_request_s_rejection_not_none():
    """There is no "last error" to be None here: the request has always been tried first."""
    with pytest.raises(RuntimeError, match="85488"):
        _driver({"requested": SHARED_MEM_ERROR}, {}, [])


def test_a_genuine_bug_from_the_request_still_surfaces():
    """Searching anyway must not swallow it: no candidate fits, so it is what gets raised."""
    with pytest.raises(RuntimeError, match="illegal memory access"):
        _driver({"requested": RuntimeError("illegal memory access")}, {"a": 90000}, ["a"])


def test_a_caller_error_from_the_request_still_surfaces():
    """A topk or dim the kernel rejects fails every candidate alike and ends here."""
    with pytest.raises(AssertionError, match="power of 2"):
        _driver(
            {"requested": AssertionError("dim must be power of 2, got 500")},
            {"a": AssertionError("dim must be power of 2, got 500")},
            ["a"],
        )


def test_an_unrecognised_rejection_still_finds_a_fitting_tiling():
    """A tilelang release that rewords "exceeds device limit" must not disable the search."""
    (_, chosen), compiled, _ = _driver(
        {"requested": RuntimeError("shared memory request of 85488 over the 65536 B cap")},
        {"fits": 51184},
        ["fits"],
    )

    assert chosen == "fits"
    assert compiled == ["requested", "fits"]


def test_an_inapplicable_block_h_is_retried():
    """`heads % block_H != 0` means the candidate does not apply, so the next one gets a turn."""
    (_, chosen), _, _ = _driver(
        {"requested": AssertionError("heads (48) should be a multiple of block_H (32)")},
        {"fits": 51184},
        ["fits"],
    )
    assert chosen == "fits"


@pytest.mark.parametrize("marker", tiling.RETRYABLE_BUILD_ERRORS)
def test_every_declared_marker_is_recognised(marker):
    """Recognition only labels a log line; the search runs either way."""
    assert tiling.is_retryable_build_error(RuntimeError(f"... {marker} ..."))


def test_the_two_markers_without_python_source_are_still_declared():
    """ "Divide by zero" and "is_scalar" come from tilelang's C++ passes, so no canary is possible."""
    assert set(tiling.RETRYABLE_BUILD_ERRORS) - {tiling.SHARED_MEMORY_EXCEEDED} == {
        tiling.WARP_PARTITION_DEGENERATE,
        tiling.VECTORIZE_FAILED,
    }


# ---------------------------------------------------------------------------------------------
# What the derivation offers, pinned against what gfx942 was validated on
# ---------------------------------------------------------------------------------------------


def test_the_forward_derivation_offers_the_validated_tiling_at_64_heads():
    """CP=8 on one node forces TP=1, so every rank holds all 64 heads.

    tilelang reports 51184 B for this tiling, the first offered candidate that fits 64 KiB.
    """
    offered = list(tiling.sparse_mla_forward_tilings(padded_heads=64, topk=512, block_I=64, limits=GFX942))
    assert tiling.ForwardTiling(num_stages=1, block_I=16, threads=64, block_H=32) in offered


def test_the_forward_derivation_offers_the_validated_tiling_at_32_heads():
    """CP=4 / TP=2. block_H stays None because the kernel's own default already blocks at 32."""
    offered = list(tiling.sparse_mla_forward_tilings(padded_heads=32, topk=512, block_I=64, limits=GFX942))
    assert tiling.ForwardTiling(num_stages=1, block_I=16, threads=64, block_H=None) in offered


def test_the_forward_derivation_shrinks_the_kv_block_before_the_head_block():
    """Head blocks cost grid blocks; KV blocks cost inner iterations. Order encodes that."""
    offered = list(tiling.sparse_mla_forward_tilings(padded_heads=64, topk=512, block_I=64, limits=GFX942))
    head_blocks = [c.block_H for c in offered]
    assert head_blocks == sorted(head_blocks, key=lambda b: -(b or 64)), head_blocks


def test_the_backward_derivation_prefers_keeping_dq_staged():
    """tilelang reuses dQ_shared, so staging dQ costs 480 B, not the 16 KiB the buffer suggests.

    Keeping it saves a pass of uncoalesced global writes, so it is offered first (52704 B of 65536).
    """
    offered = list(tiling.sparse_mla_backward_tilings(padded_heads=64, topk=512, block_size=32, limits=GFX942))
    staged = tiling.BackwardTiling(
        block_size=16, threads=64, split_store=16, stage_dq_through_shared=True, max_block_H=16
    )
    unstaged = tiling.BackwardTiling(
        block_size=16, threads=64, split_store=16, stage_dq_through_shared=False, max_block_H=16
    )
    assert offered.index(staged) < offered.index(unstaged)


def test_the_indexer_derivation_halves_from_the_request():
    assert list(tiling.indexer_forward_block_ns(block_N=256, limits=GFX942)) == [128, 64, 32, 16]


def test_candidates_that_do_not_divide_topk_are_never_offered():
    """topk % block_I != 0 would trip sparse_mqa_fwd's own assert, so those never get measured."""
    offered = list(tiling.sparse_mla_forward_tilings(padded_heads=64, topk=48, block_I=64, limits=GFX942))
    assert offered, "the generator must still offer something for an odd topk"
    assert all(48 % c.block_I == 0 for c in offered)


# ---------------------------------------------------------------------------------------------
# Memoization
# ---------------------------------------------------------------------------------------------


def test_the_fitted_tiling_is_reused_without_searching_again(monkeypatch):
    """The search runs at most once per shape.

    Counting builds cannot show this: a request that compiles takes one build either way. What the
    memo removes is the approach to the search, so the target lookup that begins it is counted.
    """
    built, targets = [], []
    monkeypatch.setattr(sparse_mla_fwd, "_fitted_tiling", {})
    monkeypatch.setattr(sparse_mla_fwd, "sparse_mqa_fwd", lambda *a, **k: built.append(k) or "kernel")
    monkeypatch.setattr(sparse_mla_fwd, "current_target", lambda: targets.append(1) or "target")
    monkeypatch.setattr(sparse_mla_fwd.DeviceLimits, "from_target", staticmethod(lambda _: GFX942))

    request = dict(heads=64, dim=512, topk=512, sm_scale=0.044, block_I=64, num_stages=2, threads=256)
    sparse_mla_fwd._compile_within_shared_mem(**request)
    sparse_mla_fwd._compile_within_shared_mem(**request)

    assert len(built) == 2, "one build per call"
    assert built[0] == built[1]
    assert len(targets) == 1, "the second call must not reach the search at all"


def test_the_backward_search_is_not_repeated_for_every_sequence_length(monkeypatch):
    """Every shared buffer is sized from block_H, block_size and D, so B/S/S_kv cannot move it.

    Keying the memo on the shape re-runs the search for every new sequence length, and THD training
    produces a new one per microbatch.
    """
    built, targets = [], []
    monkeypatch.setattr(sparse_mla_bwd, "_fitted_tiling", {})
    monkeypatch.setattr(sparse_mla_bwd, "bwd", lambda *a, **k: built.append(k) or "kernel")
    monkeypatch.setattr(sparse_mla_bwd, "current_target", lambda: targets.append(1) or "target")
    monkeypatch.setattr(sparse_mla_bwd.DeviceLimits, "from_target", staticmethod(lambda _: GFX942))

    for seq_len in (2048, 4096, 8192):
        sparse_mla_bwd.bwd_within_shared_mem(1, seq_len, seq_len, 64, 512, 512, 0.044)

    assert len(built) == 3, "one build per call"
    assert len(targets) == 1, "the shape changed, the tiling question did not"
