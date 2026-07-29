from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, suite="stage-c-4-gpu-h200", labels=[])

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import AllreduceOptions, ReduceScatterOptions, _coalescing_manager

from miles.utils.test_utils.det_process_group import (
    DET_NCCL_BACKEND_NAME,
    _CompletedWork,
    _fold_gathered_sum,
    _reduce_op_of,
    det_all_reduce,
    register_det_nccl_backend,
)

_WORLD_SIZE = 4
_NUMEL = 1_048_576
_SEED = 1234


# --------------------------------------------------------------------------- #
# CPU-only tests (no GPU, no distributed init)
# --------------------------------------------------------------------------- #


class _FakeFlatGroup:
    """CPU stand-in with the c10d backend gather (_allgather_base); serves ascending chunk requests."""

    def __init__(self, per_rank: list[torch.Tensor]) -> None:
        self._per_rank = per_rank
        self._offset = 0

    def size(self) -> int:
        return len(self._per_rank)

    def _allgather_base(self, output: torch.Tensor, input: torch.Tensor, opts: object) -> _CompletedWork:
        count = input.numel()
        rows = output.view(self.size(), count)
        for row, src in zip(rows, self._per_rank, strict=True):
            row.copy_(src.reshape(-1)[self._offset : self._offset + count])
        self._offset += count
        return _CompletedWork()


class _FakeTorchftGroup(dist.ProcessGroup):
    """CPU stand-in mirroring torchft: a ProcessGroup subclass overriding only the list-form allgather
    (so it inherits a broken ``_allgather_base`` from the C++ base, exactly like the real wrappers);
    serves ascending chunk requests."""

    def __init__(self, per_rank: list[torch.Tensor]) -> None:
        super().__init__(0, len(per_rank))
        self._per_rank = per_rank
        self._offset = 0

    def allgather(
        self, output_lists: list[list[torch.Tensor]], input_list: list[torch.Tensor], opts: object
    ) -> _CompletedWork:
        count = input_list[0].numel()
        for row, src in zip(output_lists[0], self._per_rank, strict=True):
            row.copy_(src.reshape(-1)[self._offset : self._offset + count])
        self._offset += count
        return _CompletedWork()


def _pairwise_tree_fold(partials: list[torch.Tensor]) -> torch.Tensor:
    """Inline reference fold (pairwise tree for power-of-two): independent of the module."""
    running = list(partials)
    while len(running) > 1:
        running = [running[i] + running[i + 1] for i in range(0, len(running), 2)]
    return running[0]


def test_det_nccl_backend_name_constant_value():
    """DET_NCCL_BACKEND_NAME is the literal "det_nccl" the backend registers and reports under."""
    assert DET_NCCL_BACKEND_NAME == "det_nccl"


def test_reduceop_equality_vs_containment_footgun():
    """Documents why dispatch uses explicit ==: an options ReduceOp equals SUM yet tuple containment is False."""
    opts = dist.AllreduceOptions()
    opts.reduceOp = dist.ReduceOp.SUM

    assert opts.reduceOp == dist.ReduceOp.SUM
    assert opts.reduceOp not in (dist.ReduceOp.SUM, dist.ReduceOp.AVG)


@pytest.mark.parametrize(
    "parts,dtype,expected",
    [
        # Single rank / order-free sanity.
        ([3.5], torch.float64, 3.5),
        ([1.0, 1e-5], torch.float64, 1.0 + 1e-5),
        ([1, 2, 3, 4], torch.int64, 10),
        # One 2**-53 is absorbed by 1.0 (ties-to-even), but 2**-53 + 2**-53 = ulp(1.0)
        # is not -- so the pairwise tree and a sequential fold give different bits.
        # 4 ranks must be the tree (a+b)+(c+d), not sequential (which gives 1.0):
        ([0.5, 0.5, 2**-53, 2**-53], torch.float64, (0.5 + 0.5) + (2**-53 + 2**-53)),
        # 3 ranks (non-power-of-two) must be the ascending fold ((a+b)+c), which
        # absorbs both halves (a wrong pairing would give 1.0 + 2**-52):
        ([1.0, 2**-53, 2**-53], torch.float64, (1.0 + 2**-53) + 2**-53),
        # 5 ranks: ascending fold absorbs every half:
        ([1.0, 2**-53, 2**-53, 2**-53, 2**-53], torch.float64, 1.0),
        # 8 ranks: the full three-level tree cancels exactly to 0.0 (sequential
        # would leave -2**-52):
        (
            [0.5, 0.5, 2**-53, 2**-53, -0.5, -0.5, -(2**-53), -(2**-53)],
            torch.float64,
            ((0.5 + 0.5) + (2**-53 + 2**-53)) + ((-0.5 + -0.5) + (-(2**-53) + -(2**-53))),
        ),
        # Same construction at float32 precision (ulp(1.0) = 2**-23):
        ([0.5, 0.5, 2**-24, 2**-24], torch.float32, (0.5 + 0.5) + (2**-24 + 2**-24)),
    ],
)
def test_fold_gathered_sum(parts: list[float], dtype: torch.dtype, expected: float):
    """The fold's exact bracketing is pinned by hand-computed literals where any other order changes the bits."""
    tensors = [torch.tensor([value], dtype=dtype) for value in parts]

    actual = _fold_gathered_sum(tensors)

    assert actual.item() == expected


def test_reduce_op_of_extracts_reduceop_from_options_object():
    """_reduce_op_of reads .reduceOp from an options object and passes a bare ReduceOp through."""
    ar_opts = AllreduceOptions()
    ar_opts.reduceOp = dist.ReduceOp.SUM
    assert _reduce_op_of(ar_opts) == dist.ReduceOp.SUM

    rs_opts = ReduceScatterOptions()
    rs_opts.reduceOp = dist.ReduceOp.AVG
    assert _reduce_op_of(rs_opts) == dist.ReduceOp.AVG

    assert _reduce_op_of(dist.ReduceOp.MAX) == dist.ReduceOp.MAX


def test_det_all_reduce_equals_manual_pairwise_tree_fold():
    """det_all_reduce over a fake group equals the manual pairwise-tree fold bitwise."""
    gen = torch.Generator().manual_seed(_SEED)
    per_rank = [torch.randn(16, generator=gen, dtype=torch.float32) for _ in range(_WORLD_SIZE)]
    expected = _pairwise_tree_fold(per_rank)

    tensor = per_rank[0].clone()
    det_all_reduce(tensor, group=_FakeFlatGroup(per_rank))
    assert torch.equal(tensor, expected)


def test_det_all_reduce_avg_equals_tree_fold_divided_by_world():
    """det_all_reduce with reduce_op=AVG equals the pairwise-tree fold divided by world_size bitwise."""
    gen = torch.Generator().manual_seed(_SEED + 31)
    per_rank = [torch.randn(16, generator=gen, dtype=torch.float32) for _ in range(_WORLD_SIZE)]
    expected = _pairwise_tree_fold(per_rank) / _WORLD_SIZE

    tensor = per_rank[0].clone()
    det_all_reduce(tensor, group=_FakeFlatGroup(per_rank), reduce_op=dist.ReduceOp.AVG)
    assert torch.equal(tensor, expected)


def test_det_all_reduce_avg_non_contiguous_recursion_passes_reduce_op_through():
    """A non-contiguous input with reduce_op=AVG recurses with the op preserved, writing tree/world back."""
    gen = torch.Generator().manual_seed(_SEED + 32)
    per_rank = [torch.randn(8 * 4, generator=gen, dtype=torch.float32) for _ in range(_WORLD_SIZE)]
    expected_flat = _pairwise_tree_fold(per_rank) / _WORLD_SIZE

    base = per_rank[0].reshape(4, 8).clone()
    non_contiguous = base.t()
    assert not non_contiguous.is_contiguous()

    det_all_reduce(non_contiguous, group=_FakeFlatGroup(per_rank), reduce_op=dist.ReduceOp.AVG)
    assert torch.equal(non_contiguous.contiguous().reshape(-1), expected_flat)


def test_det_all_reduce_multi_chunk_matches_single_chunk(monkeypatch: pytest.MonkeyPatch):
    """A tiny gather-buffer cap (forcing many chunks) gives bitwise-identical results to one chunk."""
    import miles.utils.test_utils.det_process_group as dpg

    gen = torch.Generator().manual_seed(_SEED + 21)
    per_rank = [torch.randn(50, generator=gen, dtype=torch.float32) for _ in range(_WORLD_SIZE)]
    expected = _pairwise_tree_fold(per_rank)

    monkeypatch.setattr(dpg, "_GATHER_BUFFER_CAP_BYTES", _WORLD_SIZE * 7 * 4)
    tensor = per_rank[0].clone()
    det_all_reduce(tensor, group=_FakeFlatGroup(per_rank))

    assert torch.equal(tensor, expected)


def test_det_reduce_scatter_multi_chunk_slice_matches_full_fold(monkeypatch: pytest.MonkeyPatch):
    """det_reduce_scatter under a tiny chunk cap writes exactly this rank's slice of the full fold."""
    import miles.utils.test_utils.det_process_group as dpg

    gen = torch.Generator().manual_seed(_SEED + 22)
    per_rank = [torch.randn(48, generator=gen, dtype=torch.float32) for _ in range(_WORLD_SIZE)]
    expected_full = _pairwise_tree_fold(per_rank)

    monkeypatch.setattr(dpg, "_GATHER_BUFFER_CAP_BYTES", _WORLD_SIZE * 7 * 4)
    rank = 1
    out = torch.empty(12, dtype=torch.float32)
    dpg.det_reduce_scatter(out, per_rank[0].clone(), group=_FakeFlatGroup(per_rank), rank=rank, world_size=_WORLD_SIZE)

    assert torch.equal(out, expected_full[12:24])


def test_det_all_reduce_torchft_list_gather_matches_flat_gather_bitwise():
    """The torchft list-form gather path is bitwise-identical to the _allgather_base path."""
    gen = torch.Generator().manual_seed(_SEED + 7)
    per_rank = [torch.randn(32, generator=gen, dtype=torch.float32) for _ in range(_WORLD_SIZE)]

    via_flat = per_rank[0].clone()
    det_all_reduce(via_flat, group=_FakeFlatGroup(per_rank))

    via_list = per_rank[0].clone()
    det_all_reduce(via_list, group=_FakeTorchftGroup(per_rank))

    assert torch.equal(via_flat, via_list)
    assert torch.equal(via_flat, _pairwise_tree_fold(per_rank))


def test_gather_into_routes_process_group_instances_to_list_form_allgather():
    """Regression: ProcessGroup subclasses inherit _allgather_base from the C++ base (hasattr is
    always True), so routing must key on isinstance and take the overridden list-form allgather."""
    import miles.utils.test_utils.det_process_group as dpg

    gen = torch.Generator().manual_seed(_SEED + 23)
    per_rank = [torch.randn(8, generator=gen, dtype=torch.float32) for _ in range(_WORLD_SIZE)]
    group = _FakeTorchftGroup(per_rank)
    assert hasattr(group, "_allgather_base")

    out = torch.empty(_WORLD_SIZE * 8, dtype=torch.float32)
    dpg._gather_into(group, out, per_rank[0].clone())

    assert torch.equal(out.view(_WORLD_SIZE, 8), torch.stack(per_rank))


def test_det_all_reduce_non_contiguous_input_writes_summed_values_back():
    """A non-contiguous (.t() view) input gets the correct summed values written back."""
    gen = torch.Generator().manual_seed(_SEED + 11)
    per_rank = [torch.randn(8 * 4, generator=gen, dtype=torch.float32) for _ in range(_WORLD_SIZE)]
    expected_flat = _pairwise_tree_fold(per_rank)

    base = per_rank[0].reshape(4, 8).clone()
    non_contiguous = base.t()
    assert not non_contiguous.is_contiguous()

    det_all_reduce(non_contiguous, group=_FakeFlatGroup(per_rank))
    assert torch.equal(non_contiguous.contiguous().reshape(-1), expected_flat)


def test_det_all_reduce_world_size_one_leaves_tensor_unchanged():
    """A 1-rank group gathers only the local copy, so the tensor is unchanged."""
    original = torch.tensor([1.0, -2.5, 3.25, 0.0], dtype=torch.float32)
    tensor = original.clone()
    det_all_reduce(tensor, group=_FakeFlatGroup([original]))
    assert torch.equal(tensor, original)


def test_det_all_reduce_fold_order_checksum_pin():
    """Pin exact fold result so an accidental fold-order change (tree->linear) fails loudly."""
    per_rank = [
        torch.tensor([1.0, 1e8, -1e8, 0.25], dtype=torch.float32),
        torch.tensor([2.0, -1e8, 1e8, 0.25], dtype=torch.float32),
        torch.tensor([3.0, 1e8, -1e8, 0.25], dtype=torch.float32),
        torch.tensor([4.0, -1e8, 1e8, 0.25], dtype=torch.float32),
    ]
    reference = _pairwise_tree_fold(per_rank)

    tensor = per_rank[0].clone()
    det_all_reduce(tensor, group=_FakeFlatGroup(per_rank))
    assert torch.equal(tensor, reference)

    # Hardcoded pin: element 0 is the plain sum; element 3 sums four 0.25 -> 1.0.
    assert torch.equal(tensor, torch.tensor([10.0, 0.0, 0.0, 1.0], dtype=torch.float32))
    assert tensor[0].item().hex() == (10.0).hex()
    assert tensor[3].item().hex() == (1.0).hex()


def test_completed_work_future_wait_returns_result():
    """_CompletedWork().get_future().wait() returns the (None) result without blocking."""
    work = _CompletedWork()
    assert work.wait() is True
    assert work.get_future().wait() is None


def _order_sensitive_input(rank: int, seed: int = _SEED) -> torch.Tensor:
    """Per-rank input whose cross-rank sum catastrophically cancels (~1e-4 from +-0.5)."""
    shared = torch.randn(_NUMEL, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
    own = torch.randn(_NUMEL, generator=torch.Generator().manual_seed(seed + 1 + rank), dtype=torch.float32)
    sign = -1.0 if rank % 2 else 1.0
    return (sign * 0.5 * shared + 1e-4 * own).cuda()


def _manual_tree_sum(partials: list[torch.Tensor]) -> torch.Tensor:
    running = list(partials)
    while len(running) > 1:
        running = [running[i] + running[i + 1] for i in range(0, len(running), 2)]
    return running[0]


def _fixed_tree_reference(x: torch.Tensor) -> torch.Tensor:
    """Gather every rank's tensor (data movement only) and fold in the fixed tree order."""
    gathered = [torch.empty_like(x) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered, x)
    return _manual_tree_sum(gathered)


def _assert_bitwise(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if torch.equal(actual, expected):
        return
    mismatch = int((actual != expected).sum().item())
    max_abs = float((actual - expected).abs().max().item())
    raise AssertionError(f"{name}: mismatch_elems={mismatch}/{actual.numel()} max_abs={max_abs:.3e}")


def _shard_of(full: torch.Tensor, rank: int) -> torch.Tensor:
    shard_numel = full.numel() // _WORLD_SIZE
    return full[rank * shard_numel : (rank + 1) * shard_numel]


def _check_allreduce(rank: int, det1: dist.ProcessGroup, det2: dist.ProcessGroup, x, tree) -> None:
    a = x.clone()
    dist.all_reduce(a, op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("allreduce SUM vs fixed tree", a, tree)

    b = x.clone()
    dist.all_reduce(b, op=dist.ReduceOp.SUM, group=det2)
    _assert_bitwise("allreduce bitwise across communicator instances", b, tree)

    averaged = x.clone()
    dist.all_reduce(averaged, op=dist.ReduceOp.AVG, group=det1)
    _assert_bitwise("allreduce AVG == SUM/world", averaged, tree / _WORLD_SIZE)

    max_det = x.clone()
    dist.all_reduce(max_det, op=dist.ReduceOp.MAX, group=det1)
    max_native = x.clone()
    dist.all_reduce(max_native, op=dist.ReduceOp.MAX)
    _assert_bitwise("allreduce MAX delegates to native", max_det, max_native)


def _check_reduce_scatter_vs_allreduce(rank: int, det1: dist.ProcessGroup, x, tree) -> None:
    expected_shard = _shard_of(tree, rank)

    rs = torch.empty_like(expected_shard)
    dist.reduce_scatter_tensor(rs, x.clone(), op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("reduce_scatter_tensor == slice of allreduce", rs, expected_shard)

    # Megatron distributed-optimizer style: the output shard is a view of the input.
    buf = x.clone()
    shard_view = buf.view(_WORLD_SIZE, -1)[rank]
    dist.reduce_scatter_tensor(shard_view, buf, op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("aliased reduce_scatter_tensor", shard_view, expected_shard.view(shard_view.shape))

    inputs = [chunk.contiguous() for chunk in x.clone().chunk(_WORLD_SIZE)]
    out = torch.empty_like(expected_shard)
    dist.reduce_scatter(out, inputs, op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("reduce_scatter (list) == slice of allreduce", out, expected_shard)


def _check_uneven_reduce_scatter(rank: int, det1: dist.ProcessGroup) -> None:
    """List-form reduce_scatter with uneven slot sizes folds each slot at its true offset."""
    device = torch.device("cuda", torch.cuda.current_device())
    slot_sizes = [3, 5, 7, 9]
    gen = torch.Generator().manual_seed(_SEED + 300 + rank)
    inputs = [torch.randn(size, generator=gen, dtype=torch.float32).to(device) for size in slot_sizes]

    gathered_inputs: list[list[torch.Tensor]] = []
    for slot, size in enumerate(slot_sizes):
        slot_copies = [torch.empty(size, device=device) for _ in range(_WORLD_SIZE)]
        dist.all_gather(slot_copies, inputs[slot])
        gathered_inputs.append(slot_copies)
    expected = _manual_tree_sum(gathered_inputs[rank])

    out = torch.empty(slot_sizes[rank], device=device)
    dist.reduce_scatter(out, inputs, op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("uneven reduce_scatter (list)", out, expected)


def _expected_slot_fold(rank: int, inputs: list[torch.Tensor]) -> torch.Tensor:
    """Reference for list reduce_scatter: gather every rank's copy of MY slot and tree-fold.

    Gathers slot by slot: within one all_gather every rank contributes the SAME slot,
    so shapes match across ranks (slots have per-rank-distinct shapes, and an uneven
    all_gather is undefined over NCCL - it deadlocks). Returned flat, matching how
    callers compare (their outputs are flattened views)."""
    my_slot_copies: list[torch.Tensor] = []
    for slot in range(_WORLD_SIZE):
        slot_input = inputs[slot].contiguous()
        slot_copies = [torch.empty_like(slot_input) for _ in range(_WORLD_SIZE)]
        dist.all_gather(slot_copies, slot_input)
        if slot == rank:
            my_slot_copies = slot_copies
    return _manual_tree_sum(my_slot_copies).reshape(-1)


def _check_uneven_reduce_scatter_shapes(rank: int, det1: dist.ProcessGroup) -> None:
    """List reduce_scatter with per-slot distinct multi-dim shapes, non-contiguous slots/output,
    bf16, and a forced multi-chunk fold all match the per-slot tree fold bitwise."""
    import miles.utils.test_utils.det_process_group as dpg

    device = torch.device("cuda", torch.cuda.current_device())
    slot_shapes = [(2, 3), (5,), (4, 2), (3, 3)]

    def make_inputs(seed: int, dtype: torch.dtype) -> list[torch.Tensor]:
        gen = torch.Generator().manual_seed(seed + 17 * rank)
        return [torch.randn(shape, generator=gen, dtype=dtype).to(device) for shape in slot_shapes]

    # Distinct multi-dim shapes per slot, via the dist API.
    inputs = make_inputs(_SEED + 400, torch.float32)
    expected = _expected_slot_fold(rank, inputs)
    out = torch.empty(slot_shapes[rank], device=device)
    dist.reduce_scatter(out, inputs, op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("uneven multi-dim reduce_scatter", out.view(-1), expected)

    # bf16 variant.
    inputs_bf16 = make_inputs(_SEED + 401, torch.bfloat16)
    expected_bf16 = _expected_slot_fold(rank, inputs_bf16)
    out_bf16 = torch.empty(slot_shapes[rank], dtype=torch.bfloat16, device=device)
    dist.reduce_scatter(out_bf16, inputs_bf16, op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("uneven bf16 reduce_scatter", out_bf16.view(-1), expected_bf16)

    # Non-contiguous slot inputs and a non-contiguous output, via the group method
    # directly (the dist wrapper would densify).
    nc_shapes = [(3, 2), (5, 1), (2, 4), (3, 3)]
    bases = make_inputs(_SEED + 402, torch.float32)
    nc_inputs = [base.reshape(shape).t() for base, shape in zip(bases, nc_shapes, strict=True)]
    assert all(not t.is_contiguous() for t in nc_inputs if t.dim() > 1 and min(t.shape) > 1)
    expected_nc = _expected_slot_fold(rank, nc_inputs)
    out_base = torch.empty(nc_shapes[rank], device=device)
    out_nc = out_base.t()
    opts = dist.ReduceScatterOptions()
    opts.reduceOp = dist.ReduceOp.SUM
    det1.reduce_scatter([out_nc], [nc_inputs], opts).wait()
    _assert_bitwise("uneven non-contiguous reduce_scatter", out_nc.contiguous().view(-1), expected_nc)

    # Forced multi-chunk fold through the real gather path.
    original_cap = dpg._GATHER_BUFFER_CAP_BYTES
    dpg._GATHER_BUFFER_CAP_BYTES = _WORLD_SIZE * 2 * 4
    try:
        inputs_chunked = make_inputs(_SEED + 403, torch.float32)
        expected_chunked = _expected_slot_fold(rank, inputs_chunked)
        out_chunked = torch.empty(slot_shapes[rank], device=device)
        dist.reduce_scatter(out_chunked, inputs_chunked, op=dist.ReduceOp.SUM, group=det1)
    finally:
        dpg._GATHER_BUFFER_CAP_BYTES = original_cap
    _assert_bitwise("uneven multi-chunk reduce_scatter", out_chunked.view(-1), expected_chunked)


def _check_coalescing_manager(rank: int, det1: dist.ProcessGroup, x, tree, x2, tree2) -> None:
    device = torch.device("cuda", torch.cuda.current_device())

    ar1, ar2 = x.clone(), x2.clone()
    with _coalescing_manager(group=det1, device=device):
        dist.all_reduce(ar1, op=dist.ReduceOp.SUM, group=det1)
        dist.all_reduce(ar2, op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("coalescing_manager AR (1st)", ar1, tree)
    _assert_bitwise("coalescing_manager AR (2nd)", ar2, tree2)

    rs1 = torch.empty_like(_shard_of(tree, rank))
    rs2 = torch.empty_like(_shard_of(tree2, rank))
    with _coalescing_manager(group=det1, device=device):
        dist.reduce_scatter_tensor(rs1, x.clone(), op=dist.ReduceOp.SUM, group=det1)
        dist.reduce_scatter_tensor(rs2, x2.clone(), op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("coalescing_manager RS (1st)", rs1, _shard_of(tree, rank))
    _assert_bitwise("coalescing_manager RS (2nd)", rs2, _shard_of(tree2, rank))

    shard_in = torch.full((128,), float(rank), device=device)
    full_out = torch.empty(128 * _WORLD_SIZE, device=device)
    with _coalescing_manager(group=det1, device=device):
        dist.all_gather_into_tensor(full_out, shard_in, group=det1)
    expected = torch.cat([torch.full((128,), float(r), device=device) for r in range(_WORLD_SIZE)])
    _assert_bitwise("coalescing_manager all_gather_into_tensor", full_out, expected)


def _check_non_contiguous(rank: int, det1: dist.ProcessGroup) -> None:
    base = torch.randn(64, 64, generator=torch.Generator().manual_seed(_SEED + 100 + rank)).cuda()
    non_contiguous = base.t()
    assert not non_contiguous.is_contiguous()

    gathered = [torch.empty_like(base) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered, base)
    expected = _manual_tree_sum([g.t().contiguous() for g in gathered])

    dist.all_reduce(non_contiguous, op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("non-contiguous allreduce", non_contiguous.contiguous(), expected)


def _check_delegation(rank: int, det1: dist.ProcessGroup) -> None:
    device = torch.device("cuda", torch.cuda.current_device())

    broadcasted = torch.full((8,), float(rank), device=device)
    dist.broadcast(broadcasted, src=0, group=det1)
    _assert_bitwise("broadcast", broadcasted, torch.zeros(8, device=device))

    piece = torch.full((4,), float(rank), device=device)
    pieces = [torch.empty_like(piece) for _ in range(_WORLD_SIZE)]
    dist.all_gather(pieces, piece, group=det1)
    for source_rank, gathered_piece in enumerate(pieces):
        _assert_bitwise(
            f"all_gather piece {source_rank}", gathered_piece, torch.full((4,), float(source_rank), device=device)
        )

    full = torch.empty(4 * _WORLD_SIZE, device=device)
    dist.all_gather_into_tensor(full, piece, group=det1)
    expected = torch.cat([torch.full((4,), float(r), device=device) for r in range(_WORLD_SIZE)])
    _assert_bitwise("all_gather_into_tensor", full, expected)

    reduced = torch.full((4,), float(rank), device=device)
    dist.reduce(reduced, dst=0, op=dist.ReduceOp.MAX, group=det1)
    if rank == 0:
        _assert_bitwise("reduce MAX to dst", reduced, torch.full((4,), float(_WORLD_SIZE - 1), device=device))

    # rank r receives element r from every rank q: value = r + 10*q at position q
    scatter_in = torch.arange(_WORLD_SIZE, dtype=torch.float32, device=device) + rank * 10
    a2a_out = torch.empty(_WORLD_SIZE, device=device)
    dist.all_to_all_single(a2a_out, scatter_in, group=det1)
    expected_a2a = torch.tensor([float(rank + 10 * q) for q in range(_WORLD_SIZE)], device=device)
    _assert_bitwise("all_to_all_single", a2a_out, expected_a2a)

    peer = rank + 1 if rank % 2 == 0 else rank - 1
    outgoing = torch.full((4,), float(rank), device=device)
    incoming = torch.empty(4, device=device)
    if rank % 2 == 0:
        dist.send(outgoing, dst=peer, group=det1)
        dist.recv(incoming, src=peer, group=det1)
    else:
        dist.recv(incoming, src=peer, group=det1)
        dist.send(outgoing, dst=peer, group=det1)
    _assert_bitwise("send/recv", incoming, torch.full((4,), float(peer), device=device))

    dist.barrier(group=det1)

    assert dist.get_backend(det1) == DET_NCCL_BACKEND_NAME, f"unexpected backend name {dist.get_backend(det1)}"


def _check_batch_isend_irecv_ring(rank: int, det1: dist.ProcessGroup) -> None:
    """Ring batch_isend_irecv over det1 exercises the no-op coalescing hooks with batched p2p."""
    device = torch.device("cuda", torch.cuda.current_device())
    next_rank = (rank + 1) % _WORLD_SIZE
    prev_rank = (rank - 1) % _WORLD_SIZE

    send_tensor = torch.full((16,), float(rank), device=device)
    recv_tensor = torch.empty(16, device=device)
    ops = [
        dist.P2POp(dist.isend, send_tensor, peer=next_rank, group=det1),
        dist.P2POp(dist.irecv, recv_tensor, peer=prev_rank, group=det1),
    ]
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    _assert_bitwise("batch_isend_irecv ring", recv_tensor, torch.full((16,), float(prev_rank), device=device))


def _check_dtype_allreduce(rank: int, det1: dist.ProcessGroup) -> None:
    """bf16 and int64 SUM allreduce over det1 match the manual fold / exact integer sum bitwise."""
    device = torch.device("cuda", torch.cuda.current_device())

    bf16_inputs = [(_order_sensitive_input(r).to(torch.bfloat16)) for r in range(_WORLD_SIZE)]
    bf16_expected = _manual_tree_sum(bf16_inputs)
    bf16_actual = bf16_inputs[rank].clone()
    dist.all_reduce(bf16_actual, op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("bf16 SUM allreduce == bf16 tree fold", bf16_actual, bf16_expected)

    int_value = torch.full((256,), rank + 1, dtype=torch.int64, device=device)
    expected_int = torch.full((256,), sum(range(1, _WORLD_SIZE + 1)), dtype=torch.int64, device=device)
    dist.all_reduce(int_value, op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("int64 SUM allreduce == exact integer sum", int_value, expected_int)


def _check_reduce_scatter_avg(rank: int, det1: dist.ProcessGroup, x: torch.Tensor, tree: torch.Tensor) -> None:
    """AVG reduce_scatter (tensor + list variant) equals this rank's slice of tree/world bitwise."""
    expected_shard = _shard_of(tree, rank) / _WORLD_SIZE

    rs = torch.empty_like(expected_shard)
    dist.reduce_scatter_tensor(rs, x.clone(), op=dist.ReduceOp.AVG, group=det1)
    _assert_bitwise("reduce_scatter_tensor AVG == slice of tree/world", rs, expected_shard)

    inputs = [chunk.contiguous() for chunk in x.clone().chunk(_WORLD_SIZE)]
    out = torch.empty_like(expected_shard)
    dist.reduce_scatter(out, inputs, op=dist.ReduceOp.AVG, group=det1)
    _assert_bitwise("reduce_scatter (list) AVG == slice of tree/world", out, expected_shard)


def _check_reduce_sum_avg_fold(rank: int, det1: dist.ProcessGroup, x: torch.Tensor, tree: torch.Tensor) -> None:
    """dist.reduce SUM/AVG over det1 folds in fixed tree order; the dst rank matches tree (/world)."""
    summed = x.clone()
    dist.reduce(summed, dst=0, op=dist.ReduceOp.SUM, group=det1)
    if rank == 0:
        _assert_bitwise("reduce SUM to dst == tree fold", summed, tree)

    averaged = x.clone()
    dist.reduce(averaged, dst=0, op=dist.ReduceOp.AVG, group=det1)
    if rank == 0:
        _assert_bitwise("reduce AVG to dst == tree/world", averaged, tree / _WORLD_SIZE)


def _check_min_delegation(rank: int, det1: dist.ProcessGroup) -> None:
    """MIN allreduce and MIN reduce over det1 delegate to native NCCL bitwise."""
    device = torch.device("cuda", torch.cuda.current_device())
    operand = torch.full((64,), float(rank + 1), device=device)

    min_det = operand.clone()
    dist.all_reduce(min_det, op=dist.ReduceOp.MIN, group=det1)
    min_native = operand.clone()
    dist.all_reduce(min_native, op=dist.ReduceOp.MIN)
    _assert_bitwise("allreduce MIN delegates to native", min_det, min_native)

    reduce_min = operand.clone()
    dist.reduce(reduce_min, dst=0, op=dist.ReduceOp.MIN, group=det1)
    if rank == 0:
        _assert_bitwise("reduce MIN to dst", reduce_min, torch.ones(64, device=device))


def _check_reduce_scatter_uneven_avg(rank: int, det1: dist.ProcessGroup) -> None:
    """Uneven list-form reduce_scatter with AVG folds each slot at its true offset and divides by world."""
    device = torch.device("cuda", torch.cuda.current_device())
    slot_sizes = [3, 5, 7, 9]
    gen = torch.Generator().manual_seed(_SEED + 400 + rank)
    inputs = [torch.randn(size, generator=gen, dtype=torch.float32).to(device) for size in slot_sizes]

    expected = _expected_slot_fold(rank, inputs) / _WORLD_SIZE

    out = torch.empty(slot_sizes[rank], device=device)
    dist.reduce_scatter(out, inputs, op=dist.ReduceOp.AVG, group=det1)
    _assert_bitwise("uneven reduce_scatter (list) AVG == slot tree/world", out, expected)


def _check_multi_chunk_through_nccl(rank: int, det1: dist.ProcessGroup, x: torch.Tensor, tree: torch.Tensor) -> None:
    """A tiny gather-buffer cap forces multi-chunk gathers through real NCCL; SUM allreduce and
    reduce_scatter_tensor over det1 stay bitwise equal to the single-chunk tree reference."""
    import miles.utils.test_utils.det_process_group as dpg

    original_cap = dpg._GATHER_BUFFER_CAP_BYTES
    try:
        dpg._GATHER_BUFFER_CAP_BYTES = _WORLD_SIZE * 64 * x.element_size()

        ar = x.clone()
        dist.all_reduce(ar, op=dist.ReduceOp.SUM, group=det1)
        _assert_bitwise("multi-chunk allreduce SUM == tree fold", ar, tree)

        expected_shard = _shard_of(tree, rank)
        rs = torch.empty_like(expected_shard)
        dist.reduce_scatter_tensor(rs, x.clone(), op=dist.ReduceOp.SUM, group=det1)
        _assert_bitwise("multi-chunk reduce_scatter_tensor == slice of tree fold", rs, expected_shard)
    finally:
        dpg._GATHER_BUFFER_CAP_BYTES = original_cap


def _check_reduce_scatter_base_uneven_raises(rank: int, det1: dist.ProcessGroup) -> None:
    """Calling det1._reduce_scatter_base directly with an indivisible input numel fails loud."""
    from torch.distributed.distributed_c10d import ReduceScatterOptions

    device = torch.device("cuda", torch.cuda.current_device())
    opts = ReduceScatterOptions()
    opts.reduceOp = dist.ReduceOp.SUM

    output = torch.empty(4, device=device)
    uneven_input = torch.empty(4 * _WORLD_SIZE + 1, device=device)
    with pytest.raises(AssertionError):
        det1._reduce_scatter_base(output, uneven_input, opts)


def _check_object_collectives(rank: int, det1: dist.ProcessGroup) -> None:
    """Object collectives (all_gather_object + broadcast_object_list) delegate correctly over det1."""
    gathered: list[object] = [None] * _WORLD_SIZE
    dist.all_gather_object(gathered, {"rank": rank, "tag": rank * 7}, group=det1)
    assert gathered == [{"rank": r, "tag": r * 7} for r in range(_WORLD_SIZE)], gathered

    payload: list[object] = [{"from": 0, "value": "hello"}] if rank == 0 else [None]
    dist.broadcast_object_list(payload, src=0, group=det1)
    assert payload == [{"from": 0, "value": "hello"}], payload


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    register_det_nccl_backend()
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    det1 = dist.new_group(list(range(world_size)), backend="det_nccl")
    det2 = dist.new_group(list(range(world_size)), backend="det_nccl")

    x = _order_sensitive_input(rank)
    x2 = _order_sensitive_input(rank, seed=_SEED + 50)
    tree = _fixed_tree_reference(x)
    tree2 = _fixed_tree_reference(x2)

    _check_allreduce(rank, det1, det2, x, tree)
    _check_reduce_scatter_vs_allreduce(rank, det1, x, tree)
    _check_uneven_reduce_scatter(rank, det1)
    _check_uneven_reduce_scatter_shapes(rank, det1)
    _check_coalescing_manager(rank, det1, x, tree, x2, tree2)
    _check_non_contiguous(rank, det1)
    _check_delegation(rank, det1)
    _check_batch_isend_irecv_ring(rank, det1)
    _check_dtype_allreduce(rank, det1)
    _check_reduce_scatter_avg(rank, det1, x, tree)
    _check_reduce_sum_avg_fold(rank, det1, x, tree)
    _check_min_delegation(rank, det1)
    _check_reduce_scatter_uneven_avg(rank, det1)
    _check_multi_chunk_through_nccl(rank, det1, x, tree)
    _check_reduce_scatter_base_uneven_raises(rank, det1)
    _check_object_collectives(rank, det1)

    dist.barrier()
    dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def test_det_process_group_multi_gpu():
    """det_nccl backend: bitwise fixed-order SUM/AVG (allreduce + reduce_scatter, incl. under
    the coalescing manager) and faithful delegation of every other collective, on 4 GPUs."""
    if torch.cuda.device_count() < _WORLD_SIZE:
        raise RuntimeError(f"requires {_WORLD_SIZE} GPUs, found {torch.cuda.device_count()}")

    mp.spawn(_worker, args=(_WORLD_SIZE, _free_port()), nprocs=_WORLD_SIZE, join=True)


def _world_backend_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    register_det_nccl_backend()
    dist.init_process_group(backend="det_nccl", rank=rank, world_size=world_size)

    dist.barrier()

    x = _order_sensitive_input(rank)
    tree = _fixed_tree_reference(x)
    ar = x.clone()
    dist.all_reduce(ar, op=dist.ReduceOp.SUM)
    _assert_bitwise("default-group det_nccl allreduce == tree fold", ar, tree)

    assert dist.get_backend() == DET_NCCL_BACKEND_NAME, f"unexpected default backend {dist.get_backend()}"

    dist.barrier()
    dist.destroy_process_group()


def test_det_nccl_as_world_backend_multi_gpu():
    """det_nccl wired as the DEFAULT-group backend (train_actor shape): barrier + bitwise tree SUM."""
    if torch.cuda.device_count() < _WORLD_SIZE:
        raise RuntimeError(f"requires {_WORLD_SIZE} GPUs, found {torch.cuda.device_count()}")

    mp.spawn(_world_backend_worker, args=(_WORLD_SIZE, _free_port()), nprocs=_WORLD_SIZE, join=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
