"""Distributed test for the load-balanced DeepSeek-V4 CSA indexer under contiguous CP.

Run with:
    torchrun --nproc_per_node=2 tests/fast-gpu/test_dsv4_indexer_cp_balance.py
    torchrun --nproc_per_node=4 tests/fast-gpu/test_dsv4_indexer_cp_balance.py

Each rank runs V4Indexer.forward's two steps around the key gather, start_row_exchange and
topk_for_local_rows, once keeping its rows and once balancing them with a collective on the CP group
while the rows are in flight. The balanced picks must equal the local ones, bit for bit for the torch
top-k and as sets for flashinfer; unpacked, the local ones must also equal the pre-balancing
indexer's (tests/fast/test_dsv4_thd.py pins the THD bounds to running each sample alone). Cases:
unpacked batch 1 and 2, THD packs with a long document and with odd scored-row counts, a pack of
tiny documents that leaves a CP4 rank nothing to score, and a pack of equal short documents, which
must skip the exchange.
"""

import os
import sys

import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cuda_ci

from miles.kernels.attention.dsa import causal_ranges_compressed, indexer_logits_sbhd
from miles.kernels.attention.dsa.topk import get_dsa_topk_fn
from miles_plugins.models.deepseek_v4.ops.cp_row_balance import RowExchange
from miles_plugins.models.deepseek_v4.ops.thd_utils import ThdLayout, compressed_cu_seqlens
from miles_plugins.models.deepseek_v4.ops.v4_indexer import start_row_exchange, topk_for_local_rows

register_cuda_ci(
    est_time=60,
    suite="stage-c-4-gpu-h200",
    labels=["precision", "megatron"],
    hardware=["hopper", "blackwell"],
)

SEQLEN_GLOBAL = 16384
RATIO = 4
HEADS, INDEX_DIM, TOPK = 64, 128, 512

# flashinfer's default top-k breaks exact score ties differently from call to call, so two runs of
# the same rows can disagree now and then; its deterministic mode (read at call time) does not.
os.environ["SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC"] = "1"


def setup_dist():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size


def _inputs(rank, rank_rows, n_kv, bsz):
    """Keys are all-gathered in the real layer, so every rank builds the same global key block."""
    generator = torch.Generator(device="cuda").manual_seed(1234)
    q = torch.randn(SEQLEN_GLOBAL, bsz, HEADS, INDEX_DIM, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(n_kv, bsz, INDEX_DIM, device="cuda", dtype=torch.bfloat16, generator=generator)
    weights = torch.randn(SEQLEN_GLOBAL, bsz, HEADS, device="cuda", dtype=torch.float32, generator=generator)
    local = slice(rank * rank_rows, (rank + 1) * rank_rows)
    return q[local].contiguous(), k, weights[local].contiguous()


def _thd_layout(seq_lens, rank, rank_rows):
    cu_seqlens = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    layout = ThdLayout(
        cu_seqlens=cu_seqlens, seq_lens=tuple(seq_lens), global_start=rank * rank_rows, max_seqlen=max(seq_lens)
    )
    layout.cu_seqlens_compressed = compressed_cu_seqlens(cu_seqlens, RATIO)
    return layout


def _unbalanced_unpacked_topk(q, k, weights, rank, topk_fn):
    """The unpacked indexer before balancing: this rank's own rows, bounds sliced from the contiguous run."""
    rank_rows = q.shape[0]
    cu_ks, cu_ke = causal_ranges_compressed(SEQLEN_GLOBAL, RATIO, q.device)
    cu_ks, cu_ke = (
        cu_ks[rank * rank_rows : (rank + 1) * rank_rows],
        cu_ke[rank * rank_rows : (rank + 1) * rank_rows],
    )
    scores = indexer_logits_sbhd(q, k, weights, cu_ks, cu_ke)
    bsz, rows, n_kv = scores.shape
    return topk_fn(scores.reshape(bsz * rows, n_kv), min(TOPK, n_kv)).reshape(bsz, rows, -1)


def _same_picks(got, expected, topk_backend):
    if topk_backend == "torch":
        return torch.equal(got, expected)
    # flashinfer returns each row's picks unsorted
    return torch.equal(got.sort(dim=-1).values, expected.sort(dim=-1).values)


def check_picks(rank, world_size, topk_backend, thd_seq_lens=None, bsz=1):
    """Returns (the balanced picks equal the local ones, and unpacked the pre-balancing ones; the exchange ran)."""
    rank_rows = SEQLEN_GLOBAL // world_size
    thd_layout = _thd_layout(thd_seq_lens, rank, rank_rows) if thd_seq_lens else None
    n_kv = int(thd_layout.cu_seqlens_compressed[-1]) if thd_layout else SEQLEN_GLOBAL // RATIO
    q, k, weights = _inputs(rank, rank_rows, n_kv, bsz)
    topk_fn = get_dsa_topk_fn(topk_backend)
    options = dict(compress_ratio=RATIO, index_topk=TOPK, topk_fn=topk_fn)
    group = dist.group.WORLD

    kept = start_row_exchange(q, weights, thd_layout, group, balance=False)
    local = topk_for_local_rows(kept, k, thd_layout, **options)
    exchange = start_row_exchange(q, weights, thd_layout, group, balance=True)
    # in forward the compressor's CP all-gathers queue on this communicator behind the exchange
    dist.all_reduce(torch.ones(1, device="cuda"), group=group)
    balanced = topk_for_local_rows(exchange, k, thd_layout, **options)

    picks_equal = _same_picks(balanced, local, topk_backend)
    if thd_layout is None:
        expected = _unbalanced_unpacked_topk(q, k, weights, rank, topk_fn)
        picks_equal = picks_equal and _same_picks(local, expected, topk_backend)
    return picks_equal, isinstance(exchange, RowExchange)


def _topk_backends():
    backends = ["torch"]
    try:
        get_dsa_topk_fn("flashinfer")(torch.randn(2, 1024, device="cuda"), 8)
        backends.append("flashinfer")
    except ImportError:
        pass
    return backends


def main():
    rank, world_size = setup_dist()
    try:
        long_doc = SEQLEN_GLOBAL - 4 * 1000 - 116
        cases = {
            "unpacked sequence": dict(thd_seq_lens=None, bsz=1, expect_exchange=True),
            "unpacked batch of 2": dict(thd_seq_lens=None, bsz=2, expect_exchange=True),
            "pack: long doc + short docs + pad": dict(
                thd_seq_lens=[long_doc, 1000, 1000, 1000, 1000, 116], bsz=1, expect_exchange=True
            ),
            # ranks score unequal, odd row counts, so the scorer's last 2-row block runs past the end
            "pack: odd scored-row counts": dict(thd_seq_lens=[SEQLEN_GLOBAL - 2, 1, 1], bsz=1, expect_exchange=True),
            "pack: equal short docs": dict(thd_seq_lens=[512] * (SEQLEN_GLOBAL // 512), bsz=1, expect_exchange=False),
            # documents shorter than 2cp: at CP4 rank 1 scores nothing; at CP8 the gate keeps the rows local
            "pack: tiny docs": dict(
                thd_seq_lens=[1] * 7520 + [2] * 2432 + [5] * 800, bsz=1, expect_exchange=world_size in (2, 4)
            ),
        }
        passed = True
        for topk_backend in _topk_backends():
            for name, case in cases.items():
                picks_equal, exchanged = check_picks(
                    rank, world_size, topk_backend, thd_seq_lens=case["thd_seq_lens"], bsz=case["bsz"]
                )
                flags = torch.tensor(
                    [picks_equal, exchanged == case["expect_exchange"]], device="cuda", dtype=torch.int32
                )
                dist.all_reduce(flags, op=dist.ReduceOp.MIN)
                ok = bool(flags.all())
                passed = passed and ok
                if rank == 0:
                    print(
                        f"CP={world_size} {topk_backend:10s} {name:36s} picks equal: {bool(flags[0])}  "
                        f"exchange as expected: {bool(flags[1])}"
                    )
        if rank == 0:
            print(f"\nCP={world_size} test PASSED!" if passed else "FAILED!")
        if not passed:
            sys.exit(1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Self-bootstrap under torchrun when run as `python3 file.py`, the CUDA CI runner's mode.
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=4", __file__])
    main()
