import megatron.core.parallel_state as ps
import torch
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)


def cp_world():
    """(cp_size, cp_rank, cp_group) — cp_size 1 means no context parallelism."""
    cp = ps.get_context_parallel_world_size()
    if cp <= 1:
        return 1, 0, None
    return cp, ps.get_context_parallel_rank(), ps.get_context_parallel_group()


def cp_all_gather(x, group, world):
    """all-gather [s, ...] -> [s*world, ...] in CP-rank order. Assumes CONTIGUOUS CP sharding
    (rank r holds tokens [r*s,(r+1)*s)) -> REQUIRES --allgather-cp (miles' default CP layout is
    zigzag/load-balanced; the provider asserts allgather_cp when cp>1)."""
    xs = [torch.empty_like(x) for _ in range(world)]
    torch.distributed.all_gather(xs, x.contiguous(), group=group)
    return torch.cat(xs, dim=0)


def sp_residual_conv(config, conv, x_sbh, seqlens):
    """Residual depthwise sconv on [s,b,h]. Under SP/CP the sequence is sharded, so a local-shard
    causal conv misses the left context -> gather -> conv on the full sequence -> take this rank's
    slice. Exact because the conv is residual. seqlens must be the FULL-sequence segment lengths."""
    sp = getattr(config, "sequence_parallel", False) and ps.get_tensor_model_parallel_world_size() > 1
    cp, cp_rank, cp_group = cp_world()
    # tensor_parallel_output_grad=False: backward must SPLIT the grad, not reduce-scatter.
    x = gather_from_sequence_parallel_region(x_sbh, tensor_parallel_output_grad=False) if sp else x_sbh
    if cp > 1:
        x = cp_all_gather(x, cp_group, cp)  # [s_full, b, h]
    s, b, h = x.shape
    x = conv(x.reshape(s * b, h), seqlens).reshape(s, b, h)
    if cp > 1:
        sloc = s // cp
        x = x[cp_rank * sloc : (cp_rank + 1) * sloc]
    return scatter_to_sequence_parallel_region(x) if sp else x


def seqlens_from_packed(packed_seq_params, T):
    """THD packing: per-segment token lengths (over the full post-SP-gather length T) from
    cu_seqlens_q, so attention/sconv/rel-bias never cross packed-sequence boundaries. Clip to T:
    keep whole segments, split the boundary one, trailing pad as its own segment."""
    if packed_seq_params is None or getattr(packed_seq_params, "cu_seqlens_q", None) is None:
        return None
    cu = packed_seq_params.cu_seqlens_q
    raw = [int(s) for s in (cu[1:] - cu[:-1]).tolist() if s > 0]
    seqlens, acc = [], 0
    for s in raw:
        if acc + s <= T:
            seqlens.append(s)
            acc += s
        else:
            if T - acc > 0:
                seqlens.append(T - acc)
            acc = T
            break
    if acc < T:  # trailing pad tokens form their own segment
        seqlens.append(T - acc)
    return seqlens
