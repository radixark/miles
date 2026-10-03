"""LinearAttentionLayer's sequence-parallel and context-parallel paths against the replicated KDA reference,
and a TP-sharded checkpoint round trip of its sharded_state_dict.

    torchrun --nproc_per_node=2 tests/fast-gpu/test_linear_attn_layer.py
"""

import os
import sys
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci

from miles.kernels.attention.delta_rule import DeltaRuleHeads

sys.path.insert(0, os.path.dirname(__file__))
from delta_rule_reference import ReplicatedKDA, build_layer, gather, packed, rel_err  # noqa: E402

register_cuda_ci(est_time=120, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

HIDDEN = 256
HEADS = DeltaRuleHeads(num_k_heads=8, num_v_heads=8, head_k_dim=64, head_v_dim=64)
SEQLENS = [256, 256]
TOLERANCE = 2e-2


def _config(tp: int, cp: int, sequence_parallel: bool) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        params_dtype=torch.bfloat16,
        bf16=True,
        tensor_model_parallel_size=tp,
        context_parallel_size=cp,
        sequence_parallel=sequence_parallel,
    )


def _setup(tp: int, cp: int):
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
    model_parallel_cuda_manual_seed(1234)


def _reference_and_input(seed: int):
    torch.manual_seed(seed)
    ref = ReplicatedKDA(HIDDEN, HEADS, torch.bfloat16).cuda()
    x = torch.randn(1, sum(SEQLENS), HIDDEN, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(x)
    cu_seqlens = torch.tensor([0, *torch.tensor(SEQLENS).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    return ref, x, grad, cu_seqlens


def _reference_out_and_dx(ref, x, grad, cu_seqlens):
    x_ref = x.clone().requires_grad_()
    out = ref(x_ref, cu_seqlens)
    out.backward(grad)
    return out.detach()[0], x_ref.grad[0]


def check_sequence_parallel(world: int) -> dict[str, float]:
    """TP = world with sequence parallelism: the layer sees this rank's sequence shard and returns one."""
    _setup(tp=world, cp=1)
    ref, x, grad, cu_seqlens = _reference_and_input(seed=7)
    out_ref, dx_ref = _reference_out_and_dx(ref, x, grad, cu_seqlens)
    tp_group = parallel_state.get_tensor_model_parallel_group()
    layer = build_layer(ref, _config(tp=world, cp=1, sequence_parallel=True))
    shard = lambda t: t[0].chunk(world, dim=0)[tp_group.rank()].unsqueeze(1).contiguous()  # noqa: E731
    x_local = shard(x).requires_grad_()
    out, _ = layer(x_local, packed_seq_params=packed(cu_seqlens))
    out.backward(shard(grad))
    return {
        "sp_out": rel_err(out_ref, gather(out[:, 0], 0, tp_group)),
        "sp_dx": rel_err(dx_ref, gather(x_local.grad[:, 0], 0, tp_group)),
    }


def _zigzag(full: torch.Tensor, rank: int, size: int) -> torch.Tensor:
    parts = []
    for segment in full.split(SEQLENS):
        chunks = segment.chunk(2 * size)
        parts += [chunks[rank], chunks[2 * size - 1 - rank]]
    return torch.cat(parts)


def check_context_parallel(world: int) -> dict[str, float]:
    """CP = world through the zigzag relayout (allgather_cp=False), each sequence split across ranks."""
    _setup(tp=1, cp=world)
    ref, x, grad, cu_seqlens = _reference_and_input(seed=11)
    out_ref, dx_ref = _reference_out_and_dx(ref, x, grad, cu_seqlens)
    rank = parallel_state.get_context_parallel_rank()
    layer = build_layer(ref, _config(tp=1, cp=world, sequence_parallel=False), allgather_cp=False)
    x_local = _zigzag(x[0], rank, world).unsqueeze(1).contiguous().requires_grad_()
    out, _ = layer(x_local, packed_seq_params=packed(cu_seqlens))
    out.backward(_zigzag(grad[0], rank, world).unsqueeze(1))
    return {
        "cp_out": rel_err(_zigzag(out_ref, rank, world), out[:, 0]),
        "cp_dx": rel_err(_zigzag(dx_ref, rank, world), x_local.grad[:, 0]),
    }


def check_checkpoint_round_trip(world: int, ckpt_dir: str) -> dict[str, float]:
    """Save a TP-sharded layer with dist_checkpointing and load it into one holding other weights."""
    _setup(tp=world, cp=1)
    config = _config(tp=world, cp=1, sequence_parallel=False)
    saved = build_layer(_reference_and_input(seed=21)[0], config)
    dist_checkpointing.save(saved.sharded_state_dict(), ckpt_dir)
    loaded = build_layer(_reference_and_input(seed=22)[0], config)
    loaded.load_state_dict(dist_checkpointing.load(loaded.sharded_state_dict(), ckpt_dir))
    loaded_params = dict(loaded.named_parameters())
    mismatched = [name for name, p in saved.named_parameters() if not torch.equal(p, loaded_params[name])]
    return {"ckpt_mismatched_params": float(len(mismatched))}


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    errors = {**check_sequence_parallel(world), **check_context_parallel(world)}
    ckpt_dir = [tempfile.mkdtemp(prefix="linear_attn_ckpt_") if rank == 0 else None]
    dist.broadcast_object_list(ckpt_dir, src=0)
    errors.update(check_checkpoint_round_trip(world, ckpt_dir[0]))
    parallel_state.destroy_model_parallel()
    if rank == 0:
        print("  ".join(f"{k}={v:.2e}" for k, v in errors.items()))
    bad = {k: v for k, v in errors.items() if v > (0 if k.startswith("ckpt") else TOLERANCE)}
    dist.destroy_process_group()
    if bad:
        print(f"FAILED: {bad}")
        sys.exit(1)
    if rank == 0:
        print(f"LinearAttentionLayer SP / CP / checkpoint paths match at world size {world}")


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=2", __file__])
    main()
