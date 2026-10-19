"""LinearAttentionLayer's sequence-parallel and context-parallel paths against the replicated KDA reference,
and a TP-sharded checkpoint round trip of its sharded_state_dict.

On Blackwell (SM100a / SM103a) the deterministic KDA backends are added on 128-dim heads (the kernels' domain):
sequence-parallel arms per ``--kda-backend deterministic_fla_fwd`` / ``deterministic_cake_fwd`` against the replicated
fla reference (this file's TOLERANCE) and against the fla-backend layer (the kernels' 1e-2 contract) with no
fallback warning, context-parallel arms (CP = world, zigzag relayout) running the CP host protocol of
the deterministic kernels with no fallback, and -- at world size 4 -- a TP2 x CP2 arm whose fallback count follows the
drop-in's CP domain (``deterministic_backward_applies``: 0 now that the protocol is in it). Hopper CI
(``stage-b-2-gpu-h200``) skips the deterministic arms, where the drop-in would fall back.

    torchrun --nproc_per_node=2 tests/fast-gpu/test_linear_attn_layer.py
    LINEAR_ATTN_NPROC=4 python tests/fast-gpu/test_linear_attn_layer.py      # adds the TP2 x CP2 arms
"""

import os
import sys
import tempfile
import warnings
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci

from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import LinearAttnHeads

sys.path.insert(0, os.path.dirname(__file__))
from linear_attn_reference import ReplicatedKDA, build_layer, gather, packed, rel_err  # noqa: E402

register_cuda_ci(
    est_time=120, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=2
)

HIDDEN = 256
HEADS = LinearAttnHeads(num_k_heads=8, num_v_heads=8, head_k_dim=64, head_v_dim=64)
HEADS128 = LinearAttnHeads(num_k_heads=8, num_v_heads=8, head_k_dim=128, head_v_dim=128)  # deterministic kernels' domain
SEQLENS = [256, 256]
TOLERANCE = 2e-2
DET_TOLERANCE = 1e-2  # det backend vs fla backend, same layer weights (the kernels' contract)
DET_BACKENDS = ("deterministic_fla_fwd", "deterministic_cake_fwd")
BLACKWELL = ((10, 0), (10, 3))


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


def _reference_and_input(seed: int, heads: LinearAttnHeads = HEADS):
    torch.manual_seed(seed)
    ref = ReplicatedKDA(HIDDEN, heads, torch.bfloat16).cuda()
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


def _fallbacks(seen) -> float:
    return float(sum("fell back to fla" in str(w.message) for w in seen))


def check_context_parallel_det(world: int, backend: str) -> dict[str, float]:
    """CP = world, 128-dim heads, zigzag relayout: the deterministic backend's CP path (the host protocol -- the
    compressed state exchanged through fla's CP pre-process, the deterministic kernels seeded with it) against the
    replicated reference and against the fla-backend CP layer; the drop-in must not fall back (exact zero)."""
    _setup(tp=1, cp=world)
    ref, x, grad, cu_seqlens = _reference_and_input(seed=21, heads=HEADS128)
    out_ref, dx_ref = _reference_out_and_dx(ref, x, grad, cu_seqlens)
    rank = parallel_state.get_context_parallel_rank()
    results = {}
    for arm in ("fla", backend):
        layer = build_layer(ref, _config(tp=1, cp=world, sequence_parallel=False), allgather_cp=False, backend=arm)
        x_local = _zigzag(x[0], rank, world).unsqueeze(1).contiguous().requires_grad_()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            out, _ = layer(x_local, packed_seq_params=packed(cu_seqlens))
            out.backward(_zigzag(grad[0], rank, world).unsqueeze(1))
        results[arm] = (out[:, 0], x_local.grad[:, 0], _fallbacks(seen))
    out_det, dx_det, fallbacks = results[backend]
    out_fla, dx_fla, _ = results["fla"]
    return {
        f"cp_{backend}_out": rel_err(_zigzag(out_ref, rank, world), out_det),
        f"cp_{backend}_dx": rel_err(_zigzag(dx_ref, rank, world), dx_det),
        f"cp_{backend}_vs_fla_out": rel_err(out_fla, out_det) / (DET_TOLERANCE / TOLERANCE),
        f"cp_{backend}_vs_fla_dx": rel_err(dx_fla, dx_det) / (DET_TOLERANCE / TOLERANCE),
        f"cp_{backend}_fallback_mismatch": fallbacks,  # exact: 0 fallbacks expected
    }


def check_sequence_parallel_det(world: int, backend: str) -> dict[str, float]:
    """TP = world, 128-dim heads: the deterministic backend against the replicated fla reference and against the
    fla-backend layer holding the same shard; the drop-in must not fall back (counted as an exact-zero error)."""
    _setup(tp=world, cp=1)
    ref, x, grad, cu_seqlens = _reference_and_input(seed=31, heads=HEADS128)
    out_ref, dx_ref = _reference_out_and_dx(ref, x, grad, cu_seqlens)
    tp_group = parallel_state.get_tensor_model_parallel_group()
    config = _config(tp=world, cp=1, sequence_parallel=True)
    shard = lambda t: t[0].chunk(world, dim=0)[tp_group.rank()].unsqueeze(1).contiguous()  # noqa: E731
    results = {}
    for arm in ("fla", backend):
        layer = build_layer(ref, config, backend=arm)
        x_local = shard(x).requires_grad_()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            out, _ = layer(x_local, packed_seq_params=packed(cu_seqlens))
            out.backward(shard(grad))
        results[arm] = (gather(out[:, 0], 0, tp_group), gather(x_local.grad[:, 0], 0, tp_group), _fallbacks(seen))
    out_det, dx_det, fallbacks = results[backend]
    out_fla, dx_fla, _ = results["fla"]
    return {
        f"sp_{backend}_out": rel_err(out_ref, out_det),
        f"sp_{backend}_dx": rel_err(dx_ref, dx_det),
        f"sp_{backend}_vs_fla_out": rel_err(out_fla, out_det) / (DET_TOLERANCE / TOLERANCE),
        f"sp_{backend}_vs_fla_dx": rel_err(dx_fla, dx_det) / (DET_TOLERANCE / TOLERANCE),
        f"sp_{backend}_fallback_mismatch": fallbacks,  # exact: 0 fallbacks expected
    }


def check_tp_cp_det(backend: str) -> dict[str, float]:
    """TP2 x CP2 (world 4), 128-dim heads, zigzag relayout: the deterministic backend's output and dx against the
    replicated reference, and its fallback count against the drop-in's CP domain (the CP host protocol of
    the deterministic kernels is in it: 0 fallbacks expected)."""
    from miles_plugins.models.kda_chunk_train import deterministic_backward_applies

    _setup(tp=2, cp=2)
    ref, x, grad, cu_seqlens = _reference_and_input(seed=41, heads=HEADS128)
    out_ref, dx_ref = _reference_out_and_dx(ref, x, grad, cu_seqlens)
    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_rank, cp_size = parallel_state.get_context_parallel_rank(), parallel_state.get_context_parallel_world_size()
    layer = build_layer(ref, _config(tp=2, cp=2, sequence_parallel=True), allgather_cp=False, backend=backend)
    local = _zigzag(x[0], cp_rank, cp_size)  # this CP rank's tokens, then this TP rank's sequence-parallel shard
    x_local = local.chunk(2, dim=0)[tp_group.rank()].unsqueeze(1).contiguous().requires_grad_()
    grad_local = _zigzag(grad[0], cp_rank, cp_size).chunk(2, dim=0)[tp_group.rank()].unsqueeze(1)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        out, _ = layer(x_local, packed_seq_params=packed(cu_seqlens))
        out.backward(grad_local)
    cp_in_domain = deterministic_backward_applies(
        capability=torch.cuda.get_device_capability(), head_dim=128, value_dim=128, num_heads=4, num_value_heads=4,
        seq_len=sum(SEQLENS) // cp_size, cu_seqlens=None, cp_context=object(),
    )
    expected_fallbacks = 0.0 if cp_in_domain else 1.0
    return {
        f"tp2cp2_{backend}_out": rel_err(_zigzag(out_ref, cp_rank, cp_size), gather(out[:, 0], 0, tp_group)),
        f"tp2cp2_{backend}_dx": rel_err(_zigzag(dx_ref, cp_rank, cp_size), gather(x_local.grad[:, 0], 0, tp_group)),
        f"tp2cp2_{backend}_fallback_mismatch": abs(_fallbacks(seen) - expected_fallbacks),
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


def _exact(key: str) -> bool:
    return key.startswith("ckpt") or key.endswith("_mismatch")


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    errors = {**check_sequence_parallel(world), **check_context_parallel(world)}
    det_arms = torch.cuda.get_device_capability() in BLACKWELL
    if det_arms:
        for backend in DET_BACKENDS:
            errors.update(check_sequence_parallel_det(world, backend))
            errors.update(check_context_parallel_det(world, backend))
            if world == 4:
                errors.update(check_tp_cp_det(backend))
    ckpt_dir = [tempfile.mkdtemp(prefix="linear_attn_ckpt_") if rank == 0 else None]
    dist.broadcast_object_list(ckpt_dir, src=0)
    errors.update(check_checkpoint_round_trip(world, ckpt_dir[0]))
    parallel_state.destroy_model_parallel()
    if rank == 0:
        print("  ".join(f"{k}={v:.2e}" for k, v in errors.items()))
    bad = {k: v for k, v in errors.items() if v > (0 if _exact(k) else TOLERANCE)}
    dist.destroy_process_group()
    if bad:
        print(f"FAILED: {bad}")
        sys.exit(1)
    if rank == 0:
        arms = "SP / CP / checkpoint" + (" + deterministic SP / CP" + (" + TP2xCP2" if world == 4 else "") if det_arms else "")
        print(f"LinearAttentionLayer {arms} paths match at world size {world}")


if __name__ == "__main__":
    if "RANK" not in os.environ:
        nproc = os.environ.get("LINEAR_ATTN_NPROC", "2")
        os.execvp("torchrun", ["torchrun", f"--nproc_per_node={nproc}", __file__])
    main()
