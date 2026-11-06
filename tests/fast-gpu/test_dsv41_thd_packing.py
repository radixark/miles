"""DeepSeek-V4.1 attention on a THD-packed stream matches each sample run alone.

Run with:
    torchrun --nproc_per_node=4 tests/fast-gpu/test_dsv41_thd_packing.py

A scaled-down V4.1 stack (a window layer, ratio-2 and ratio-1 kv_source layers with later layers
reusing their latent and top-k, an index_source layer that reuses another layer's index keys, and
one that reads the candidate blocks) runs on a packed stream at CP=1 and under all-gather CP. The
reference runs every sample on its own through the BSHD path at CP=1, padded like a BSHD batch.

The outputs match it bit for bit on H200. The gradients cannot: the sparse-attention backward
accumulates dKV in no fixed order, so the reference run twice already differs by about 1e-2 in
the input and parameter gradients. Their bounds sit above that spread; the unpacked-only code
this replaces missed every bound by 25x or more.

Segment lengths are odd, so a ratio-2 group straddles a CP split; one segment is shorter than
ratio 2, so it has no compressed key at all; and the samples are longer than the window and the
indexer top-k, so the window clamp and the top-k selection both matter.

The Engram n-gram hash of a packed stream is checked the same way, against each sample's own hash.
"""

import os
import sys
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.nn  # cp_utils reaches for this submodule without importing it
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from tests.ci.ci_register import register_cuda_ci

from miles.backends.megatron_utils.parallel import PackedSeqParamsWithHostCuSeqlens
from miles_plugins.models.deepseek_v4.ops.thd_utils import get_q_positions_thd
from miles_plugins.models.deepseek_v4_1.deepseek_v4_1 import DeepSeekV41Attention, V41Runtime
from miles_plugins.models.deepseek_v4_1.engram import DeepSeekV41Engram, compute_hash_multipliers

register_cuda_ci(
    est_time=120,
    suite="stage-c-4-gpu-h200",
    labels=["precision", "megatron"],
    hardware=["hopper", "blackwell"],
    num_gpus=4,
)

HIDDEN = 512
RATIOS = (0, 2, 2, 2, 1, 1, 1, 1)
KV_SOURCES = (1, 3, 4)
INDEX_SOURCES = (1, 3, 4, 6)  # 6 reuses layer 4's index keys and reads its candidate blocks
CANDIDATE_SOURCE = 4
# The last segment pads the stream to a multiple of 4 * 64, the way get_batch pads under CP.
SEGMENTS = (613, 290, 1, 505, 127)
# The BSHD reference pads every sample to this length, as a BSHD micro-batch pads to its longest sample.
BSHD_LEN = 640
TOLERANCE = {"out": 1e-3, "dx": 2e-2, "grad": 3e-2}
# Samples shorter and longer than the 4-gram look-back.
ENGRAM_SEGMENTS = (5, 1, 2, 9, 3)


def _config(cp: int, topk_backend: str) -> TransformerConfig:
    config = TransformerConfig(
        num_layers=len(RATIOS),
        hidden_size=HIDDEN,
        num_attention_heads=8,
        params_dtype=torch.bfloat16,
        bf16=True,
        tensor_model_parallel_size=1,
        context_parallel_size=cp,
        layernorm_epsilon=1e-6,
    )
    fields = dict(
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_pos_emb_head_dim=64,
        o_lora_rank=256,
        o_groups=2,
        csa_window_size=128,
        csa_compress_ratios=list(RATIOS),
        csa_compress_rotary_base=160000,
        rotary_base=10000,
        rotary_scaling_factor=16,
        original_max_position_embeddings=65536,
        beta_fast=32,
        beta_slow=1,
        dsa_indexer_n_heads=32,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=64,
        miles_dsa_topk_backend=topk_backend,
        v41_kv_source_layer_ids=KV_SOURCES,
        v41_index_source_layer_ids=INDEX_SOURCES,
        v41_candidate_source_layer_id=CANDIDATE_SOURCE,
        v41_candidate_topk_blocks=8,
        v41_candidate_block_size=8,
        v41_runtime=V41Runtime(),
    )
    for name, value in fields.items():
        setattr(config, name, value)
    return config


def _build(cp: int, topk_backend: str) -> torch.nn.ModuleList:
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp)
    model_parallel_cuda_manual_seed(1234)
    config = _config(cp, topk_backend)
    pgs = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
    layers = torch.nn.ModuleList(
        DeepSeekV41Attention(config, layer_number=i + 1, pg_collection=pgs) for i in range(len(RATIOS))
    ).cuda()
    # The same weights on every build, whatever the modules initialized themselves to.
    gen = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        for _, p in sorted(layers.named_parameters()):
            noise = torch.randn(p.shape, generator=gen, device="cuda", dtype=torch.float32)
            p.copy_(1 + 0.1 * noise if p.dim() == 1 else noise * p.shape[-1] ** -0.5)
    return layers


def _forward(layers, h, packed_seq_params=None):
    layers[0].config.v41_runtime.reset()
    for layer in layers:
        out, _ = layer(h, packed_seq_params=packed_seq_params)
        h = h + out
    return h


def _inputs():
    gen = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(sum(SEGMENTS), 1, HIDDEN, generator=gen, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn(x.shape, generator=gen, device="cuda", dtype=torch.bfloat16)
    return x, grad


def _param_grads(layers):
    return {name: p.grad.float().clone() for name, p in layers.named_parameters() if p.grad is not None}


def reference(topk_backend):
    """Every sample alone through the BSHD path at CP=1, followed by padding it cannot see."""
    layers = _build(cp=1, topk_backend=topk_backend)
    x, grad = _inputs()
    gen = torch.Generator(device="cuda").manual_seed(2)
    out = torch.zeros(x.shape, device="cuda", dtype=torch.float32)
    dx = torch.zeros_like(out)
    start = 0
    for n in SEGMENTS:
        pad = torch.randn(BSHD_LEN - n, 1, HIDDEN, generator=gen, device="cuda", dtype=x.dtype)
        seg = torch.cat([x[start : start + n], pad]).requires_grad_()
        seg_out = _forward(layers, seg)
        seg_out.backward(torch.cat([grad[start : start + n], torch.zeros_like(pad)]))
        out[start : start + n] = seg_out.detach()[:n].float()
        dx[start : start + n] = seg.grad[:n].float()
        start += n
    return out, dx, _param_grads(layers)


def packed(cp, topk_backend):
    """The whole stream packed, this rank's contiguous share of it."""
    layers = _build(cp=cp, topk_backend=topk_backend)
    x, grad = _inputs()
    total = sum(SEGMENTS)
    assert total % cp == 0
    local = total // cp
    start = parallel_state.get_context_parallel_rank() * local
    cu = [0] + torch.tensor(SEGMENTS).cumsum(0).tolist()
    params = PackedSeqParamsWithHostCuSeqlens(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor(cu, device="cuda", dtype=torch.int32),
        cu_seqlens_kv=torch.tensor(cu, device="cuda", dtype=torch.int32),
        max_seqlen_q=max(SEGMENTS),
        max_seqlen_kv=max(SEGMENTS),
        cu_seqlens_host=tuple(cu),
        total_tokens=total,
    )
    x_local = x[start : start + local].clone().requires_grad_()
    out = _forward(layers, x_local, params)
    out.backward(grad[start : start + local])
    grads = _param_grads(layers)
    for g in grads.values():
        dist.all_reduce(g, group=parallel_state.get_context_parallel_group())
    return start, local, out.detach().float(), x_local.grad.float(), grads


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def check(cp, topk_backend, ref):
    ref_out, ref_dx, ref_grads = ref
    start, local, out, dx, grads = packed(cp, topk_backend)
    assert grads.keys() == ref_grads.keys(), sorted(grads.keys() ^ ref_grads.keys())
    errors = {
        "out": _rel(out, ref_out[start : start + local]),
        "dx": _rel(dx, ref_dx[start : start + local]),
        "grad": max(_rel(grads[name], ref_grads[name]) for name in ref_grads),
    }
    # Reduced over CP, so every rank of a group reports the same parameter error.
    errors_t = torch.tensor([errors["out"], errors["dx"]], device="cuda")
    dist.all_reduce(errors_t, op=dist.ReduceOp.MAX)
    errors["out"], errors["dx"] = errors_t.tolist()
    return errors


def check_engram_hash():
    """The packed stream's n-gram hash equals each sample's own; without positions it does not."""
    max_ngram, vocab = 4, 1000
    gen = torch.Generator(device="cuda").manual_seed(3)
    engram = SimpleNamespace(
        token_map=torch.randperm(vocab, generator=gen, device="cuda"),
        multipliers=compute_hash_multipliers([1], max_ngram, vocab)[0].cuda(),
        primes=torch.tensor([[1009, 1013], [1019, 1021], [1031, 1033]], device="cuda"),
        offsets=torch.tensor([0, 1009, 2022, 3041, 4062, 5093], device="cuda"),
        pad_id=7,
        max_ngram_size=max_ngram,
    )
    ids = torch.randint(0, vocab, (1, sum(ENGRAM_SEGMENTS)), generator=gen, device="cuda")
    cu = torch.tensor([0, *torch.tensor(ENGRAM_SEGMENTS).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    packed_ids = DeepSeekV41Engram.hash_ids(engram, ids, get_q_positions_thd(cu, ids.size(1)))
    alone = [DeepSeekV41Engram.hash_ids(engram, seg) for seg in ids.split(ENGRAM_SEGMENTS, dim=1)]
    return torch.equal(packed_ids, torch.cat(alone, dim=1)) and not torch.equal(
        DeepSeekV41Engram.hash_ids(engram, ids), packed_ids
    )


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=600))
    cp_sizes = sorted({1, *(cp for cp in (2, 4) if world % cp == 0)})
    failed = []
    engram_ok = check_engram_hash()
    if rank == 0:
        print(f"engram hash: {'ok' if engram_ok else 'FAIL'}")
    if not engram_ok:
        failed.append("engram hash")
    try:
        for topk_backend in ("canonical", "torch"):
            ref = reference(topk_backend)
            for cp in cp_sizes:
                errors = check(cp, topk_backend, ref)
                bad = {k: v for k, v in errors.items() if v > TOLERANCE[k]}
                if rank == 0:
                    print(
                        f"topk={topk_backend:9s} CP={cp}: "
                        + "  ".join(f"{k}={v:.2e}" for k, v in errors.items())
                        + ("  FAIL" if bad else "")
                    )
                if bad:
                    failed.append((topk_backend, cp, bad))
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
    if failed:
        if rank == 0:
            print(f"FAILED: {failed}")
        sys.exit(1)
    if rank == 0:
        print("PASSED")


if __name__ == "__main__":
    # Self-bootstrap under torchrun when run as `python3 file.py`, the CUDA CI runner's mode.
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=4", __file__])
    main()
