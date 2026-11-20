"""Inkling2 attention under context parallelism (``--allgather-cp``) against the same module at CP=1.

Run with:
    torchrun --nproc_per_node=2 tests/fast-gpu/test_inkling2_context_parallel.py

Each rank runs the module twice with the same weights: once on the whole packed sequence with CP
switched off, once on its contiguous slice with CP on. The slice's output, its input gradients and
the CP-summed parameter gradients must match the CP=1 run. The packed sequences are laid out so a
sequence, the sliding window and the short-conv context all cross the CP split, and the length
scaling floor is small enough that every position past it scales differently.
"""

import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300,
    suite="stage-b-2-gpu-h200",
    labels=["miles-plugin"],
    hardware=["hopper", "blackwell"],
    num_gpus=2,
)

HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 64, 4, 2, 16
SEQLENS = [21, 3, 30, 10]  # 64 rows; CP=2 splits the third sequence at its 9th token
GRAD_TOLERANCE = 1e-2  # relative norm error; bf16 rounding of summed partials, set from a measured run


def _config(cp: int):
    from megatron.core.transformer.transformer_config import TransformerConfig

    from miles_plugins.models.inkling2.model import Inkling2Config

    config = TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_query_groups=KV_HEADS,
        kv_channels=HEAD_DIM,
        context_parallel_size=cp,
        params_dtype=torch.bfloat16,
        bf16=True,
        add_bias_linear=False,
        perform_initialization=False,
    )
    config.inkling2 = Inkling2Config(
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        swa_num_attention_heads=HEADS,
        swa_num_key_value_heads=KV_HEADS,
        swa_head_dim=HEAD_DIM,
        sliding_window_size=8,
        d_rel=4,
        rel_extent=16,
        local_layer_ids=frozenset({0}),
        sconv_kernel_size=4,
        rms_norm_eps=1e-6,
        qk_l2_clip_threshold=22.6274,
        rel_l2_clip_threshold=8.0,
        use_value_norm_no_gamma=False,
        log_scaling_n_floor=5.0,
        log_scaling_alpha=0.1,
        n_layers_per_attn_res_block=6,
        attn_res_normalize_values=False,
        route_scale=8.0,
        mlp_gate_softcap=0.0,
        mlp_up_softcap=0.0,
        final_logit_softcapping=15.0,
        unpadded_vocab_size=1,
    )
    return config


def _attention(config, layer_number: int, device):
    from miles_plugins.models.inkling2.layers import Inkling2Attention

    module = Inkling2Attention(config, layer_number).to(device)
    generator = torch.Generator(device=device).manual_seed(layer_number)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(0.2 * torch.randn(param.shape, generator=generator, device=device))
    return module


def _run(module, sources, grad_out, packed):
    sources = [s.detach().clone().requires_grad_() for s in sources]
    out = module(tuple(sources), packed)
    out.backward(grad_out)
    grads = {name: p.grad.detach().clone() for name, p in module.named_parameters()}
    module.zero_grad(set_to_none=True)
    return out.detach(), [s.grad for s in sources], grads


def check_layer(layer_number: int, rank: int, world: int, device) -> dict[str, float]:
    from megatron.core import parallel_state as ps

    from miles_plugins.models.inkling2 import layers

    module = _attention(_config(world), layer_number, device)
    total = sum(SEQLENS)
    local = total // world
    generator = torch.Generator(device=device).manual_seed(100 + layer_number)
    sources = [torch.randn(total, 1, HIDDEN, generator=generator, device=device).to(torch.bfloat16) for _ in range(3)]
    grad_out = torch.randn(total, 1, HIDDEN, generator=generator, device=device).to(torch.bfloat16)
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, *torch.tensor(SEQLENS).cumsum(0).tolist()], device=device))

    cp_world = layers.cp_world
    layers.cp_world = lambda: (1, 0, None)
    try:
        ref_out, ref_input_grads, ref_grads = _run(module, sources, grad_out, packed)
    finally:
        layers.cp_world = cp_world

    rows = slice(rank * local, (rank + 1) * local)
    out, input_grads, grads = _run(module, [s[rows] for s in sources], grad_out[rows], packed)
    for grad in grads.values():
        dist.all_reduce(grad, group=ps.get_context_parallel_group())

    diffs = {"output": _relative_error(out, ref_out[rows])}
    for name, (grad, ref) in zip("qkv", zip(input_grads, ref_input_grads, strict=True), strict=True):
        diffs[f"grad {name} source"] = _relative_error(grad, ref[rows])
    diffs["param grads"] = max(_relative_error(grads[n], ref_grads[n]) for n in grads)
    return diffs


def _relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    return ((value.float() - reference.float()).norm() / reference.float().norm()).item()


def setup_dist():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    from megatron.core import parallel_state as ps

    ps.initialize_model_parallel(context_parallel_size=world)
    return rank, world


def main():
    rank, world = setup_dist()
    device = torch.device("cuda")
    passed = True
    try:
        for layer_number, kind in ((1, "local"), (2, "full")):
            diffs = check_layer(layer_number, rank, world, device)
            gathered = [None] * world
            dist.all_gather_object(gathered, diffs)
            worst = {key: max(d[key] for d in gathered) for key in diffs}
            if rank == 0:
                print(f"\n=== Inkling2 {kind} attention, CP={world} vs CP=1 (relative error) ===")
                for key, value in worst.items():
                    print(f"{key:18s} {value:.3e}")
            # Forward rows are computed as without CP. Gradients sum per-rank partials, so they
            # differ by bf16 rounding; a missing cross-rank contribution is an O(1) error.
            passed = passed and worst["output"] == 0 and max(worst.values()) < GRAD_TOLERANCE
    finally:
        dist.destroy_process_group()
    if rank == 0:
        print("PASSED" if passed else "FAILED")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    # Self-bootstrap under torchrun when run as `python3 file.py`, the CUDA CI runner's mode.
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=2", __file__])
    main()
