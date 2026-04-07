"""Correctness test for Qwen3.5 GDN with native fla Context Parallel.

Run with:
    torchrun --nproc_per_node=2 tests/test_qwen3_5_cp_correctness.py   # CP=2
    torchrun --nproc_per_node=4 tests/test_qwen3_5_cp_correctness.py   # CP=4

Validates that GDN forward+backward with native fla CP produces results
consistent with the non-CP (single-rank full-sequence) baseline.
"""

import os
import sys

import torch
import torch.distributed as dist


def setup_dist():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def build_gdn_module(device, dtype=torch.bfloat16):
    """Build a small Qwen3.5 GDN module for testing."""

    class FakeConfig:
        hidden_size = 256
        linear_num_value_heads = 4
        linear_num_key_heads = 2
        linear_key_head_dim = 64
        linear_value_head_dim = 64
        linear_conv_kernel_dim = 4
        hidden_act = "silu"
        rms_norm_eps = 1e-6

    FakeConfig.dtype = dtype

    from miles_plugins.models.qwen3_5 import Qwen3_5GatedDeltaNet

    return Qwen3_5GatedDeltaNet(FakeConfig, layer_idx=0).to(device=device, dtype=dtype)


def test_cp_forward_backward(rank, world_size):
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    # ---- Reference: full sequence on rank 0 (no CP) ----
    torch.manual_seed(42)
    model_ref = build_gdn_module(device, dtype)

    total_seq_len = 128 * world_size  # must be divisible by world_size
    batch = 1

    torch.manual_seed(123)
    full_hidden = torch.randn(batch, total_seq_len, 256, device=device, dtype=dtype, requires_grad=True)
    full_cu = torch.tensor([0, total_seq_len], dtype=torch.int32, device=device)

    # Forward without CP
    ref_out = model_ref(full_hidden, cu_seqlens=full_cu)
    ref_loss = ref_out.sum()
    ref_loss.backward()
    ref_grad = full_hidden.grad.clone()

    # ---- Test: CP across ranks ----
    torch.manual_seed(42)
    model_cp = build_gdn_module(device, dtype)
    # Copy weights from ref to ensure identical params
    model_cp.load_state_dict(model_ref.state_dict())

    # Set up CP context on the module
    cp_group = dist.group.WORLD
    model_cp.cp_group = cp_group
    model_cp.cp_rank = rank
    model_cp.cp_world_size = world_size

    # Each rank gets its local chunk
    local_seq_len = total_seq_len // world_size
    start = rank * local_seq_len
    end = start + local_seq_len

    torch.manual_seed(123)
    full_hidden_cp = torch.randn(batch, total_seq_len, 256, device=device, dtype=dtype)
    local_hidden = full_hidden_cp[:, start:end, :].clone().contiguous().requires_grad_(True)

    # Global cu_seqlens (build_gdn_cp_context expects global boundaries)
    global_cu = torch.tensor([0, total_seq_len], dtype=torch.int32, device=device)

    # Forward with CP
    cp_out = model_cp(local_hidden, cu_seqlens=global_cu)
    cp_loss = cp_out.sum()

    # Reduce loss across ranks to match reference
    dist.all_reduce(cp_loss, op=dist.ReduceOp.SUM)

    cp_loss.backward()

    # ---- Gather outputs for comparison ----
    gathered_out = [torch.zeros_like(cp_out) for _ in range(world_size)]
    dist.all_gather(gathered_out, cp_out.contiguous())
    full_cp_out = torch.cat(gathered_out, dim=1)

    gathered_grad = [torch.zeros_like(local_hidden.grad) for _ in range(world_size)]
    dist.all_gather(gathered_grad, local_hidden.grad.contiguous())
    full_cp_grad = torch.cat(gathered_grad, dim=1)

    if rank == 0:
        # Compare outputs
        out_diff = (ref_out.detach().float() - full_cp_out.detach().float()).abs()
        out_max_diff = out_diff.max().item()
        out_rel_diff = (out_diff / (ref_out.detach().float().abs() + 1e-8)).max().item()

        # Compare gradients
        grad_diff = (ref_grad.float() - full_cp_grad.float()).abs()
        grad_max_diff = grad_diff.max().item()
        grad_rel_diff = (grad_diff / (ref_grad.float().abs() + 1e-8)).max().item()

        print(f"\n=== CP={world_size} Correctness Test ===")
        print(f"Forward  max abs diff: {out_max_diff:.6e}  max rel diff: {out_rel_diff:.6e}")
        print(f"Backward max abs diff: {grad_max_diff:.6e}  max rel diff: {grad_rel_diff:.6e}")

        # bf16 tolerance: 1e-2 is generous for bf16 accumulated ops
        fwd_ok = out_max_diff < 1e-2
        bwd_ok = grad_max_diff < 1e-2
        print(f"Forward  PASS: {fwd_ok}")
        print(f"Backward PASS: {bwd_ok}")

        if not (fwd_ok and bwd_ok):
            print("FAILED!")
            sys.exit(1)
        else:
            print(f"CP={world_size} test PASSED!")


def main():
    rank, world_size, _ = setup_dist()
    try:
        test_cp_forward_backward(rank, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
