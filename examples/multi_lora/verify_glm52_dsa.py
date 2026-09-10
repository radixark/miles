"""GPU parity and 8K memory checks for the pressure recipe's attention kernel."""

import json

import torch
from examples.multi_lora.glm52_native_dsa import packed_sparse_sdpa
from megatron.core.transformer.experimental_attention_variant.dsa import unfused_dsa_fn


def verify():
    torch.manual_seed(41)
    device = "cuda"
    length = 24
    starts = torch.tensor([0] * 8 + [8] * 16, device=device, dtype=torch.int32)
    ends = torch.arange(1, length + 1, device=device, dtype=torch.int32)
    indices = torch.stack([torch.randperm(length, device=device)[:12] for _ in range(length)]).unsqueeze(0)
    indices[:, :, -1] = -1
    originals = [torch.randn(length, 2, 32, device=device, dtype=torch.bfloat16) for _ in range(3)]
    results = []
    for kernel in (unfused_dsa_fn, packed_sparse_sdpa):
        inputs = [value.clone().requires_grad_() for value in originals]
        output = kernel(*inputs, indices, 32**-0.5, varlen_starts=starts, varlen_ends=ends)
        output.float().square().sum().backward()
        results.append((output.detach(), [value.grad.clone() for value in inputs]))
    torch.testing.assert_close(results[0][0], results[1][0], atol=0.02, rtol=0.02)
    for expected, actual in zip(results[0][1], results[1][1], strict=True):
        torch.testing.assert_close(expected, actual, atol=0.06, rtol=0.04)
        assert torch.isfinite(actual).all()
    changed = [value.clone() for value in originals]
    changed[1][8:] *= 7
    changed[2][8:] += 11
    output = packed_sparse_sdpa(*changed, indices, 32**-0.5, varlen_starts=starts, varlen_ends=ends)
    torch.testing.assert_close(output[:8], results[1][0][:8], atol=0, rtol=0)
    print(json.dumps({"parity": "passed", "packed_sequence_isolation": "passed"}), flush=True)

    del results, originals, changed, inputs, output
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    length = 8192
    inputs = [torch.randn(length, 4, 256, device=device, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    indices = torch.arange(2048, device=device).expand(1, length, -1)
    starts = torch.zeros(length, device=device, dtype=torch.int32)
    ends = torch.arange(1, length + 1, device=device, dtype=torch.int32)
    output = packed_sparse_sdpa(*inputs, indices, 256**-0.5, varlen_starts=starts, varlen_ends=ends)
    output.float().square().mean().backward()
    assert all(torch.isfinite(value.grad).all() for value in inputs)
    torch.cuda.synchronize()
    print(json.dumps({"8k_forward_backward": "passed", "peak_bytes": torch.cuda.max_memory_allocated()}), flush=True)


if __name__ == "__main__":
    verify()
