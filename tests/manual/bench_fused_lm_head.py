"""Time and peak memory of one forward + backward: ``fused`` (materialised logits) vs ``fused_lm_head``.

torchrun --nproc-per-node 2 tests/manual/bench_fused_lm_head.py [--tokens 49152] [--scored 40960] [--tile-rows 4096]

Uses Megatron's ColumnParallelLinear over a TP group of all ranks, with gradient accumulation fusion into an fp32
``main_grad`` as in training.
"""

import argparse
import os
import statistics

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel import ColumnParallelLinear, model_parallel_cuda_manual_seed

from miles.backends.training_utils.loss.hub.fused_log_probs import (
    fused_lm_head_log_probs_and_entropy,
    fused_log_probs_and_entropy,
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=49_152)
    parser.add_argument("--scored", type=int, default=40_960)
    parser.add_argument("--hidden", type=int, default=5_120)
    parser.add_argument("--vocab", type=int, default=248_320)
    parser.add_argument("--tile-rows", type=int, default=4_096)
    parser.add_argument("--with-entropy", action="store_true")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--variants", default="fused,fused_lm_head")
    return parser.parse_args()


def _output_layer(args, tp_size):
    config = ModelParallelConfig(
        tensor_model_parallel_size=tp_size,
        params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=True,
    )
    layer = ColumnParallelLinear(
        args.hidden,
        args.vocab,
        config=config,
        init_method=lambda w: torch.nn.init.normal_(w, std=args.hidden**-0.5),
        bias=False,
        gather_output=False,
        skip_bias_add=True,
    ).cuda()
    layer.weight.main_grad = torch.zeros_like(layer.weight, dtype=torch.float32)
    return layer


def _step(variant, args, layer, hidden, rows, targets, tp_group):
    kwargs = dict(tp_group=tp_group, vocab_size=args.vocab, with_entropy=args.with_entropy)
    h = hidden.detach().requires_grad_(True)
    if variant == "fused":
        logits, _ = layer(h.unsqueeze(1))
        log_probs, entropy = fused_log_probs_and_entropy(
            logits.squeeze(1), rows, targets, inplace_backward=True, **kwargs
        )
    else:

        def project(rows_hidden):
            logits, _ = layer(rows_hidden.unsqueeze(1))
            return logits.squeeze(1)

        log_probs, entropy = fused_lm_head_log_probs_and_entropy(
            h, project, rows, targets, tile_rows=args.tile_rows, **kwargs
        )
    loss = log_probs.sum() + (entropy.sum() if entropy is not None else 0)
    loss.backward()
    return log_probs.detach(), h.grad


def main():
    args = _parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)
    model_parallel_cuda_manual_seed(0)
    tp_group = parallel_state.get_tensor_model_parallel_group()

    torch.manual_seed(0)  # same hidden states and targets on every rank, as after the TP gather
    hidden = torch.randn(args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
    rows = torch.arange(args.tokens - args.scored, args.tokens, device="cuda")
    targets = torch.randint(0, args.vocab, (args.scored,), device="cuda")
    layer = _output_layer(args, world)

    results, reference = {}, None
    for variant in args.variants.split(","):
        torch.cuda.empty_cache()
        _step(variant, args, layer, hidden, rows, targets, tp_group)  # warm-up (Triton compile, cuBLAS)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        times = []
        for _ in range(args.iters):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            log_probs, hidden_grad = _step(variant, args, layer, hidden, rows, targets, tp_group)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        peak = (torch.cuda.max_memory_allocated() - base) / 2**30
        if reference is None:
            reference = (log_probs, hidden_grad.float())
        lp_diff = (log_probs - reference[0]).abs().max().item()
        grad_diff = (hidden_grad.float() - reference[1]).abs().max().item()
        results[variant] = (statistics.median(times), peak, lp_diff, grad_diff)
        del log_probs, hidden_grad

    if rank == 0:
        print(
            f"tokens {args.tokens}, scored {args.scored}, hidden {args.hidden}, vocab {args.vocab} "
            f"(TP {world}: {args.vocab // world} per rank), tile {args.tile_rows}, entropy {args.with_entropy}"
        )
        print(f"{'variant':<15}{'fwd+bwd ms':>12}{'peak extra GiB':>16}{'max |dlogp|':>14}{'max |dh|':>12}")
        for variant, (ms, peak, lp_diff, grad_diff) in results.items():
            print(f"{variant:<15}{ms:>12.1f}{peak:>16.2f}{lp_diff:>14.2e}{grad_diff:>12.2e}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
