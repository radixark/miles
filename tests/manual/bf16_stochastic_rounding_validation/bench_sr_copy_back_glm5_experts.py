"""Benchmark HDO's FP32 -> BF16 copy-back on GLM-5 routed-expert shapes: round-to-nearest vs stochastic.

One GLM-5 (744B-A40B) routed expert is linear_fc1 [2 x 2048, 6144] (gate+up) and linear_fc2
[6144, 2048], i.e. 37,748,736 FP32 master weights. The masters are randomly initialized once in
pageable host memory, as HDO keeps them, then copied back ``--layers x --experts`` times (GLM-5:
75 MoE layers x 256 experts), each repetition with its own Philox offset as the step hook uses.

Modes:
  rn         HDO's current copy-back: ``bf16_param.copy_(fp32_master, non_blocking=True)``
  sr         the stochastic copy-back: stage the FP32 master on the GPU, then the SR kernel
  rn-kernel  device FP32 -> BF16 cast only (masters already on the GPU)
  sr-kernel  SR kernel only (masters already on the GPU)

    python bench_sr_copy_back_glm5_experts.py --mode sr --layers 75 --experts 8
"""

import argparse
import json

import torch

from miles.kernels.quant.stochastic_round import stochastic_round_to_bf16

FC1_SHAPE = (2 * 2048, 6144)
FC2_SHAPE = (6144, 2048)
GLM5_MOE_LAYERS = 75
GLM5_ROUTED_EXPERTS = 256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["rn", "sr", "rn-kernel", "sr-kernel"], required=True)
    parser.add_argument("--layers", type=int, default=GLM5_MOE_LAYERS)
    parser.add_argument("--experts", type=int, default=GLM5_ROUTED_EXPERTS)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--ranks", type=int, default=64, help="ranks sharing the routed experts, for projection")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json", help="append one JSON result line to this file")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator().manual_seed(args.seed)
    masters = [torch.randn(shape, generator=generator).mul_(0.02).reshape(-1) for shape in (FC1_SHAPE, FC2_SHAPE)]
    if args.mode.endswith("-kernel"):
        masters = [master.to(device) for master in masters]
    params = [torch.empty(master.numel(), dtype=torch.bfloat16, device=device) for master in masters]
    stream = torch.cuda.Stream()  # HDO's H2D stream

    def copy_shard(master: torch.Tensor, param: torch.Tensor, offset: int) -> None:
        # Same shape as the adapter: the FP32 staging dies with this frame, before the next shard.
        if args.mode in ("rn", "rn-kernel"):
            param.copy_(master, non_blocking=True)
        else:
            staged = master.to(device, non_blocking=True)
            stochastic_round_to_bf16(staged, param, seed=args.seed, offset=offset)

    def copy_back(step: int) -> None:
        with torch.cuda.stream(stream):
            for index, (master, param) in enumerate(zip(masters, params)):
                copy_shard(master, param, (step << 32) | index)

    for step in range(args.warmup):
        copy_back(step)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_memory = torch.cuda.memory_allocated()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = args.layers * args.experts
    start.record(stream)
    for step in range(repetitions):
        copy_back(args.warmup + step)
    stop.record(stream)
    torch.cuda.synchronize()
    seconds = start.elapsed_time(stop) / 1e3

    expert_numel = sum(master.numel() for master in masters)
    full_numel = GLM5_MOE_LAYERS * GLM5_ROUTED_EXPERTS * expert_numel
    ns_per_element = seconds * 1e9 / (repetitions * expert_numel)
    result = {
        "mode": args.mode,
        "repetitions": repetitions,
        "seconds": round(seconds, 4),
        "ms_per_expert": round(seconds * 1e3 / repetitions, 4),
        "ns_per_element": round(ns_per_element, 5),
        "fp32_gb_per_s": round(4 * repetitions * expert_numel / seconds / 1e9, 1),
        "peak_extra_device_mib": round((torch.cuda.max_memory_allocated() - base_memory) / 2**20, 1),
        "projected_full_routed_s": round(full_numel * ns_per_element / 1e9, 2),
        "projected_per_rank_s": round(full_numel * ns_per_element / 1e9 / args.ranks, 3),
        "ranks": args.ranks,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
    }
    print(json.dumps(result))
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
