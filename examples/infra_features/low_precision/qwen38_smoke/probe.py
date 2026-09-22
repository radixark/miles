"""Runtime assertions for the Qwen FP8 smoke test."""
import json
import os
from pathlib import Path

import torch

_seen = set()
_registered = False


def record_fp8(module, inputs, output) -> None:
    name = module.__class__.__name__
    key = id(module)
    if key in _seen:
        return
    _seen.add(key)
    assert module.fp8, f"{name} executed without FP8"
    if torch.distributed.get_rank() == 0:
        with (Path(os.environ["FP8_SMOKE_ROOT"]) / "fp8-execution.jsonl").open("a") as handle:
            handle.write(json.dumps({"module": name, "fp8": bool(module.fp8), "recipe": str(module.fp8_meta.get("recipe"))}) + "\n")


def before_step(args, rollout_id, step_id, model, optimizer, scheduler) -> None:
    global _registered
    assert args.fp8 == "e4m3" and args.fp8_recipe == "blockwise"
    if _registered:
        return
    count = 0
    for chunk in model:
        for module_name, module in chunk.named_modules():
            if "decoder.layers" in module_name and module.__class__.__module__.startswith("transformer_engine.") and module.__class__.__name__ in ("Linear", "LayerNormLinear", "LayerNormMLP"):
                module.register_forward_hook(record_fp8)
                count += 1
    assert count > 0, "No Transformer Engine projection found"
    _registered = True
    print(f"FP8_SMOKE_REGISTERED projections={count}", flush=True)
