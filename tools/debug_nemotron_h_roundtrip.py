"""Compare a Nemotron-H HF checkpoint with its Megatron Bridge export.

Run on a devbox with the model already staged. This only loads and compares
weights; it performs no optimizer steps or rollouts. Use torchrun for layouts
with more than one model-parallel rank.
"""

import json
import os
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from tap import Tap

from miles_plugins.megatron_bridge.nemotron_h import install
from megatron.bridge import AutoBridge
from miles.utils.megatron_bridge_utils import patch_megatron_model


class Args(Tap):
    model_dir: str
    output_dir: str
    tp: int = 1
    pp: int = 1
    ep: int = 1
    etp: int = 1


def compare_tensor(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual = actual.detach().cpu()
    converted = expected.to(actual.dtype)
    result = {
        "name": name,
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
    }
    if actual.shape != converted.shape:
        return {**result, "equal": False, "shape_mismatch": True}
    different = actual != converted
    count = int(torch.count_nonzero(different))
    result.update(equal=count == 0, different=count, numel=actual.numel())
    if count:
        delta = (actual.float() - converted.float()).abs()
        result.update(max_abs=float(delta.max()), mean_abs=float(delta.mean()))
    return result


def audit_export(bridge: AutoBridge, model: list, args: Args) -> None:
    rank = torch.distributed.get_rank()
    directory = Path(args.model_dir)
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    seen = set()
    with ExitStack() as stack, torch.no_grad(), patch_megatron_model(model):
        handles = {}
        report = stack.enter_context((output / f"tensors-rank{rank}.jsonl").open("w"))
        for name, weight, source in bridge.export_hf_weights(model, cpu=True, show_progress=False):
            if rank != 0:
                continue
            counts["exported"] += 1
            counts["duplicates"] += name in seen
            seen.add(name)
            if name not in index:
                row = {"name": name, "source": source, "unknown_export": True}
                counts["unknown_export"] += 1
            else:
                filename = index[name]
                if filename not in handles:
                    handles[filename] = stack.enter_context(safe_open(directory / filename, framework="pt"))
                row = compare_tensor(name, weight, handles[filename].get_tensor(name))
                row["source"] = source
                counts["equal" if row["equal"] else "different"] += 1
            report.write(json.dumps(row) + "\n")
            if not row.get("equal", False):
                print("MISMATCH", json.dumps(row), flush=True)
            if counts["exported"] % 500 == 0:
                print("PROGRESS", dict(counts), flush=True)
        if rank == 0:
            missing = sorted(set(index) - seen)
            summary = {"layout": {"tp": args.tp, "pp": args.pp, "ep": args.ep, "etp": args.etp}, "counts": dict(counts), "missing": missing}
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            print("SUMMARY", json.dumps(summary), flush=True)


def main() -> None:
    args = Args().parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    install()
    bridge = AutoBridge.from_hf_pretrained(args.model_dir, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = args.tp
    provider.pipeline_model_parallel_size = args.pp
    provider.expert_model_parallel_size = args.ep
    provider.expert_tensor_parallel_size = args.etp
    provider.params_dtype = torch.bfloat16
    provider.pipeline_dtype = torch.bfloat16
    provider.bf16 = True
    provider.finalize()
    print("LOADING", args.tp, args.pp, args.ep, flush=True)
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    print("LOADED", flush=True)
    audit_export(bridge, model, args)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
