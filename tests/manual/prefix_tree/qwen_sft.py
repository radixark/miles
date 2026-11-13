"""Single-GPU shared-prefix SFT correctness and performance experiment.

Run from the Miles root with PYTHONPATH=.: python tests/manual/prefix_tree/qwen_sft.py
Requires torch, transformers and kernels; fused runs use FA4 on Blackwell.
This intentionally does not start Ray, rollout servers, or an RL controller.
"""

import argparse
import copy
import importlib.metadata
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree
from miles.backends.training_utils.prefix_tree.transformers import TreeSFTBatch, enable_transformers_prefix_tree


def _data(prefix_length, suffix_length, branches, vocab_size):
    generator = torch.Generator().manual_seed(123)
    prefix = torch.randint(10, vocab_size, (prefix_length,), generator=generator).tolist()
    sequences, targets = [], []
    for sample in range(branches):
        suffix = torch.randint(10, vocab_size, (suffix_length + sample % 3,), generator=generator).tolist()
        sequence = prefix + suffix
        sequences.append(sequence)
        # Include the final prompt prediction, plus decode tokens; ignore the final input.
        targets.append([-100] * (prefix_length - 1) + sequence[prefix_length:] + [-100])
    return sequences, targets


def _batch(sequences, targets, *, shared):
    start = time.perf_counter()
    plan = build_prefix_tree(sequences, share_prefixes=shared)
    cpu_ms = 1000 * (time.perf_counter() - start)
    batch = TreeSFTBatch.from_plan(plan, targets, device="cuda")
    torch.cuda.synchronize()
    return batch, cpu_ms


def _backward(model, batch, varlen):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = batch.loss(model, varlen_func=varlen)
    loss.backward()
    return loss.detach()


def _differences(actual, expected):
    squared_error, squared_reference, maximum = 0.0, 0.0, 0.0
    bitwise, per_parameter = True, []
    for (name, parameter), (_, reference) in zip(actual.named_parameters(), expected.named_parameters(), strict=True):
        x, y = parameter.grad, reference.grad
        if x is None or y is None:
            raise AssertionError(f"missing gradient: {name}")
        delta = (x - y).double()
        error = delta.square().sum().item()
        norm = y.double().square().sum().item()
        abs_max = delta.abs().max().item()
        squared_error += error
        squared_reference += norm
        maximum = max(maximum, abs_max)
        bitwise &= torch.equal(x, y)
        per_parameter.append({"name": name, "relative_l2": (error / max(norm, 1e-60)) ** 0.5, "max_abs": abs_max})
    return {
        "bitwise_equal": bitwise,
        "relative_l2": (squared_error / squared_reference) ** 0.5,
        "max_abs": maximum,
        "per_parameter": per_parameter,
    }


def _expanded_microbatch_control(reference, sequences, targets, varlen):
    """Measure BF16 regrouping noise without sharing any decoder computation."""
    control = copy.deepcopy(reference)
    control.zero_grad(set_to_none=True)
    total = sum(sum(label != -100 for label in labels) for labels in targets)
    for sequence, labels in zip(sequences, targets, strict=True):
        batch, _ = _batch([sequence], [labels], shared=False)
        count = sum(label != -100 for label in labels)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = batch.loss(control, varlen_func=varlen) * (count / total)
        loss.backward()
    differences = _differences(control, reference)
    del control
    return differences


def _measure(model, batch, varlen, iterations):
    for _ in range(2):
        _backward(model, batch, varlen)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _backward(model, batch, varlen)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    median = statistics.median(times)
    return {
        "seconds": times,
        "median_seconds": median,
        "expanded_input_tokens_per_second": batch.attention.plan.expanded_tokens / median,
        "supervised_tokens_per_second": batch.supervision.rows.numel() / median,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def run(model_name="Qwen/Qwen3-0.6B", revision="main", prefix=2049, suffix=257, branches=8, iterations=5):
    if min(prefix, suffix, branches, iterations) < 1:
        raise ValueError("prefix, suffix, branches and iterations must all be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("this benchmark explicitly requires Blackwell and FA4")
    # Optional GPU kernel dependency, not imported by the CPU correctness suite.
    from kernels import get_kernel

    varlen = get_kernel("kernels-community/flash-attn4", version=1).flash_attn_varlen_func
    torch.manual_seed(19)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"Loading {model_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=torch.float32).cuda().train()
    enable_transformers_prefix_tree(model)
    sequences, targets = _data(prefix, suffix, branches, model.config.vocab_size)
    expanded, expanded_cpu_ms = _batch(sequences, targets, shared=False)
    compact, compact_cpu_ms = _batch(sequences, targets, shared=True)
    reference = copy.deepcopy(model)
    print("Comparing losses and every parameter gradient", flush=True)
    expanded_loss = _backward(reference, expanded, varlen)
    compact_loss = _backward(model, compact, varlen)
    differences = _differences(model, reference)
    torch.testing.assert_close(compact_loss, expanded_loss, atol=2e-3, rtol=2e-3)
    # Numerical gates are explicit; report bit equality independently.
    if differences["relative_l2"] > 0.02:
        raise AssertionError(f"BF16 gradient relative L2 too large: {differences['relative_l2']}")
    worst = max(p["relative_l2"] for p in differences["per_parameter"])
    if worst > 0.05:
        raise AssertionError(f"BF16 per-parameter gradient relative L2 too large: {worst}")
    print("Measuring expanded-only BF16 microbatch regrouping control", flush=True)
    regrouping_control = _expanded_microbatch_control(reference, sequences, targets, varlen)
    update_max_abs = 0.0
    strict_optimizer_errors = []
    for trained in (model, reference):
        torch.optim.SGD(trained.parameters(), lr=0.01).step()
    for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        update_max_abs = max(update_max_abs, (actual - expected).abs().max().item())
        try:
            torch.testing.assert_close(actual, expected, atol=5e-5, rtol=1e-4)
        except AssertionError:
            strict_optimizer_errors.append(name)
    del reference
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    print("Benchmarking matched expanded/tree FA4 forward+backward", flush=True)
    baseline = _measure(model, expanded, varlen, iterations)
    tree = _measure(model, compact, varlen, iterations)
    return {
        "model": model_name,
        "revision": getattr(model.config, "_commit_hash", revision),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "kernels": importlib.metadata.version("kernels"),
        "varlen_module": varlen.__module__,
        "gpu": torch.cuda.get_device_name(),
        "precision": "FP32 parameters/gradients, BF16 autocast, TF32 disabled",
        "prefix": prefix,
        "suffix": suffix,
        "branches": branches,
        "expanded_tokens": expanded.attention.plan.expanded_tokens,
        "unique_tokens": len(compact.attention.plan.tokens),
        "gathered_kv_tokens": len(compact.attention.plan.kv_indices),
        "expanded_loss": expanded_loss.item(),
        "tree_loss": compact_loss.item(),
        "gradients": differences,
        "expanded_microbatch_control_gradients": regrouping_control,
        "sgd_update_max_abs": update_max_abs,
        "strict_optimizer_match": not strict_optimizer_errors,
        "strict_optimizer_mismatched_parameters": strict_optimizer_errors,
        "planner_cpu_ms": {"expanded": expanded_cpu_ms, "tree": compact_cpu_ms},
        "expanded": baseline,
        "tree": tree,
        "speedup": baseline["median_seconds"] / tree["median_seconds"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("prefix-tree-sft.json"))
    parser.add_argument("--prefix", type=int, default=2049)
    parser.add_argument("--suffix", type=int, default=257)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    report = run(
        model_name=args.model,
        revision=args.revision,
        prefix=args.prefix,
        suffix=args.suffix,
        branches=args.branches,
        iterations=args.iterations,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if "gradients" not in key}, indent=2))
    # Preserve a failed strict gate while retaining diagnostic performance and
    # numerical-control evidence. Never turn a BF16 mismatch into an exact pass.
    if not report["strict_optimizer_match"]:
        sys.exit(1)
