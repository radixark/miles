"""Separate repeatability, tree layout, accumulation precision, attention and TP.

Example: torchrun --standalone --nproc-per-node=2 diagnose_equivalence.py
    --model Qwen/Qwen3-30B-A3B --output /tmp/moe-tp2.json --short

This diagnostic never updates weights. Gradient references live on CPU so the
larger TP model requires only one model and one gradient buffer on each GPU.
"""

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree
from miles.backends.training_utils.prefix_tree.qwen import TreeSFTBatch, enable_qwen_prefix_tree


def _local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _snapshot(model):
    return {
        name: _local(parameter.grad).detach().to(device="cpu", copy=True) if parameter.grad is not None else None
        for name, parameter in model.named_parameters()
    }


def _gradient_difference(model, reference):
    totals = torch.zeros(4, device="cuda", dtype=torch.float64)
    for name, parameter in model.named_parameters():
        if parameter.grad is None or reference[name] is None:
            if (parameter.grad is None) != (reference[name] is None):
                raise AssertionError(f"gradient presence changed: {name}")
            continue
        actual = _local(parameter.grad).detach().reshape(-1)
        expected = reference[name].reshape(-1)
        for start in range(0, actual.numel(), 16 * 1024 * 1024):
            a = actual[start : start + 16 * 1024 * 1024].double()
            # Transfer the stored FP32 shard first, then convert on GPU. A
            # simultaneous device+dtype copy can cast on CPU and double PCIe
            # traffic for a 30B reference.
            b = expected[start : start + 16 * 1024 * 1024].to(device="cuda").double()
            delta = a - b
            totals[0].add_(delta.square().sum())
            totals[1].add_(b.square().sum())
            totals[2] = torch.maximum(totals[2], delta.abs().max())
            totals[3].add_((a != b).sum())
    if dist.is_initialized():
        dist.all_reduce(totals[:2], op=dist.ReduceOp.SUM)
        dist.all_reduce(totals[2:3], op=dist.ReduceOp.MAX)
        dist.all_reduce(totals[3:4], op=dist.ReduceOp.SUM)
    error, norm, maximum, unequal = totals.tolist()
    return {
        "relative_l2": (error / max(norm, 1e-60)) ** 0.5,
        "max_abs": maximum,
        "unequal_local_shard_elements_summed_over_ranks": int(unequal),
        "bitwise_equal": unequal == 0,
    }


def _tensor_difference(actual, expected):
    if actual.dtype in (torch.int32, torch.int64):
        return {
            "changed_rows": int((actual != expected).any(dim=-1).sum()),
            "changed_expert_sets": int((actual.sort(dim=-1).values != expected.sort(dim=-1).values).any(dim=-1).sum()),
            "rows": actual.shape[0],
        }
    a, b = actual.double(), expected.double()
    return {
        "bitwise_equal": torch.equal(actual, expected),
        "max_abs": (a - b).abs().max().item(),
        "relative_l2": ((a - b).square().sum() / b.square().sum().clamp_min(1e-60)).sqrt().item(),
    }


def _replicated_norm_gradient_difference(model):
    if not dist.is_initialized():
        return 0.0
    values = [
        _local(p.grad).detach().reshape(-1)
        for name, p in model.named_parameters()
        if name.endswith(("q_norm.weight", "k_norm.weight")) and p.grad is not None
    ]
    value = torch.cat(values)
    replicas = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(replicas, value)
    return max((replica - replicas[0]).abs().max().item() for replica in replicas)


class _Trace:
    def __init__(self, model, occurrence_rows, *, enabled):
        self.values, self.routes, self.handles = {}, {}, []
        if not enabled:
            return
        index = torch.tensor(occurrence_rows, device="cuda", dtype=torch.long)
        for name, module in model.named_modules():
            if name.endswith((".self_attn", ".mlp")):
                self.handles.append(module.register_forward_hook(self._activation_hook(name, index)))
            if name.endswith(".mlp.gate") and "Router" in type(module).__name__:
                self.handles.append(module.register_forward_hook(self._router_hook(name, index)))

    def _activation_hook(self, name, index):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            value = _local(value).detach()
            if value.ndim >= 3 and value.shape[0] == 1:
                value = value[0]
            self.values[name] = value.index_select(0, index).float().cpu()

        return hook

    def _router_hook(self, name, index):
        def hook(module, inputs, output):
            self.routes[name] = _local(output[2]).detach().index_select(0, index).cpu()

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


@dataclass(frozen=True)
class _Case:
    name: str
    shared: bool = True
    torch_deterministic: bool = True
    fa_deterministic: bool = True
    ordered_kv: bool = False
    ordered_loss: bool = False
    reverse: bool = False
    canonical: bool = False
    repeats: int = 2


def _cases(short):
    cases = [
        # Use a deterministic expanded reference. Native attention can itself
        # vary between repeats; mixing that noise into equivalence is misleading.
        _Case("expanded", shared=False),
        _Case("tree_native", torch_deterministic=False, fa_deterministic=False),
    ]
    if not short:
        cases.extend(
            [
                _Case("expanded_native", shared=False, torch_deterministic=False, fa_deterministic=False),
                _Case("tree_torch_deterministic", fa_deterministic=False),
                _Case("tree_fa_deterministic"),
                _Case("tree_ordered_kv", ordered_kv=True),
                _Case("tree_ordered_loss", ordered_loss=True),
            ]
        )
    cases.extend(
        [
            _Case("tree_ordered_both", ordered_kv=True, ordered_loss=True),
            _Case("tree_reverse", ordered_kv=True, ordered_loss=True, reverse=True, repeats=1),
            _Case("tree_canonical", ordered_kv=True, ordered_loss=True, canonical=True),
        ]
    )
    return cases


def _attention_ablation(kernel, deterministic):
    """Intentionally override automatic propagation only in this diagnostic.

    This keeps the Torch-only versus FA-deterministic causal comparison valid
    after the production adapter starts forwarding Torch's deterministic flag.
    """

    def call(*args, **kwargs):
        kwargs["deterministic"] = deterministic
        return kernel(*args, **kwargs)

    return call


def _data(prefix, suffix, branches, vocab):
    generator = torch.Generator().manual_seed(123)
    common = torch.randint(10, vocab, (prefix,), generator=generator).tolist()
    sequences = [
        common + torch.randint(10, vocab, (suffix + i % 3,), generator=generator).tolist() for i in range(branches)
    ]
    targets = [[-100] * (prefix - 1) + sequence[prefix:] + [-100] for sequence in sequences]
    return sequences, targets


def _prepare(sequences, targets, case, repeat):
    order = list(range(len(sequences)))
    if case.reverse or (case.canonical and repeat):
        order.reverse()
    if case.canonical:
        order.sort(key=lambda sample: (sequences[sample], targets[sample]))
    plan = build_prefix_tree([sequences[i] for i in order], share_prefixes=case.shared)
    batch = TreeSFTBatch.from_plan(
        plan,
        [targets[i] for i in order],
        device="cuda",
        deterministic_reduction=case.ordered_kv or case.ordered_loss,
    )
    if not case.ordered_kv:
        batch = replace(batch, attention=replace(batch.attention, kv_reduction=None))
    if not case.ordered_loss:
        batch = replace(batch, supervision=replace(batch.supervision, reduction=None))
    inverse = {sample: position for position, sample in enumerate(order)}
    occurrence_rows = [node for sample in range(len(order)) for node in plan.sample_nodes[inverse[sample]]]
    return batch, occurrence_rows


def _load(args):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    kwargs = {"dtype": torch.float32, "attn_implementation": "eager"}
    if world > 1:
        from torch.distributed.device_mesh import init_device_mesh

        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
        kwargs.update(tp_plan="auto", device_mesh=init_device_mesh("cuda", (world,), mesh_dim_names=("tp",)))
    print(f"Loading {args.model}, TP={world}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if world == 1:
        model.cuda()
    if hasattr(model, "set_experts_implementation"):
        model.set_experts_implementation("eager")
    model.train()
    enable_qwen_prefix_tree(model)
    if args.fp32_head:
        original = model.lm_head.forward

        def head_fp32(hidden_states):
            with torch.autocast("cuda", enabled=False):
                return original(hidden_states.float())

        model.lm_head.forward = head_fp32
    if world > 1 and getattr(model, "_tp_size", None) != world:
        raise AssertionError("requested tensor parallelism was not installed")
    return model, world


def run(args):
    torch.manual_seed(19)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.attention == "reference":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    model, world = _load(args)
    if args.attention == "fa4":
        from kernels import get_kernel

        varlen = get_kernel("kernels-community/flash-attn4", version=1).flash_attn_varlen_func
    else:
        varlen = None
    sequences, targets = _data(args.prefix, args.suffix, args.branches, model.config.vocab_size)
    reference, reference_trace, anchor, layout_anchor = None, None, None, None
    results = []
    cases = _cases(args.short)
    if args.focus:
        cases = [_Case("expanded", shared=False), _Case("tree_ordered_both", ordered_kv=True, ordered_loss=True)]
    if args.smoke:
        cases = [
            _Case("expanded", shared=False, repeats=1),
            _Case("tree_ordered_both", ordered_kv=True, ordered_loss=True),
        ]
    for case in cases:
        anchor = None
        gc.collect()
        for repeat in range(case.repeats):
            torch.use_deterministic_algorithms(case.torch_deterministic)
            batch, occurrence_rows = _prepare(sequences, targets, case, repeat)
            trace = _Trace(model, occurrence_rows, enabled=repeat == 0)
            model.zero_grad(set_to_none=True)
            function = _attention_ablation(varlen, case.fa_deterministic) if varlen else None
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                    loss = batch.loss(model, varlen_func=function)
                loss.backward()
                torch.cuda.synchronize()
            finally:
                trace.close()
            record = {
                "case": case.name,
                "repeat": repeat,
                "loss": loss.item(),
                "forward_backward_seconds": time.perf_counter() - start,
                "physical_tokens": len(batch.attention.plan.tokens),
                "replicated_qk_norm_gradient_max_abs": _replicated_norm_gradient_difference(model),
            }
            if reference is None:
                reference = _snapshot(model)
                reference_trace = trace
                record["versus_expanded"] = {"bitwise_equal": True, "relative_l2": 0.0, "max_abs": 0.0}
            else:
                record["versus_expanded"] = _gradient_difference(model, reference)
            if anchor is not None:
                record["versus_same_layout_first_run"] = _gradient_difference(model, anchor)
                record["repeatability_expected"] = case.torch_deterministic and (
                    varlen is None or case.fa_deterministic
                )
            elif case.repeats > 1:
                anchor = reference if case.name == "expanded" else _snapshot(model)
            if case.name == "tree_ordered_both" and repeat == 0:
                layout_anchor = anchor
            if case.name == "tree_reverse" and layout_anchor is not None:
                record["versus_original_tree_layout"] = _gradient_difference(model, layout_anchor)
                layout_anchor = None
            if repeat == 0:
                record["activations"] = {
                    name: _tensor_difference(value, reference_trace.values[name])
                    for name, value in trace.values.items()
                }
                record["routing"] = {
                    name: _tensor_difference(value, reference_trace.routes[name])
                    for name, value in trace.routes.items()
                }
            results.append(record)
            print(json.dumps({k: v for k, v in record.items() if k not in ("activations", "routing")}), flush=True)
    violations = [
        row["case"]
        for row in results
        if row.get("repeatability_expected") and not row["versus_same_layout_first_run"]["bitwise_equal"]
    ]
    return {
        "model": args.model,
        "revision": getattr(model.config, "_commit_hash", None),
        "tensor_parallel_size": world,
        "precision": args.precision,
        "attention": args.attention,
        "fp32_head": args.fp32_head,
        "expanded_reference_deterministic": True,
        "repeatability_passed": not violations,
        "repeatability_failures": violations,
        "local_parameter_elements": sum(_local(p).numel() for p in model.parameters()),
        "prefix": args.prefix,
        "suffix": args.suffix,
        "branches": args.branches,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prefix", type=int, default=513)
    parser.add_argument("--suffix", type=int, default=65)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--attention", choices=["fa4", "reference"], default="fa4")
    parser.add_argument("--short", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--focus", action="store_true", help="Compare only deterministic expanded/tree repeats")
    parser.add_argument("--fp32-head", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args)
        if not dist.is_initialized() or dist.get_rank() == 0:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        if not report["repeatability_passed"]:
            raise AssertionError(f"deterministic repeatability failed: {report['repeatability_failures']}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
