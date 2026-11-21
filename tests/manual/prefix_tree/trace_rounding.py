"""Trace the first forward/backward difference with small, unchanged HF models.

Run on CPU: python tests/manual/prefix_tree/trace_rounding.py --output /tmp/trace.json
Compare adjoints after summing expanded occurrences into their compact tree row;
comparing a parent's summed adjoint with one child's adjoint would be incorrect.
FP64 replaces only internally-FP32 RMSNorms with an FP64 mathematical oracle.
No model weights are updated, and reports are external to the source tree.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree
from miles.backends.training_utils.prefix_tree.transformers import TreeSFTBatch, enable_transformers_prefix_tree


def _difference(actual, expected):
    a, b = actual.double(), expected.double()
    return {
        "relative_l2": float((a - b).norm() / b.norm().clamp_min(1e-30)),
        "max_abs": float((a - b).abs().max()),
        "bitwise_equal": torch.equal(actual, expected),
    }


def _fp64_norms(module):
    for name, child in module.named_children():
        if "RMSNorm" in type(child).__name__ and hasattr(child, "variance_epsilon"):
            replacement = torch.nn.RMSNorm(child.weight.numel(), eps=child.variance_epsilon, dtype=torch.float64)
            replacement.weight.data.copy_(child.weight.data)
            setattr(module, name, replacement)
        else:
            _fp64_norms(child)


class _FP32Output(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states):
        with torch.autocast(hidden_states.device.type, enabled=False):
            return self.layer(hidden_states.float())


def _model(family, precision, layers):
    torch.manual_seed(37)
    config_type, model_type = {
        "llama": (LlamaConfig, LlamaForCausalLM),
        "qwen_moe": (Qwen3MoeConfig, Qwen3MoeForCausalLM),
    }[family]
    config = config_type(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        attention_dropout=0.0,
        sliding_window=None,
    )
    config._attn_implementation = "eager"
    model = model_type(config).to(dtype=torch.float64 if precision == "fp64" else torch.float32).train()
    if hasattr(model, "set_experts_implementation"):
        # The grouped-MM backend does not support FP64; keep one expert
        # implementation across precisions for this mathematical comparison.
        model.set_experts_implementation("eager")
    if precision == "fp64":
        _fp64_norms(model)
    if precision == "bf16_fp32_head":
        model.set_output_embeddings(_FP32Output(model.get_output_embeddings()))
    enable_transformers_prefix_tree(model)
    return model


class _Trace:
    def __init__(self, model, tokens):
        self.values, self.adjoints, self.routes, self.handles = {}, {}, {}, []
        self.tokens = tokens
        for name, module in model.named_modules():
            if name.endswith(("self_attn", ".mlp", "input_layernorm", "post_attention_layernorm", "lm_head")):
                self.handles.append(module.register_forward_hook(self._hook(name)))
            if name.endswith(".mlp.gate"):
                self.handles.append(module.register_forward_hook(self._router(name)))

    def _hook(self, name):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor):
                return
            self.values[name] = value.detach().reshape(self.tokens, -1).clone()
            if value.requires_grad:

                def adjoint(gradient):
                    self.adjoints[name] = gradient.detach().reshape(self.tokens, -1).clone()

                value.register_hook(adjoint)

        return hook

    def _router(self, name):
        def hook(module, inputs, output):
            self.routes[name] = output[2].detach().reshape(self.tokens, -1).clone()

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def _run(model, sequences, targets, shared, precision, ordered):
    plan = build_prefix_tree(sequences, share_prefixes=shared)
    batch = TreeSFTBatch.from_plan(plan, targets, device="cpu", deterministic_reduction=ordered)
    model.zero_grad(set_to_none=True)
    trace = _Trace(model, len(plan.tokens))
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision.startswith("bf16")):
            loss = batch.loss(model)
        loss.backward()
    finally:
        trace.close()
    return {
        "loss": float(loss.detach()),
        "plan": plan,
        "trace": trace,
        "gradients": {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None},
    }


def _compare(actual, expanded):
    trace, reference = actual["trace"], expanded["trace"]
    occurrences = torch.tensor([node for path in actual["plan"].sample_nodes for node in path])
    adjoints = {}
    for name, value in trace.adjoints.items():
        expected = torch.zeros_like(value, dtype=torch.float64)
        expected.index_add_(0, occurrences, reference.adjoints[name].double())
        adjoints[name] = _difference(value, expected)
    gradients, reference_gradients = actual["gradients"], expanded["gradients"]
    assert gradients.keys() == reference_gradients.keys()
    numerator = sum(float((gradients[n].double() - reference_gradients[n].double()).square().sum()) for n in gradients)
    denominator = sum(float(g.double().square().sum()) for g in reference_gradients.values())
    return {
        "loss": actual["loss"],
        "gradient_relative_l2": (numerator / max(denominator, 1e-30)) ** 0.5,
        "forward": {
            name: _difference(value[occurrences], reference.values[name]) for name, value in trace.values.items()
        },
        "backward_fan_in": adjoints,
        "parameters": {name: _difference(g, reference_gradients[name]) for name, g in gradients.items()},
        "routing": {
            name: {
                "changed_rows": int(
                    (value[occurrences].sort(-1).values != reference.routes[name].sort(-1).values).any(-1).sum()
                ),
                "rows": len(occurrences),
            }
            for name, value in trace.routes.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(71)
    prefix = torch.randint(1, 96, (129,), generator=generator).tolist()
    sequences = [prefix + [20 + i] + torch.randint(1, 96, (16,), generator=generator).tolist() for i in range(8)]
    targets = [[-100] * 128 + sequence[129:] + [-100] for sequence in sequences]
    reports = []
    for family in ("llama", "qwen_moe"):
        for precision in ("fp64", "fp32", "bf16", "bf16_fp32_head"):
            model = _model(family, precision, args.layers)
            cold_expanded = _run(model, sequences, targets, False, precision, True)
            expanded = _run(model, sequences, targets, False, precision, True)
            comparisons = {"cold_expanded": _compare(cold_expanded, expanded)}
            for name, shared, ordered in (
                ("expanded_repeat", False, True),
                ("tree", True, True),
                ("tree_repeat", True, True),
                ("tree_unordered", True, False),
            ):
                actual = _run(model, sequences, targets, shared, precision, ordered)
                comparisons[name] = _compare(actual, expanded)
            row = {"family": family, "precision": precision, "layers": args.layers, "comparisons": comparisons}
            reports.append(row)
            print(
                json.dumps(
                    {
                        "family": family,
                        "precision": precision,
                        "gradient_errors": {k: v["gradient_relative_l2"] for k, v in comparisons.items()},
                    }
                ),
                flush=True,
            )
            args.output.write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
