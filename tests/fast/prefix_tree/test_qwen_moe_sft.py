import copy

import torch
import torch.nn.functional as F
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree
from miles.backends.training_utils.prefix_tree.qwen import TreeSFTBatch, enable_qwen_prefix_tree


def test_qwen_moe_expert_routes_and_all_gradients_match_expanded_sft():
    torch.manual_seed(83)
    config = Qwen3MoeConfig(
        vocab_size=17,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        attention_dropout=0,
        sliding_window=None,
    )
    config._attn_implementation = "eager"
    expanded = Qwen3MoeForCausalLM(config).train()
    compact = copy.deepcopy(expanded)
    enable_qwen_prefix_tree(compact)
    sequences = [[1, 2, 3], [1, 2, 4, 5], [1, 2, 3]]
    targets = [[2, 3, -100], [2, 4, 5, -100], [2, 3, -100]]
    plan = build_prefix_tree(sequences)
    batch = TreeSFTBatch.from_plan(plan, targets, device="cpu", deterministic_reduction=True)
    traces = []
    handles = []
    for model in (expanded, compact):
        trace = {}
        for name, module in model.named_modules():
            if name.endswith(".mlp.gate"):
                trace[name] = []
                handles.append(
                    module.register_forward_hook(
                        lambda module, inputs, output, rows=trace[name]: rows.append(output[2].detach()),
                    )
                )
        traces.append(trace)
    try:
        compact_loss = batch.loss(compact)
        terms = [
            F.cross_entropy(
                expanded(torch.tensor([seq]), use_cache=False).logits[0], torch.tensor(labels), reduction="sum"
            )
            for seq, labels in zip(sequences, targets, strict=True)
        ]
        expanded_loss = sum(terms) / 7
        torch.testing.assert_close(compact_loss, expanded_loss, atol=1e-6, rtol=1e-6)
        compact_loss.backward()
        expanded_loss.backward()
        for actual, expected in zip(compact.parameters(), expanded.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)
        occurrence_rows = [node for nodes in plan.sample_nodes for node in nodes]
        for name in traces[0]:
            assert torch.equal(torch.cat(traces[0][name]), traces[1][name][0][occurrence_rows])
    finally:
        for handle in handles:
            handle.remove()
