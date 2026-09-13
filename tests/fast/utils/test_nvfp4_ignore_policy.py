"""Exact module-prefix policy matching without the optional Megatron imports."""

import importlib.util
from pathlib import Path
import re

import pytest
import torch


path = Path(__file__).resolve().parents[3] / "miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_nvfp4.py"
spec = importlib.util.spec_from_file_location("nvfp4_ignore_policy", path)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


@pytest.mark.parametrize(
    "rules",
    [[], ["model.layers.1"], ["model.layers.1."], ["", "."], ["a", "a", "ab"], ["literal*", "a[0]"], ["re:^model\\.layers\\.[12]\\.", "a"], ["a", "re:["], ["re:[", "a"]],
)
def test_exact_literal_and_regex_short_circuit(rules):
    compiled = policy._literal_ignore_rules(tuple(rules))
    for name in ("", ".", ".x", "a", "a.", "a..b", "ab", "abc", "a[0].weight", "literal*.weight", "model.layers.1", "model.layers.1.weight", "model.layers.10.weight", "model.layers.1..weight"):
        try:
            expected = policy._is_ignored(name, rules)
        except re.error:
            with pytest.raises(re.error):
                policy._is_ignored(name, rules, compiled)
        else:
            assert policy._is_ignored(name, rules, compiled) == expected


def test_mutable_config_content_invalidates_without_mutating_config():
    config = {"ignore": ["a"], "exclude_modules": ["b", "a"]}
    original = {key: list(value) for key, value in config.items()}
    rules = policy._get_ignore_rules(config)
    first = policy._literal_ignore_rules(tuple(rules))
    assert rules == ["a", "b"] and config == original
    config["ignore"][0] = "c"
    config["exclude_modules"].append("d")
    rules = policy._get_ignore_rules(config)
    second = policy._literal_ignore_rules(tuple(rules))
    assert second == frozenset({"a", "b", "c", "d"}) and second != first
    config["ignore"] = "new"
    config["exclude_modules"] = []
    rules = policy._get_ignore_rules(config)
    assert rules == ["new"]
    assert not policy._is_ignored("a.weight", rules, policy._literal_ignore_rules(tuple(rules)))
    for index in range(32):
        policy._literal_ignore_rules((str(index),))
    assert policy._literal_ignore_rules.cache_info().currsize <= 16


def test_real_moe_entry_keeps_ignored_weights_and_quantizes_complete_pair(monkeypatch):
    calls = []
    pair = torch.zeros((32, 16), dtype=torch.bfloat16)
    gate, up = pair.chunk(2)
    down = torch.zeros((16, 16), dtype=torch.bfloat16)
    base = "model.layers.3.mlp.experts.0"
    names = [base + suffix for suffix in (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")]
    inputs = list(zip(names, (gate, up, down), strict=True))

    def quantize_pair(first, second):
        calls.append((first, second))
        return (first, first, first), (second, second, second)

    monkeypatch.setattr(policy, "nvfp4_quantize_1d_pair", quantize_pair)
    rules = [base + ".down_proj"]
    outputs = policy._quantize_moe_params(inputs, rules)
    assert len(calls) == 1 and calls[0][0] is gate and calls[0][1] is up
    assert [name for name, _ in outputs] == [
        names[0],
        names[0].replace(".weight", ".weight_scale"),
        names[0].replace(".weight", ".weight_scale_2"),
        names[1],
        names[1].replace(".weight", ".weight_scale"),
        names[1].replace(".weight", ".weight_scale_2"),
        names[2],
    ]
    assert outputs[-1][1] is down
    rules[:] = [base]
    assert all(actual is expected for (_, actual), (_, expected) in zip(policy._quantize_moe_params(inputs, rules), inputs, strict=True))
    assert len(calls) == 1
