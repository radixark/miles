"""targets_expert_leaves gates the MoE-specific multi-LoRA handling (permute-fusion
off, expert validations); a false negative silently skips those."""

from miles.utils.multi_lora import targets_expert_leaves


def test_mlp_leaf_names_target_experts():
    # These names match the dense MLP and the routed experts alike.
    assert targets_expert_leaves(["gate_proj", "up_proj", "down_proj"])
    assert targets_expert_leaves(["linear_fc1"])
    assert targets_expert_leaves(["linear_fc2"])


def test_expert_scoped_wildcards_target_experts():
    assert targets_expert_leaves(["*.layers.*.mlp.experts.linear_fc1"])


def test_attention_only_targets_do_not():
    assert not targets_expert_leaves(["linear_qkv", "linear_proj"])
    assert not targets_expert_leaves(["q_proj", "k_proj", "v_proj", "o_proj"])


def test_bulk_aliases_target_experts():
    # "all" is only resolved by the later target-module conversion, so the alias itself counts.
    for alias in ("all", "all-linear", "all_linear", "ALL"):
        assert targets_expert_leaves([alias]), alias


def test_bare_string_is_accepted():
    assert targets_expert_leaves("gate_proj")
    assert not targets_expert_leaves("linear_qkv")


def test_empty_targets_do_not():
    assert not targets_expert_leaves(None)
    assert not targets_expert_leaves([])
