"""CP is TWO orthogonal axes, and the contract must check both without conflating either with degree.

    train_cp_comm_type  -> how ATTENTION communicates.   "a2a" = Ulysses, "p2p" = ring.
    train_allgather_cp  -> miles' SEQUENCE LAYOUT.        contiguous chunks vs the zigzag ring layout.

Two failures this pins, both of which have happened:

1. `uses_ulysses_cp` was once just `context_parallel_size > 1`, so ANY cp>1 was validated against the
   `ulysses_cp` capability -- making CP unreachable for every DSA family even though the plugin
   implemented allgather-CP. The profile had no way to say "supports CP, but not that flavour".
2. `--cp-comm-type a2a` was set on a launcher args object that never reached the training command
   line. Megatron's own default is `["p2p"]`, so an unemitted default silently selects ring -- the
   one scheme that can never be bitwise. A default invisible where it matters is not a default.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from miles.true_on_policy.config import build_true_on_policy_config
from miles.true_on_policy.model_profiles import QWEN3_DENSE_PROFILE


def _args(**overrides):
    values = {
        "true_on_policy": True,
        "model_name": "Qwen3-4B",
        "train_backend": "megatron",
        "tensor_model_parallel_size": 1,
        "context_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "rollout_num_gpus_per_engine": 1,
        "true_on_policy_contract": None,
        "cp_comm_type": "a2a",
        "allgather_cp": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# --- axis 1: the attention comm type ----------------------------------------------------------

def test_ulysses_is_derived_from_the_comm_type_not_the_degree():
    layout = build_true_on_policy_config(_args(context_parallel_size=4)).parallel_layout
    assert layout.uses_train_cp
    assert layout.train_cp_attention_scheme == "ulysses"
    assert layout.required_cp_layout == "ulysses_cp"
    assert layout.uses_ulysses_cp


def test_cp_degree_one_is_not_a_cp_program_whatever_the_comm_type():
    for comm in ("a2a", "p2p", None):
        layout = build_true_on_policy_config(
            _args(context_parallel_size=1, cp_comm_type=comm)
        ).parallel_layout
        assert not layout.uses_train_cp
        assert not layout.uses_ulysses_cp
        # and it must VALIDATE -- cp=1 never touches the CP clauses
        build_true_on_policy_config(_args(context_parallel_size=1, cp_comm_type=comm)).validate()


@pytest.mark.parametrize("comm", ["p2p", "a2a+p2p", None])
def test_ring_is_refused_by_name_including_megatrons_default(comm):
    config = build_true_on_policy_config(_args(cp_comm_type=comm))
    assert config.parallel_layout.train_cp_attention_scheme == "ring"
    with pytest.raises(ValueError, match="online softmax"):
        config.validate()


def test_all_gather_is_its_OWN_scheme_not_ring():
    """all_gather adds no reduction; it runs megatron's EAGER attention, the wrong KERNEL.

    Folding it into "ring" mislabelled the reason for refusing it.
    """
    config = build_true_on_policy_config(_args(cp_comm_type="all_gather"))
    assert config.parallel_layout.train_cp_attention_scheme == "allgather"
    with pytest.raises(ValueError, match="does not support allgather-CP"):
        config.validate()


def test_the_refusal_explains_where_the_declaration_LIVES():
    """megatron's --cp-comm-type cannot carry it: under transformer_impl=local megatron demands
    all_gather at cp>1 for its own attention, which TOP replaces."""
    with pytest.raises(ValueError) as exc:
        build_true_on_policy_config(_args(cp_comm_type=None)).validate()
    message = str(exc.value)
    assert "p2p" in message
    assert "--transformer-impl local" in message


def test_per_layer_comm_type_is_refused_unless_uniform():
    build_true_on_policy_config(_args(cp_comm_type=["a2a", "a2a"])).validate()
    with pytest.raises(ValueError, match="per-layer"):
        build_true_on_policy_config(_args(cp_comm_type=["a2a", "p2p"]))


# --- axis 2: the sequence layout --------------------------------------------------------------

def test_sequence_layout_is_independent_of_the_comm_type():
    zig = build_true_on_policy_config(_args(allgather_cp=False)).parallel_layout
    gather = build_true_on_policy_config(_args(allgather_cp=True)).parallel_layout
    assert zig.train_cp_sequence_layout == "zigzag"
    assert gather.train_cp_sequence_layout == "allgather"
    # Both are Ulysses on axis 1 -- the axes do not constrain each other.
    assert zig.train_cp_attention_scheme == gather.train_cp_attention_scheme == "ulysses"


def test_ulysses_requires_zigzag_layout():
    assert not QWEN3_DENSE_PROFILE.requires_allgather_cp
    build_true_on_policy_config(_args(allgather_cp=False)).validate()
    with pytest.raises(ValueError, match="requires per-sequence zigzag"):
        build_true_on_policy_config(_args(allgather_cp=True)).validate()


def test_a_family_declaring_allgather_cp_refuses_the_zigzag_layout():
    from dataclasses import replace

    dsa_like = replace(
        QWEN3_DENSE_PROFILE,
        supported_train_layouts=QWEN3_DENSE_PROFILE.supported_train_layouts + ("allgather_cp",),
    )
    assert dsa_like.requires_allgather_cp
    config = build_true_on_policy_config(_args(allgather_cp=False, cp_comm_type="all_gather"))
    strict = type(config)(**{**config.__dict__, "model_profile": dsa_like})
    with pytest.raises(ValueError, match="allgather-cp"):
        strict.validate()
    ok = type(config)(**{**config.__dict__, "model_profile": dsa_like, "allgather_cp": True})
    ok.validate()


# --- the flag must reach the program ----------------------------------------------------------

def test_a2a_is_emitted_because_megatrons_parser_defaults_to_p2p():
    """An unemitted declaration is not a declaration: megatron's parser defaults this to ["p2p"]
    and forwards it, so leaving it off silently selects ring."""
    plan = build_true_on_policy_config(_args(context_parallel_size=2)).build_launch_plan()
    values = plan.miles_args.values
    assert "--cp-comm-type" in values
    assert values[values.index("--cp-comm-type") + 1] == "a2a"


def test_nothing_cp_is_emitted_at_cp_1():
    plan = build_true_on_policy_config(
        _args(context_parallel_size=1, cp_comm_type=None)
    ).build_launch_plan()
    assert "--cp-comm-type" not in plan.miles_args.values
    assert "--allgather-cp" not in plan.miles_args.values
