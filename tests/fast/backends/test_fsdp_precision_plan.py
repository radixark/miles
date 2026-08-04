"""Compiling PrecisionSpec rules into FSDP2 wrap units.

Every test uses this model with default_dtype=bf16, and every docstring draws the resulting gather
dtype per node (`[U]` = the module becomes its own wrap unit). `layers.1` mirrors `layers.0`, so most
diagrams only draw layer 0. The shape mimics a decoder stack: a block per layer, norms at two depths,
and a buffer-only rotary module.

    Tiny                            classes and own float tensors
    └── layers          ModuleList  -
        ├── 0           Block       -
        │   ├── proj    Linear      weight, bias
        │   ├── norm    LayerNorm   weight, bias
        │   ├── attn    Attn        -
        │   │   └── q_norm  LayerNorm  weight, bias
        │   └── rotary  Rotary      inv_freq (buffer only)
        └── 1           Block       (same)
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import pytest
import torch
import torch.nn as nn

from miles.backends.experimental.fsdp_utils.adaptations.precision import (
    ModuleSel,
    PrecisionSpec,
    Rule,
    compile_precision,
    parse_precision_rules,
)


class Rotary(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("inv_freq", torch.zeros(4))


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_norm = nn.LayerNorm(8)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
        self.attn = Attn()
        self.rotary = Rotary()


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block(), Block()])


def _units(compiled):
    return {unit.fqn: unit.param_dtype for unit in compiled.wrap_units}


def _plan(model, compiled):
    return [(unit.fqn, unit.param_dtype) for unit in compiled.wrap_plan(model, list(model.layers))]


NORM_FQNS = {f"layers.{i}{suffix}" for i in range(2) for suffix in (".norm", ".attn.q_norm")}


def test_empty_spec_compiles_to_nothing():
    """No rules, so every node keeps the default and nothing is emitted.

    layers.0            bf16
    ├── proj            bf16
    ├── norm            bf16
    ├── attn            bf16
    │   └── q_norm      bf16
    └── rotary          bf16
    """
    compiled = compile_precision(Tiny(), PrecisionSpec(), default_dtype=torch.bfloat16)
    assert compiled.wrap_units == []


def test_fqn_glob_selects_norms_across_depths():
    """`*norm*` crosses dots, so one rule catches both norm depths and skips the Linear siblings.

    Rule(fqn="*norm*", gather=fp32)

    layers.0            bf16
    ├── proj            bf16
    ├── norm      [U]   fp32
    ├── attn            bf16
    │   └── q_norm [U]  fp32
    └── rotary          bf16
    """
    spec = PrecisionSpec(rules=(Rule(ModuleSel(fqn="*norm*"), gather="fp32"),))
    compiled = compile_precision(Tiny(), spec, default_dtype=torch.bfloat16)
    assert _units(compiled) == dict.fromkeys(NORM_FQNS, torch.float32)


def test_cls_glob_selects_by_class():
    """Selecting by class name reaches the same two norms without naming any path.

    Rule(cls="*LayerNorm", gather=fp32)

    layers.0            bf16
    ├── norm      [U]   fp32
    └── attn
        └── q_norm [U]  fp32
    """
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="*LayerNorm"), gather="fp32"),))
    compiled = compile_precision(Tiny(), spec, default_dtype=torch.bfloat16)
    assert set(_units(compiled)) == NORM_FQNS


def test_rule_covers_the_matched_subtree():
    """A rule on a container hands its dtype to everything below, so one unit covers the subtree.

    Rule(fqn="layers.1", gather=fp32)

    layers.0            bf16      layers.1        [U] fp32
    ├── proj            bf16      ├── proj            fp32  (inherits, no unit)
    ├── norm            bf16      ├── norm            fp32  (inherits, no unit)
    └── attn            bf16      └── attn            fp32  (inherits, no unit)
        └── q_norm      bf16          └── q_norm      fp32  (inherits, no unit)
    """
    model = Tiny()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(fqn="layers.1"), gather="fp32"),))
    compiled = compile_precision(model, spec, default_dtype=torch.bfloat16)
    assert _units(compiled) == {"layers.1": torch.float32}
    assert compiled.gather_dtypes["layers.1.attn.q_norm"] is torch.float32
    assert compiled.gather_dtypes["layers.0.attn.q_norm"] is torch.bfloat16


def test_later_rule_overrides_earlier_selection():
    """Both rules select layer 0's norms; the later one wins there while layer 1 keeps the first.

    Rule(cls="LayerNorm",        gather=fp32)   # rule 1
    Rule(fqn="layers.0.*norm*",  gather=fp16)   # rule 2, wins where they overlap

    layers.0                      layers.1
    ├── norm      [U]   fp16      ├── norm      [U]   fp32
    └── attn                      └── attn
        └── q_norm [U]  fp16          └── q_norm [U]  fp32
    """
    spec = PrecisionSpec(
        rules=(
            Rule(ModuleSel(cls="LayerNorm"), gather="fp32"),
            Rule(ModuleSel(fqn="layers.0.*norm*"), gather="fp16"),
        )
    )
    compiled = compile_precision(Tiny(), spec, default_dtype=torch.bfloat16)
    assert _units(compiled) == {
        "layers.0.norm": torch.float16,
        "layers.0.attn.q_norm": torch.float16,
        "layers.1.norm": torch.float32,
        "layers.1.attn.q_norm": torch.float32,
    }


def test_empty_module_sel_rejected():
    """A selector with neither fqn nor cls would silently match every module."""
    with pytest.raises(ValueError, match="needs fqn or cls"):
        ModuleSel()


def test_every_node_of_a_nested_chain_wraps_bottom_up():
    """Three nested rules that each differ from their parent need one unit per node, and the plan
    hands them back deepest first so each outer wrap excludes the inner ones.

        Rule(fqn="layers.0",              gather=fp16)
        Rule(fqn="layers.0.attn",         gather=fp32)
        Rule(fqn="layers.0.attn.q_norm",  gather=default)   # carved back out

        layers.0        [U] fp16   wrap order 3
        ├── proj            fp16   (inherits layers.0)
        ├── norm            fp16   (inherits layers.0)
        ├── attn        [U] fp32   wrap order 2
        │   └── q_norm  [U] bf16   wrap order 1, wraps first
        └── rotary          buffer only, never gathered
        layers.1        [U] bf16   block unit only, at the default dtype
    """
    model = Tiny()
    spec = PrecisionSpec(
        rules=(
            Rule(ModuleSel(fqn="layers.0"), gather="fp16"),
            Rule(ModuleSel(fqn="layers.0.attn"), gather="fp32"),
            Rule(ModuleSel(fqn="layers.0.attn.q_norm"), gather="default"),
        )
    )
    compiled = compile_precision(model, spec, default_dtype=torch.bfloat16)
    assert _units(compiled) == {
        "layers.0.attn.q_norm": torch.bfloat16,
        "layers.0.attn": torch.float32,
        "layers.0": torch.float16,
    }
    assert _plan(model, compiled) == [
        ("layers.0.attn.q_norm", torch.bfloat16),
        ("layers.0.attn", torch.float32),
        ("layers.0", torch.float16),
        ("layers.1", torch.bfloat16),
    ]


def test_block_inside_an_override_wraps_at_the_override_dtype():
    """The rule sits above the block units, so the blocks wrap deeper than the override; at the
    default dtype they would be the innermost wrap and silently undo it.

    Rule(fqn="layers", gather=fp32)

    layers          [U] fp32   wrap order 3 (the override)
    ├── 0               fp32   wrap order 1, block unit forced to fp32
    └── 1               fp32   wrap order 2, block unit forced to fp32
    """
    model = Tiny()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(fqn="layers"), gather="fp32"),))
    compiled = compile_precision(model, spec, default_dtype=torch.bfloat16)
    assert _units(compiled) == {"layers": torch.float32}
    assert _plan(model, compiled) == [
        ("layers.0", torch.float32),
        ("layers.1", torch.float32),
        ("layers", torch.float32),
    ]


def test_paramless_module_gets_no_unit():
    """Buffers are never gathered, so pinning a buffer-only module lowers to nothing.

    Rule(cls="Rotary", gather=fp32)

    layers.0
    └── rotary.inv_freq     buffer -> no unit
    """
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="Rotary"), gather="fp32"),))
    compiled = compile_precision(Tiny(), spec, default_dtype=torch.bfloat16)
    assert compiled.wrap_units == []


def test_unmatched_rule_rejected():
    """A rule matching nothing is a typo'd pattern or class name, not a silent no-op."""
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="NoSuchModule"), gather="fp32"),))
    with pytest.raises(ValueError, match="matched no module"):
        compile_precision(Tiny(), spec, default_dtype=torch.bfloat16)


def test_cli_rules_parse_into_spec_rules():
    assert parse_precision_rules(None) == ()
    assert parse_precision_rules("cls:*LayerNorm=fp32, fqn:layers.0.attn=default") == (
        Rule(ModuleSel(cls="*LayerNorm"), gather="fp32"),
        Rule(ModuleSel(fqn="layers.0.attn"), gather="default"),
    )


@pytest.mark.parametrize(
    "text",
    ["*LayerNorm=fp32", "cls:*LayerNorm", "cls:=fp32", "cls:*LayerNorm=float32"],
)
def test_malformed_cli_rule_rejected(text):
    with pytest.raises(ValueError):
        parse_precision_rules(text)
