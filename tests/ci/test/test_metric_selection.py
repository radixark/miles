"""Offline unit tests for step selection, constraints, and register_ci_gate
parsing (including the canonical declaration keys)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from tests.ci.ci_register import CIRegistry, HWBackend, register_cpu_ci, ut_parse_one_file
from tests.ci.metric_history.constraints import evaluate_constraint
from tests.ci.metric_history.register import GATE_DEFAULTS, parse_ci_gate_specs
from tests.ci.metric_history.selection import SelectionError, select

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

# --- step selection -----------------------------------------------------------


def test_last_picks_last_numeric_point():
    got = select([[0, 0.1], [1, 0.2], [2, 0.35]], "last")
    assert len(got) == 1
    # Identity step is the -1 reduction sentinel; the landing step is reporting.
    assert got[0].step == -1
    assert got[0].at_step == 2
    assert got[0].value == pytest.approx(0.35)


def test_selection_skips_bool_and_none_values():
    # bool sneaks through isinstance(int); None is not a number. Non-numeric
    # points are ignored -- only numeric points count toward selection.
    series = [[0, True], [1, None], [2, 2.5]]
    got = select(series, "last")
    assert got[0].value == pytest.approx(2.5)


def test_last_non_finite_at_selected_point_errors():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(SelectionError, match="non-finite"):
            select([[0, 1.0], [1, bad]], "last")


def test_last_ignores_non_selected_non_finite():
    # A mid-series NaN is not the selected coordinate: last gates the actual
    # last point, which is finite here.
    got = select([[0, float("nan")], [1, 2.5]], "last")
    assert got[0].at_step == 1
    assert got[0].value == pytest.approx(2.5)


def test_all_non_finite_errors():
    with pytest.raises(SelectionError, match="all: non-finite value nan at step 1"):
        select([[0, 1.0], [1, float("nan")]], "all")


def test_steps_non_finite_at_named_step_errors():
    with pytest.raises(SelectionError, match="non-finite value -inf at required step 1"):
        select([[0, 1.0], [1, float("-inf")]], [1])


def test_empty_series_errors_clearly():
    for steps in ("last", "all", [0]):
        with pytest.raises(SelectionError):
            select([], steps)


def test_all_fans_out_one_extraction_per_step():
    got = select([[0, 1.0], [1, 2.0], [2, 3.0]], "all")
    assert [(e.step, e.at_step, e.value) for e in got] == [
        (0, 0, 1.0),
        (1, 1, 2.0),
        (2, 2, 3.0),
    ]


def test_all_null_step_errors():
    with pytest.raises(SelectionError, match="no step index"):
        select([[0, 1.0], [None, 2.0]], "all")


def test_all_duplicate_step_errors():
    with pytest.raises(SelectionError, match="duplicate step"):
        select([[0, 1.0], [0, 2.0]], "all")


def test_steps_picks_named_steps():
    got = select([[0, 0.001], [1, 0.5], [2, 0.9]], [0, 2])
    assert [(e.step, e.value) for e in got] == [(0, 0.001), (2, 0.9)]


def test_steps_missing_named_step_errors():
    with pytest.raises(SelectionError, match="required step 3 missing"):
        select([[0, 0.001], [1, 0.5]], [3])


# --- constraints --------------------------------------------------------------


def test_symmetric_corridor():
    c = {"rel_up": 0.20, "abs_floor_up": 0.0, "rel_down": 0.20, "abs_floor_down": 0.0}
    assert evaluate_constraint(c, 1.1, 1.0).ok
    assert not evaluate_constraint(c, 1.3, 1.0).ok
    assert not evaluate_constraint(c, 0.7, 1.0).ok


def test_band_scales_from_ref_only():
    # The tolerance is fixed by history: a deviating cur must not widen its
    # own band (a max over both magnitudes would pass 0.99 here, an effective
    # ceiling of 2x ref instead of the declared 1.5x).
    c = {"rel_up": 0.50, "abs_floor_up": 0.0, "rel_down": 0.50, "abs_floor_down": 0.0}
    assert evaluate_constraint(c, 0.74, 0.5).ok
    assert not evaluate_constraint(c, 0.76, 0.5).ok
    assert not evaluate_constraint(c, 0.99, 0.5).ok


def test_abs_floor_covers_near_zero():
    # A reference near zero makes the relative band vanish; the floor remains.
    c = {"rel_up": 0.20, "abs_floor_up": 1e-6, "rel_down": 0.20, "abs_floor_down": 1e-6}
    assert evaluate_constraint(c, 1e-7, 0.0).ok
    assert not evaluate_constraint(c, 0.5, 0.0).ok


def test_each_side_band_is_max_of_rel_and_floor():
    c = {"rel_up": 0.30, "abs_floor_up": 0.1, "rel_down": 0.10, "abs_floor_down": 0.1}
    out = evaluate_constraint(c, 0.5, 0.5)
    assert out.hi == pytest.approx(0.65)  # rel side wins: 0.3*0.5 > 0.1
    assert out.lo == pytest.approx(0.4)  # floor wins: 0.1*0.5 < 0.1


def test_asymmetric_sides():
    # Tight upper band, loose-but-finite lower band: a moderate drop passes,
    # a rise beyond band and a collapse far below both fail.
    c = {"rel_up": 0.10, "abs_floor_up": 0.0, "rel_down": 0.80, "abs_floor_down": 0.0}
    assert evaluate_constraint(c, 2.1, 2.0).ok
    assert not evaluate_constraint(c, 2.3, 2.0).ok
    assert evaluate_constraint(c, 0.5, 2.0).ok
    assert not evaluate_constraint(c, 0.3, 2.0).ok


# --- register_ci_gate parsing -----------------------------------------------


def _make_fixture(body: str, tmp_path: Path, name: str = "test_gatefix.py") -> str:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body).lstrip("\n"))
    return str(p)


def test_parse_single_spec_with_defaults(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.ci_register import register_cuda_ci
        from tests.ci.metric_history import register_ci_gate
        register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100")
        register_ci_gate(
            metric_key="train/grad_norm",
            steps="all",
            constraint={"rel_up": 0.20, "rel_down": 0.20},
        )
        """,
        tmp_path,
    )
    specs = parse_ci_gate_specs(path)
    assert len(specs) == 1
    s = specs[0]
    assert s.metric_key == "train/grad_norm"
    assert s.steps == "all"
    assert s.constraint == {"rel_up": 0.20, "abs_floor_up": 0.0, "rel_down": 0.20, "abs_floor_down": 0.0}
    assert s.enforce is False
    assert s.allowlist_reason is None
    assert s.filename == path


def test_parse_all_fields(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(
            metric_key="train/ppo_kl",
            steps=[0, 1],
            constraint={"abs_floor_up": 1e-6, "rel_up": 0.5, "rel_down": 0.8},
            enforce=True,
            allowlist_reason="known noisy",
        )
        """,
        tmp_path,
    )
    s = parse_ci_gate_specs(path)[0]
    assert s.steps == [0, 1]
    assert s.constraint == {"rel_up": 0.5, "abs_floor_up": 1e-6, "rel_down": 0.8, "abs_floor_down": 0.0}
    assert s.enforce is True
    assert s.allowlist_reason == "known noisy"


def test_declaration_keys_are_canonical_json(tmp_path):
    # Dict keys are sorted and every literal is whitespace-free regardless of
    # how the author wrote it; this pins the exact stored-identity format.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/ppo_kl",
                         steps=[0,  1],
                         constraint={"abs_floor_up": 0.02, "abs_floor_down": 0.02})
        register_ci_gate(metric_key="train/x", steps="last",
                         constraint={"rel_up": 0.2, "rel_down": 0.2})
        """,
        tmp_path,
    )
    fanned, reduced = parse_ci_gate_specs(path)
    assert fanned.steps_key == "[0,1]"
    assert fanned.constraint_key == '{"abs_floor_down":0.02,"abs_floor_up":0.02}'
    assert reduced.steps_key == '"last"'


def test_declaration_keys_use_raw_literal_not_normalized(tmp_path):
    # The normalized constraint fills the omitted params from code; the key
    # must come from the literal as written, so a code-side default change
    # can never silently rewrite keys and reset every series.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="last", constraint={"rel_up": 0.2, "rel_down": 0.2})
        """,
        tmp_path,
    )
    s = parse_ci_gate_specs(path)[0]
    assert s.constraint == {"rel_up": 0.2, "abs_floor_up": 0.0, "rel_down": 0.2, "abs_floor_down": 0.0}
    assert s.constraint_key == '{"rel_down":0.2,"rel_up":0.2}'
    assert "abs_floor" not in s.constraint_key


def test_abs_optional_rel_defaults_to_zero(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(
            metric_key="train/ppo_kl",
            steps="last",
            constraint={"abs_floor_up": 1e-6, "abs_floor_down": 1e-6},
        )
        """,
        tmp_path,
    )
    s = parse_ci_gate_specs(path)[0]
    assert s.constraint == {"rel_up": 0.0, "abs_floor_up": 1e-6, "rel_down": 0.0, "abs_floor_down": 1e-6}


def test_parse_multiple_specs(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/grad_norm",
                         steps="all", constraint={"rel_up": 0.2, "rel_down": 0.2})
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last",
                         constraint={"rel_up": 0.5, "rel_down": 0.2})
        """,
        tmp_path,
    )
    specs = parse_ci_gate_specs(path)
    assert [s.metric_key for s in specs] == ["train/grad_norm", "rollout/raw_reward"]


def test_unknown_kwarg_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="last", constraint={"rel_up": 0.2, "rel_down": 0.2}, bogus=3)
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="unknown argument 'bogus'"):
        parse_ci_gate_specs(path)


def test_non_literal_arg_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        X = "last"
        register_ci_gate(metric_key="train/x", steps=X,
                         constraint={"rel_up": 0.2, "rel_down": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="must be a literal"):
        parse_ci_gate_specs(path)


def test_non_literal_inside_dict_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        X = 0.2
        register_ci_gate(metric_key="train/x",
                         steps="last", constraint={"rel_up": X, "rel_down": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="must be a literal"):
        parse_ci_gate_specs(path)


@pytest.mark.parametrize("missing", ["metric_key", "steps", "constraint"])
def test_missing_required_field_rejected(tmp_path, missing):
    fields = {
        "metric_key": 'metric_key="train/x"',
        "steps": 'steps="last"',
        "constraint": 'constraint={"rel_up": 0.2, "rel_down": 0.2}',
    }
    del fields[missing]
    call = f"register_ci_gate({', '.join(fields.values())})"
    path = _make_fixture(
        f"""
        from tests.ci.metric_history import register_ci_gate
        {call}
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match=f"{missing} is required"):
        parse_ci_gate_specs(path)


def test_one_liner_fills_from_gate_defaults(tmp_path):
    # A bare metric_key declaration on a standard metric is complete: steps and
    # constraint come from the table, and the declaration keys derive from the
    # TABLE literals (canonical JSON), not from any normalized form.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/ppo_kl")
        """,
        tmp_path,
    )
    s = parse_ci_gate_specs(path)[0]
    assert s.steps == "last"
    assert s.constraint == {"rel_up": 0.5, "abs_floor_up": 0.02, "rel_down": 0.8, "abs_floor_down": 0.02}
    assert s.steps_key == '"last"'
    assert s.constraint_key == '{"abs_floor_down":0.02,"abs_floor_up":0.02,"rel_down":0.8,"rel_up":0.5}'


def test_partial_default_written_literal_wins(tmp_path):
    # Each omitted field fills independently: an explicit constraint keeps its
    # own literal (and key) while steps still comes from the table.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="rollout/raw_reward",
                         constraint={"rel_up": 0.1, "rel_down": 0.1})
        """,
        tmp_path,
    )
    s = parse_ci_gate_specs(path)[0]
    assert s.steps == "last"  # from GATE_DEFAULTS
    assert s.constraint_key == '{"rel_down":0.1,"rel_up":0.1}'  # written literal, not the table's


def test_one_liner_without_table_entry_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/not_a_standard_metric")
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="has no GATE_DEFAULTS entry"):
        parse_ci_gate_specs(path)


def test_gate_defaults_within_capture_whitelist_and_valid(tmp_path):
    # Two table invariants: every defaulted key must be captured (a default for
    # an uncaptured metric guarantees an ERROR verdict), and every table entry
    # must survive the parser's own schema validation.
    from miles.utils.tracking_utils.ci_history import TARGET_METRIC_KEYS

    assert set(GATE_DEFAULTS) <= set(TARGET_METRIC_KEYS)

    # Concatenate instead of interpolating into an indented f-string: the
    # joined lines would defeat the fixture's dedent.
    body = "from tests.ci.metric_history import register_ci_gate\n" + "\n".join(
        f'register_ci_gate(metric_key="{key}")' for key in GATE_DEFAULTS
    )
    path = _make_fixture(body, tmp_path)
    specs = parse_ci_gate_specs(path)
    assert len(specs) == len(GATE_DEFAULTS)


def test_one_liner_runtime_is_noop():
    # The Python signature must accept the one-liner form at import time; the
    # parser, not the signature, decides validity.
    from tests.ci.metric_history import register_ci_gate

    assert register_ci_gate(metric_key="rollout/raw_reward") is None


def test_positional_arg_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate("train/x", 1.0)
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="only keyword arguments"):
        parse_ci_gate_specs(path)


def test_unknown_steps_keyword_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="mean_last_9000", constraint={"rel_up": 0.2, "rel_down": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match='steps must be "last", "all"'):
        parse_ci_gate_specs(path)


def test_constraint_name_key_is_gone(tmp_path):
    # The old name-keyed constraint registry was merged into one band family;
    # a declaration still passing "name" must fail loud, not silently drop.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="last", constraint={"name": "band", "rel_up": 0.2, "rel_down": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="unknown key 'name' for constraint"):
        parse_ci_gate_specs(path)


@pytest.mark.parametrize(
    "literal,match",
    [
        ('{"rel_down": 0.2}', "upper band param"),
        ('{"rel_up": 0.2}', "lower band param"),
    ],
    ids=["missing-up", "missing-down"],
)
def test_constraint_missing_side_rejected(tmp_path, literal, match):
    # An all-default side has band 0 and fails on any deviation; the parser
    # demands at least one written band param per side.
    path = _make_fixture(
        f"""
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="last", constraint={literal})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match=match):
        parse_ci_gate_specs(path)


def test_non_string_non_list_steps_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps=0, constraint={"rel_up": 0.2, "rel_down": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match='steps must be "last", "all"'):
        parse_ci_gate_specs(path)


@pytest.mark.parametrize(
    "steps_literal",
    ["[]", "[1.5]", "[True]", "[-1]", "[0, 0]"],
    ids=["empty", "float", "bool", "negative", "duplicate"],
)
def test_bad_steps_list_rejected(tmp_path, steps_literal):
    path = _make_fixture(
        f"""
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps={steps_literal},
                         constraint={{"rel_up": 0.2, "rel_down": 0.2}})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="steps: "):
        parse_ci_gate_specs(path)


def test_direction_key_is_gone(tmp_path):
    # The one-sided direction abstraction was removed: every constraint is
    # two-sided, a lenient side gets a wide band. A stale declaration must
    # fail loud, not silently drop its direction.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="last",
                         constraint={"rel_up": 0.2, "rel_down": 0.8, "direction": "higher_is_worse"})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="unknown key 'direction'"):
        parse_ci_gate_specs(path)


def test_negative_rel_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="last", constraint={"rel_up": -0.2, "rel_down": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="param 'rel_up'"):
        parse_ci_gate_specs(path)


def test_duplicate_dict_key_rejected(tmp_path):
    # A plain dict would silently keep the last value; the parser must reject.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="last",
                         constraint={"rel_up": 0.2, "rel_up": 0.3, "rel_down": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="duplicate key 'rel_up'"):
        parse_ci_gate_specs(path)


def test_sub_label_argument_is_gone(tmp_path):
    # The old author-label argument was removed with the encoded-coordinate
    # design; a declaration still passing it must fail loud, not silently drop.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="last", constraint={"rel_up": 0.2, "rel_down": 0.2},
                         sub_label="shard-0")
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="unknown argument 'sub_label'"):
        parse_ci_gate_specs(path)


def test_non_bool_enforce_rejected(tmp_path):
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="last", constraint={"rel_up": 0.2, "rel_down": 0.2}, enforce=1)
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="enforce must be a boolean"):
        parse_ci_gate_specs(path)


def test_register_ci_gate_does_not_disturb_suite_parsing(tmp_path):
    # The suite RegistryVisitor must still find exactly the register_cuda_ci
    # call and ignore the register_ci_gate calls beside it.
    path = _make_fixture(
        """
        from tests.ci.ci_register import register_cuda_ci
        from tests.ci.metric_history import register_ci_gate
        register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100", labels=["megatron"])
        register_ci_gate(metric_key="train/grad_norm",
                         steps="all", constraint={"rel_up": 0.2, "rel_down": 0.2})
        """,
        tmp_path,
    )
    registries = ut_parse_one_file(path)
    assert len(registries) == 1
    assert isinstance(registries[0], CIRegistry)
    assert registries[0].backend == HWBackend.CUDA
    assert registries[0].suite == "stage-c-8-gpu-h100"


def test_register_ci_gate_runtime_is_noop():
    from tests.ci.metric_history import register_ci_gate

    assert (
        register_ci_gate(
            metric_key="train/grad_norm",
            steps="all",
            constraint={"rel_up": 0.2, "rel_down": 0.2},
        )
        is None
    )
