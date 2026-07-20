# doc-dev: docs/ci/03-metric-history-gate.md
"""Declare and parse metric-history regression gates.

* `register_ci_gate(...)` is the marker a test file uses to declare a gate.
* Like `register_cuda_ci` it is a runtime no-op, parsed out of the file's AST
  rather than executed.
* A gate composes `steps` (which value(s) to pull from a metric's series:
  `"last"` | `"all"` | a non-empty list of step indices) and a `constraint`
  (the pass/fail rule).
* `constraint` is a literal dict of the two-sided band params (`rel_up` /
  `abs_floor_up` / `rel_down` / `abs_floor_down`), validated against the
  schema in :mod:`constraints`.
* For the captured standard metrics, :data:`GATE_DEFAULTS` supplies `steps`
  and `constraint`: each field a declaration omits is filled from the table at
  parse time, through the same validation, so
  `register_ci_gate(metric_key="train/ppo_kl")` alone is a complete
  declaration. A written literal always wins; a metric with no table entry
  must write both fields.
* A spec also carries `steps_key` / `constraint_key` -- canonical JSON of the
  raw `steps` / `constraint` literals (see :func:`_canonical_key` for the
  exact serialization) -- which, plus the selection's `step`, form the
  identity a stored value's history is keyed under.
* `parse_ci_gate_specs` extracts every declaration as a :class:`CiGateSpec`.

Caveats:

* `register_ci_gate`'s Python signature does NOT validate calls -- the parser
  here does, at parse time.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass

from tests.ci.metric_history.constraints import CONSTRAINT_SCHEMA


def register_ci_gate(
    *,
    metric_key: str,
    steps: str | list[int] | None = None,
    constraint: dict | None = None,
    enforce: bool = False,
    allowlist_reason: str | None = None,
):
    """Declare one history-gate spec for the test file it sits in.

    Parsed via AST (like `register_cuda_ci`); a runtime no-op. Every argument
    is keyword-only and must be a literal. `metric_key` names the target
    metric. `steps` picks the
    comparison value(s): `"last"` (the series' last point), `"all"` (every
    step present, fanned out), or a non-empty list of step indices.
    `constraint` is a literal dict of the two-sided band params -- see
    :data:`constraints.CONSTRAINT_SCHEMA`; each side needs at least one of its
    `rel` / `abs_floor` written. An omitted `steps` / `constraint` is filled from
    :data:`GATE_DEFAULTS` when `metric_key` has an entry there; the Python
    defaults of None exist only so a one-liner call runs as a no-op -- the
    parser, not this signature, decides validity. `enforce` and
    `allowlist_reason` are policy metadata the gate carries without acting on
    (the verdict is informational this round).
    """
    return None


_REGISTER_NAME = "register_ci_gate"
_REQUIRED = object()

# Top-level register_ci_gate fields: name -> (required, default). `steps` and
# `constraint` are required unless GATE_DEFAULTS fills them first (the fill
# runs before this table is enforced).
_FIELDS: dict[str, tuple[bool, object]] = {
    "metric_key": (True, _REQUIRED),
    "steps": (True, _REQUIRED),
    "constraint": (True, _REQUIRED),
    "enforce": (False, False),
    "allowlist_reason": (False, None),
}

# Per-metric_key declaration defaults for the captured standard metrics: the
# `steps` / `constraint` literals filled in when a declaration omits them.
# These literals feed the SAME key derivation as written literals, so a
# defaulted coordinate keys on the table entry -- editing a value here re-keys
# every declaration that relied on it and cold-starts those baselines. Band
# values are shadow-calibration starting points, deliberately loose. Keys must
# stay within the capture whitelist (miles.utils.tracking_utils
# TARGET_METRIC_KEYS; test-enforced, not imported here).
GATE_DEFAULTS: dict[str, dict] = {
    "train/grad_norm": {
        "steps": "last",
        "constraint": {"rel_up": 0.5, "abs_floor_up": 0.1, "rel_down": 0.8, "abs_floor_down": 0.1},
    },
    "train/ppo_kl": {
        "steps": "last",
        "constraint": {"rel_up": 0.5, "abs_floor_up": 0.02, "rel_down": 0.8, "abs_floor_down": 0.02},
    },
    "train/train_rollout_logprob_abs_diff": {
        "steps": "last",
        "constraint": {"rel_up": 0.5, "abs_floor_up": 0.02, "rel_down": 0.8, "abs_floor_down": 0.02},
    },
    "train/train_rollout_kl": {
        "steps": "last",
        "constraint": {"rel_up": 0.5, "abs_floor_up": 0.02, "rel_down": 0.8, "abs_floor_down": 0.02},
    },
    "rollout/raw_reward": {
        "steps": "last",
        "constraint": {"rel_up": 0.5, "abs_floor_up": 0.05, "rel_down": 0.2, "abs_floor_down": 0.05},
    },
}


@dataclass(frozen=True)
class CiGateSpec:
    """One parsed `register_ci_gate` declaration.

    `steps` is the validated selection literal (`"last"` | `"all"` | a list of
    step indices); `constraint` is a normalized dict (name + validated params
    + filled defaults); both drive execution. `steps_key` / `constraint_key` are
    canonical JSON of the same literals as written and, with the extraction's
    `step`, form the stored value's identity. `filename` is the test file the
    spec governs; run identity comes from its CIRegistry.
    """

    filename: str
    metric_key: str
    steps: str | list[int]
    constraint: dict
    steps_key: str
    constraint_key: str
    enforce: bool = False
    allowlist_reason: str | None = None


class _ParseError(Exception):
    """Internal: a bare message the caller wraps with file + field context."""


def _literal(node: ast.AST) -> object:
    """A Python literal from an AST node: a constant, a negative number, or a
    list / dict of the same. Rejects any non-literal (name, call, expression).

    Negative numbers matter because `-1.0` is an `ast.UnaryOp`, not an
    `ast.Constant`; a plain constant check would wrongly reject them. Dict keys
    must be string literals and duplicates are rejected (a plain dict would
    silently keep the last).
    """
    if isinstance(node, ast.Constant):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        return -node.operand.value
    if isinstance(node, ast.List):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        out: dict = {}
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if key_node is None:
                raise _ParseError("dict unpacking (**) is not allowed")
            key = _literal(key_node)
            if not isinstance(key, str):
                raise _ParseError("dict keys must be string literals")
            if key in out:
                raise _ParseError(f"duplicate key {key!r}")
            out[key] = _literal(value_node)
        return out
    raise _ParseError(f"must be a literal (got {type(node).__name__})")


def _validate_param(validator: str, value: object) -> object:
    """Validate one schema param (also the flat `steps` list); return it
    (normalized)."""
    if validator == "float_nonneg":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _ParseError("must be a number")
        if not math.isfinite(value) or value < 0:
            raise _ParseError("must be a finite number >= 0")
        return float(value)
    if validator == "step_list":
        if not isinstance(value, list) or not value:
            raise _ParseError("must be a non-empty list of step indices")
        seen: set[int] = set()
        for s in value:
            if isinstance(s, bool) or not isinstance(s, int):
                raise _ParseError("step indices must be integers")
            if s < 0:
                raise _ParseError("step indices must be >= 0")
            if s in seen:
                raise _ParseError(f"duplicate step {s}")
            seen.add(s)
        return list(value)
    raise _ParseError(f"internal: unknown validator {validator!r}")


def _normalize_constraint(raw: object) -> dict:
    """Validate the constraint literal against `CONSTRAINT_SCHEMA` and return
    a normalized dict (validated params + filled defaults). Each side needs at
    least one written band param -- an all-default side has band 0 and would
    fail on any deviation."""
    if not isinstance(raw, dict):
        raise _ParseError("constraint must be a dict")
    for key in raw:
        if key not in CONSTRAINT_SCHEMA:
            raise _ParseError(f"unknown key {key!r} for constraint; valid: {sorted(CONSTRAINT_SCHEMA)}")
    if "rel_up" not in raw and "abs_floor_up" not in raw:
        raise _ParseError("constraint requires at least one upper band param ('rel_up' or 'abs_floor_up')")
    if "rel_down" not in raw and "abs_floor_down" not in raw:
        raise _ParseError("constraint requires at least one lower band param ('rel_down' or 'abs_floor_down')")
    normalized: dict = {}
    for param, (validator, required, default) in CONSTRAINT_SCHEMA.items():
        if param in raw:
            try:
                normalized[param] = _validate_param(validator, raw[param])
            except _ParseError as e:
                raise _ParseError(f"constraint param {param!r}: {e}") from None
        elif required:
            raise _ParseError(f"constraint requires {param!r}")
        else:
            normalized[param] = default
    return normalized


def _validate_steps(value: object) -> str | list[int]:
    """The declaration's `steps` literal: `"last"`, `"all"`, or a step list."""
    if value == "last" or value == "all":
        return value
    if isinstance(value, list):
        try:
            return _validate_param("step_list", value)
        except _ParseError as e:
            raise _ParseError(f"steps: {e}") from None
    raise _ParseError('steps must be "last", "all", or a non-empty list of step indices')


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise _ParseError(f"{field} must be a string")
    return value


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise _ParseError(f"{field} must be a boolean")
    return value


def _require_opt_str(value: object, field: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise _ParseError(f"{field} must be a string or None")
    return value


def _canonical_key(raw: object) -> str:
    """Canonical JSON of a declaration literal (the `steps` string/list or
    the `constraint` dict), as the stored identity key: no whitespace, dict
    keys sorted, list order kept as written, a bare string keyword
    serialized with its JSON quotes (`"last"` -> `'"last"'`).

    Deliberately built from the literal as written in the test file, NOT the
    normalized form: filled-in defaults live in code, so a code-side default
    change would silently rewrite normalized keys and reset every series. The
    raw literal changes only with an edit to the declaration itself -- the
    intended reset lever for that coordinate's history.
    """
    return json.dumps(raw, sort_keys=True, separators=(",", ":"))


def _parse_ci_gate_call(call: ast.Call, filename: str) -> CiGateSpec:
    prefix = f"{filename}: {_REGISTER_NAME}()"
    if call.args:
        raise ValueError(f"{prefix} takes only keyword arguments (got {len(call.args)} positional)")

    raw: dict[str, object] = {}
    for kw in call.keywords:
        if kw.arg is None:
            raise ValueError(f"{prefix}: **kwargs are not supported")
        if kw.arg not in _FIELDS:
            raise ValueError(f"{prefix}: unknown argument {kw.arg!r}; valid: {sorted(_FIELDS)}")
        if kw.arg in raw:
            raise ValueError(f"{prefix}: duplicated argument {kw.arg!r}")
        try:
            raw[kw.arg] = _literal(kw.value)
        except _ParseError as e:
            raise ValueError(f"{prefix}: {kw.arg} {e}") from None

    # Defaults fill runs before the required check: an omitted steps/constraint
    # is an error only when the metric has no GATE_DEFAULTS entry. The table
    # literal enters the same validation and key derivation a written literal
    # would (see GATE_DEFAULTS).
    metric_key = raw.get("metric_key")
    defaults = GATE_DEFAULTS.get(metric_key) if isinstance(metric_key, str) else None
    if defaults is not None:
        for field in ("steps", "constraint"):
            if field not in raw:
                raw[field] = defaults[field]

    for field, (required, default) in _FIELDS.items():
        if field not in raw:
            if required:
                if field in ("steps", "constraint"):
                    raise ValueError(
                        f"{prefix}: {field} is required (metric_key {metric_key!r} has no GATE_DEFAULTS entry)"
                    )
                raise ValueError(f"{prefix}: {field} is required")
            raw[field] = default

    try:
        return CiGateSpec(
            filename=filename,
            metric_key=_require_str(raw["metric_key"], "metric_key"),
            steps=_validate_steps(raw["steps"]),
            constraint=_normalize_constraint(raw["constraint"]),
            steps_key=_canonical_key(raw["steps"]),
            constraint_key=_canonical_key(raw["constraint"]),
            enforce=_require_bool(raw["enforce"], "enforce"),
            allowlist_reason=_require_opt_str(raw["allowlist_reason"], "allowlist_reason"),
        )
    except _ParseError as e:
        raise ValueError(f"{prefix}: {e}") from None


def parse_ci_gate_specs(filename: str) -> list[CiGateSpec]:
    """Return every `register_ci_gate` spec declared at top level in `filename`.

    Parsed the same way as `register_cuda_ci`: top-level `Expr(Call)` whose
    callee is the bare name `register_ci_gate`. Non-literal / invalid args raise
    ValueError naming the file and field.

    Two specs may still map to the same baseline coordinate (identical
    steps + constraint literals, differing only in policy metadata); the
    harness writer (`tests.ci.ci_utils.run_gate_hook`) collapses such
    duplicates to one `metric_values` row per coordinate.
    """
    with open(filename) as f:
        tree = ast.parse(f.read(), filename=filename)
    specs: list[CiGateSpec] = []
    for stmt in tree.body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        if not isinstance(call.func, ast.Name) or call.func.id != _REGISTER_NAME:
            continue
        specs.append(_parse_ci_gate_call(call, filename))
    return specs
