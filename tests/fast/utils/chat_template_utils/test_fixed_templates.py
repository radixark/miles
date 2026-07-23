"""Unit tests for ``resolve_fixed_chat_template`` smallest-superset lookup.

Rows live as ``SUPPORTED_TEMPLATES`` class attributes on each
``TITOTokenizer`` subclass; the resolver picks the smallest superset of the
caller's ``allowed_append_roles`` and returns ``(template_path, extra_kwargs)``.
"""

import os

import pytest

from miles.utils.chat_template_utils import TEMPLATE_DIR, TITOTokenizerType, resolve_fixed_chat_template
from miles.utils.chat_template_utils.tito_tokenizer import (
    FixedTemplateRow,
    Qwen3TITOTokenizer,
    Qwen35TITOTokenizer,
    TITOTokenizer,
)

_QWEN3_FIXED = str(TEMPLATE_DIR / "qwen3_fixed.jinja")
_QWEN35_FIXED = str(TEMPLATE_DIR / "qwen3.5_fixed.jinja")
_THINKING_FIXED = str(TEMPLATE_DIR / "qwen3_thinking_2507_and_next_fixed.jinja")


@pytest.mark.parametrize(
    "tito_model, expected_path",
    [
        (TITOTokenizerType.QWEN3, _QWEN3_FIXED),
        (TITOTokenizerType.QWEN35, _QWEN35_FIXED),
        (TITOTokenizerType.QWENNEXT, _THINKING_FIXED),
    ],
)
def test_tool_only_resolution_matches_registered_template(tito_model, expected_path):
    path, kwargs = resolve_fixed_chat_template(tito_model, ["tool"])
    assert path == expected_path
    # Preserve-think kwargs are family constants on every surface (carried
    # assistant deltas re-render mid-path assistants; F5 spike verified the
    # pure tool-loop rendering is byte-identical with and without the kwarg).
    assert kwargs == {"clear_thinking": False}
    assert os.path.isfile(path)


def test_deepseek_v4_tool_only_pins_drop_thinking_false():
    # drop_thinking=False is a family constant on every surface. The encoder
    # forces it only when `tools` are present; a tool-loop driven WITHOUT the
    # tools field (or a carried assistant delta) would otherwise strip prior
    # thinking once merge_tool_messages advances last_user_index.
    path, kwargs = resolve_fixed_chat_template(TITOTokenizerType.DEEPSEEKV4, ["tool"])
    assert path is None
    assert kwargs == {"drop_thinking": False}


def test_deepseek_v4_tool_user_pins_drop_thinking_false():
    # OpenEnv-style text-protocol scaffolds append env output as plain `user`
    # turns and pass no `tools`. Without drop_thinking=False the encoder strips
    # prior assistants' thinking once a new user turn advances last_user_index —
    # a non-append-only mutation that corrupts the pretokenized prefix and NaNs
    # the first backward. This row must pin drop_thinking=False.
    path, kwargs = resolve_fixed_chat_template(TITOTokenizerType.DEEPSEEKV4, ["tool", "user"])
    assert path is None
    assert kwargs == {"drop_thinking": False}


@pytest.mark.parametrize(
    "tito_model",
    [TITOTokenizerType.QWEN3, TITOTokenizerType.QWEN35, TITOTokenizerType.QWENNEXT],
)
def test_no_superset_raises(tito_model):
    # ``{"tool", "system"}`` has no superset registered for the Qwen fixed
    # families (which register ``{"tool"}`` and ``{"tool", "user"}``).  The
    # resolver must surface this as a hard error so the caller cannot silently
    # fall through to a non-append-only HF render.
    with pytest.raises(ValueError) as excinfo:
        resolve_fixed_chat_template(tito_model, ["tool", "system"])
    msg = str(excinfo.value)
    assert f"tito_model={tito_model.value}" in msg
    assert "allowed_append_roles=['system', 'tool']" in msg
    assert "SUPPORTED_TEMPLATES" in msg


# ---------------------------------------------------------------------------
# Smallest-superset semantics — covered with monkeypatched class attributes
# so the behavior is pinned independently of which rows happen to be
# registered today.
# ---------------------------------------------------------------------------


def test_subset_caller_falls_through_to_multi_role_row(monkeypatch):
    # Only a multi-role row is registered; a strict-subset caller resolves to it.
    monkeypatch.setattr(
        Qwen3TITOTokenizer,
        "SUPPORTED_TEMPLATES",
        (
            FixedTemplateRow(
                allowed_roles=frozenset({"tool", "user"}),
                template="fake.jinja",
            ),
        ),
    )
    path, kwargs = resolve_fixed_chat_template(TITOTokenizerType.QWEN3, ["tool"])
    assert path == str(TEMPLATE_DIR / "fake.jinja")
    assert kwargs == {}


def test_exact_match_wins_over_strict_superset(monkeypatch):
    # Both an exact ``{tool}`` row and a multi-role ``{tool, user}`` row exist.
    # Smallest-cardinality wins, so caller ``["tool"]`` gets the exact row.
    monkeypatch.setattr(
        Qwen3TITOTokenizer,
        "SUPPORTED_TEMPLATES",
        (
            FixedTemplateRow(
                allowed_roles=frozenset({"tool"}),
                template="exact.jinja",
            ),
            FixedTemplateRow(
                allowed_roles=frozenset({"tool", "user"}),
                template="multi.jinja",
            ),
        ),
    )
    path, kwargs = resolve_fixed_chat_template(TITOTokenizerType.QWEN3, ["tool"])
    assert path == str(TEMPLATE_DIR / "exact.jinja")
    assert kwargs == {}


def test_ambiguous_minimal_supersets_raise(monkeypatch):
    # Two rows of equal cardinality both ⊇ caller's roles → ambiguous.
    monkeypatch.setattr(
        Qwen3TITOTokenizer,
        "SUPPORTED_TEMPLATES",
        (
            FixedTemplateRow(
                allowed_roles=frozenset({"tool", "user"}),
                template="a.jinja",
            ),
            FixedTemplateRow(
                allowed_roles=frozenset({"tool", "system"}),
                template="b.jinja",
            ),
        ),
    )
    with pytest.raises(ValueError, match="Ambiguous fixed-template registration"):
        resolve_fixed_chat_template(TITOTokenizerType.QWEN3, ["tool"])


def test_other_tito_model_rows_ignored(monkeypatch):
    # Rows for a different tito_model never participate in the match — when
    # the requested family has nothing registered, the resolver must throw.
    monkeypatch.setattr(Qwen3TITOTokenizer, "SUPPORTED_TEMPLATES", ())
    monkeypatch.setattr(
        Qwen35TITOTokenizer,
        "SUPPORTED_TEMPLATES",
        (
            FixedTemplateRow(
                allowed_roles=frozenset({"tool"}),
                template="qwen35.jinja",
            ),
        ),
    )
    with pytest.raises(ValueError, match="No SUPPORTED_TEMPLATES row registered"):
        resolve_fixed_chat_template(TITOTokenizerType.QWEN3, ["tool"])


def test_kwargs_only_row_returns_none_path_and_kwargs(monkeypatch):
    # A row with template=None registers a kwargs-only fix (HF native + kwargs
    # override).  Resolver returns (None, kwargs) so caller can keep the HF
    # template and merge the kwargs.
    monkeypatch.setattr(
        Qwen3TITOTokenizer,
        "SUPPORTED_TEMPLATES",
        (
            FixedTemplateRow(
                allowed_roles=frozenset({"tool", "user"}),
                template=None,
                extra_kwargs={"clear_thinking": False},
            ),
        ),
    )
    path, kwargs = resolve_fixed_chat_template(TITOTokenizerType.QWEN3, ["tool", "user"])
    assert path is None
    assert kwargs == {"clear_thinking": False}


def test_invalid_role_raises():
    with pytest.raises(ValueError, match="Unknown roles"):
        resolve_fixed_chat_template(TITOTokenizerType.QWEN3, ["tool", "users"])


def test_preserve_think_kwargs_are_family_constants():
    """Trajectory-tree precondition (MULTI_LINEAGE_DESIGN, F5 spike): carried
    assistant deltas are re-rendered canonically, so think-preservation must
    not depend on which append surface was requested — every registered row
    of a family must resolve to the same (template, extra_kwargs)."""

    def own_registries(cls):
        for sub in cls.__subclasses__():
            rows = sub.__dict__.get("SUPPORTED_TEMPLATES")
            if rows:
                yield sub, rows
            yield from own_registries(sub)

    checked = 0
    for family, rows in own_registries(TITOTokenizer):
        configs = {(row.template, tuple(sorted(row.extra_kwargs.items()))) for row in rows}
        assert len(configs) == 1, (
            f"{family.__name__}: (template, extra_kwargs) must be a family constant "
            f"across append surfaces, got {configs}"
        )
        checked += 1
    assert checked >= 10
