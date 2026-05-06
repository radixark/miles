"""Unit tests for ``verify_append_only_via_tito_instance`` /
``run_all_checks_via_tito``: PASS on registered TITO families, FAIL on the
unfixed Qwen3 chat template, FAIL on a test-local ``_BuggyQwen3TITOTokenizer``
that omits the ``\\n`` insertion at the ``<|im_end|>`` boundary.
"""

from copy import deepcopy

import pytest
from tests.ci.ci_register import register_cpu_ci
from transformers import AutoTokenizer

register_cpu_ci(est_time=120, suite="stage-a-fast")


from miles.utils.chat_template_utils import TITOTokenizerType, resolve_fixed_chat_template
from miles.utils.chat_template_utils.tito_tokenizer import Qwen3TITOTokenizer
from miles.utils.test_utils.chat_template_verify import run_all_checks_via_tito, verify_append_only_via_tito_instance
from miles.utils.test_utils.mock_trajectories import SingleToolTrajectory

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _setup_tokenizer_with_registered_template(
    model_id: str,
    family: TITOTokenizerType,
    roles: list[str],
):
    """Mirror what production wiring does at startup.

    Loads tokenizer, looks up the registered ``SUPPORTED_TEMPLATES`` row for
    ``(family, roles)``, and applies the resolved fixed template (if any) onto
    ``tokenizer.chat_template``. Returns ``(tokenizer, extra_kwargs)``.

    A fresh tokenizer instance per call avoids state-mutation hazards from
    overwriting ``chat_template``.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    fixed_path, extra_kwargs = resolve_fixed_chat_template(family, roles)
    if fixed_path is not None:
        with open(fixed_path) as f:
            tokenizer.chat_template = f.read()
    return tokenizer, dict(extra_kwargs)


# ---------------------------------------------------------------------------
# (1) PASS on registered families × role surfaces
# ---------------------------------------------------------------------------


_PASS_PARAMS = [
    pytest.param(TITOTokenizerType.QWEN3, "Qwen/Qwen3-0.6B", frozenset({"tool"}), id="qwen3-tool"),
    pytest.param(TITOTokenizerType.QWEN3, "Qwen/Qwen3-0.6B", frozenset({"tool", "user"}), id="qwen3-tool_user"),
    pytest.param(TITOTokenizerType.QWEN35, "Qwen/Qwen3.5-0.8B", frozenset({"tool"}), id="qwen35-tool"),
    pytest.param(TITOTokenizerType.QWEN35, "Qwen/Qwen3.5-0.8B", frozenset({"tool", "user"}), id="qwen35-tool_user"),
    pytest.param(TITOTokenizerType.QWENNEXT, "Qwen/Qwen3-4B-Thinking-2507", frozenset({"tool"}), id="qwennext-tool"),
    pytest.param(
        TITOTokenizerType.QWENNEXT,
        "Qwen/Qwen3-4B-Thinking-2507",
        frozenset({"tool", "user"}),
        id="qwennext-tool_user",
    ),
    pytest.param(TITOTokenizerType.GLM47, "zai-org/GLM-4.7-Flash", frozenset({"tool"}), id="glm47-tool"),
    pytest.param(TITOTokenizerType.GLM47, "zai-org/GLM-4.7-Flash", frozenset({"tool", "user"}), id="glm47-tool_user"),
    pytest.param(
        TITOTokenizerType.GLM47,
        "zai-org/GLM-4.7-Flash",
        frozenset({"tool", "user", "system"}),
        id="glm47-tool_user_system",
    ),
]


@pytest.mark.parametrize("family,model_id,roles", _PASS_PARAMS)
def test_via_tito_pass_on_registered_families(family, model_id, roles):
    """All 4 registered TITO families round-trip cleanly via decode-roundtrip."""
    tokenizer, extra_kwargs = _setup_tokenizer_with_registered_template(model_id, family, sorted(roles))
    results = run_all_checks_via_tito(
        tokenizer,
        family,
        allowed_append_roles=set(roles),
        thinking="both",
        extra_template_kwargs=extra_kwargs,
    )
    failures = [r for r in results if not r.passed]
    assert not failures, (
        f"Expected all PASS for {family.value} × {sorted(roles)} via TITO primitive; "
        f"got {len(failures)} FAIL(s) out of {len(results)}:\n"
        + "\n".join(f"  [{r.case_name}] {r.error}" for r in failures[:5])
    )


# ---------------------------------------------------------------------------
# (2) FAIL on the original unfixed Qwen3 chat template
# ---------------------------------------------------------------------------


def test_via_tito_fail_on_original_qwen3_template():
    """The original Qwen3 chat template uses ``loop.last`` and breaks append-only.

    Bypass ``resolve_fixed_chat_template`` entirely — keep the HF default
    ``tokenizer.chat_template`` and assert the primitive surfaces a FAIL.
    """
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    # Do NOT overwrite tokenizer.chat_template — keep the broken HF default.

    # Cast wide: thinking=both + multi-user-turn surface so trajectories that
    # actually advance ``last_query_index`` between prefix and full are exercised.
    # Those are the ones where ``loop.index0 > ns.last_query_index`` truncation
    # in the original Qwen3 template renders the same assistant turn differently
    # depending on the boundary position.
    results = run_all_checks_via_tito(
        tokenizer,
        TITOTokenizerType.QWEN3,
        allowed_append_roles={"tool", "user"},
        thinking="both",
    )
    failures = [r for r in results if not r.passed]
    assert failures, "Expected ≥1 FAIL on the original (unfixed) Qwen3 chat template; got all PASS."


# ---------------------------------------------------------------------------
# (3) FAIL on a test-local buggy subclass
# ---------------------------------------------------------------------------


class _BuggyQwen3TITOTokenizer(Qwen3TITOTokenizer):
    """Test-only Qwen3 variant with the ``\\n`` boundary insertion deleted.

    Real ``Qwen3TITOTokenizer.merge_tokens`` appends ``self._newline_id`` after
    a trailing ``<|im_end|>`` because the model stops without emitting the
    newline the chat template would otherwise produce.  This variant skips
    that fixup; the decode-roundtrip primitive is expected to surface it as a
    single-character diff at the prefix-suffix junction.
    """

    def merge_tokens(self, old_messages, new_messages, pretokenized_token_ids, tools=None):
        incremental = self.tokenize_additional_non_assistant(old_messages, new_messages, tools)
        # Intentionally omit the `+\n` insertion — that's the bug we're catching.
        return list(pretokenized_token_ids) + incremental


def test_via_tito_fail_on_buggy_qwen3_subclass():
    """A buggy ``merge_tokens`` produces a junction-level diff that the verifier surfaces."""
    tokenizer, _ = _setup_tokenizer_with_registered_template("Qwen/Qwen3-0.6B", TITOTokenizerType.QWEN3, ["tool"])
    buggy = _BuggyQwen3TITOTokenizer(tokenizer, allowed_append_roles=["tool"])

    result = verify_append_only_via_tito_instance(
        buggy,
        tokenizer,
        deepcopy(SingleToolTrajectory.MESSAGES),
        pretokenized_num_message=3,
        tools=SingleToolTrajectory.TOOLS,
        case_name="buggy_qwen3-single_tool-N3",
    )
    assert not result.passed, "Expected FAIL on _BuggyQwen3TITOTokenizer (omits the `+\\n` boundary patch); got PASS."
    assert "Decode-roundtrip mismatch" in (
        result.error or ""
    ), f"Expected decode-roundtrip diff in error message; got: {result.error}"
