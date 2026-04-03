"""Tests for TITOTokenizer: merge_tokens boundary logic, incremental tokenization, and factory.

## Test structure

TestConfig
    Smoke-checks that each subclass stores the correct model-specific config
    (assistant_start_str, trailing_token_ids, max_trim_tokens) and propagates
    them to the comparator.  These are NOT behavioral tests — they guard
    against accidental config regressions when modifying __init__.

TestMergeTokensBoundary
    Unit tests for the core merge_tokens boundary logic, using *synthetic*
    prefix IDs ([100, 200, ...]) so the assertions are purely about prefix
    manipulation — not about template rendering.

    Why synthetic IDs?  merge_tokens is: ``prefix + [boundary fix] + incremental``.
    The incremental part comes from tokenize_additional_non_assistant (tested
    separately); boundary logic depends only on the last token of the prefix.
    Synthetic IDs isolate this and make failures trivially diagnosable.

    Covers three subclass behaviors:
    - Qwen3: inserts ``\\n`` when prefix ends with ``<|im_end|>`` (model stops
      at im_end without the trailing newline the template expects).
    - GLM47: strips trailing ``<|observation|>`` or ``<|user|>`` (model emits
      the stop token, but the template also emits it as the next turn's opener).
    - Default: plain concatenation (no boundary handling).

TestTokenizeAdditional
    Behavioral tests for tokenize_additional_non_assistant — the dummy-prefix
    diff that computes incremental token IDs for appended non-assistant messages.

    ``test_produces_nonempty_incremental`` is parametrized over:
      _TOOL_TRAJECTORIES (trajectory classes) × _TITO_MODELS (qwen3, glm47)
    Split points are auto-detected by _find_tito_splits from message structure,
    so adding a trajectory to _TOOL_TRAJECTORIES automatically extends coverage.

    Remaining tests verify append-only validation (reject prefix mutation,
    fewer messages, or forbidden roles like assistant).

TestFactory
    get_tito_tokenizer factory: string/enum dispatch, invalid input handling.
"""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from miles.utils.chat_template_utils.tito_tokenizer import (
    GLM47TITOTokenizer,
    Qwen3TITOTokenizer,
    TITOTokenizer,
    TITOTokenizerType,
    get_tito_tokenizer,
)
from miles.utils.test_utils.mock_trajectories import (
    IntermediateSystemTrajectory,
    LongChainTrajectory,
    MultiToolSingleTurnTrajectory,
    MultiTurnTrajectory,
    ParallelToolsTrajectory,
    RetrySystemTrajectory,
    SingleToolThinkingTrajectory,
    SingleToolTrajectory,
)

# ---------------------------------------------------------------------------
# Tokenizer cache
# ---------------------------------------------------------------------------

_TOK_CACHE: dict[str, AutoTokenizer] = {}


def _get_tokenizer(model_id: str) -> AutoTokenizer:
    if model_id not in _TOK_CACHE:
        _TOK_CACHE[model_id] = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return _TOK_CACHE[model_id]


# ---------------------------------------------------------------------------
# Fixtures — model-specific TITO tokenizers
#
# `tito` is parametrized over all supported models; use it for tests that
# should run against every model.  Named fixtures (qwen3_tito, etc.) are
# for tests specific to one subclass's boundary logic.
# ---------------------------------------------------------------------------

_TITO_MODELS: dict[str, tuple[str, type[TITOTokenizer]]] = {
    "qwen3": ("Qwen/Qwen3-4B", Qwen3TITOTokenizer),
    "glm47": ("zai-org/GLM-4.7-Flash", GLM47TITOTokenizer),
}


# TODO: "user" is intentionally excluded — the dummy-prefix diff in
# tokenize_additional_non_assistant assumes appended messages don't change how
# earlier turns render, which breaks for user messages on context-sensitive
# templates (e.g. Qwen3's last_query_index).  Only tool and system are safe.
_TOOL_AND_SYSTEM = ["tool", "system"]


@pytest.fixture(params=list(_TITO_MODELS.keys()))
def tito(request) -> TITOTokenizer:
    model_id, cls = _TITO_MODELS[request.param]
    return cls(_get_tokenizer(model_id), allowed_append_roles=_TOOL_AND_SYSTEM)


@pytest.fixture
def qwen3_tito() -> Qwen3TITOTokenizer:
    return Qwen3TITOTokenizer(_get_tokenizer("Qwen/Qwen3-4B"), allowed_append_roles=_TOOL_AND_SYSTEM)


@pytest.fixture
def glm47_tito() -> GLM47TITOTokenizer:
    return GLM47TITOTokenizer(_get_tokenizer("zai-org/GLM-4.7-Flash"), allowed_append_roles=_TOOL_AND_SYSTEM)


@pytest.fixture
def default_tito() -> TITOTokenizer:
    return TITOTokenizer(_get_tokenizer("Qwen/Qwen3-4B"), allowed_append_roles=_TOOL_AND_SYSTEM)


# ---------------------------------------------------------------------------
# Trajectory parametrization
#
# Instead of relying on PRETOKENIZE_POSITIONS (which serves the pretokenized
# *chat* tests), we derive TITO split points directly from message structure:
# every assistant(tool_calls) followed by a tool/system message is a valid
# split.  This way new trajectories get coverage automatically.
#
# To extend: add a trajectory class to _TOOL_TRAJECTORIES.
# To add a model: add an entry to _TITO_MODELS above.
# ---------------------------------------------------------------------------


def _find_tito_splits(traj_cls) -> list[int]:
    """Find TITO split positions from message structure.

    A valid split is at index ``i+1`` whenever ``messages[i]`` is an assistant
    message with tool_calls and ``messages[i+1]`` is a tool or system message.
    Returns a list of such positions (the index of the first appended message).
    """
    msgs = traj_cls.MESSAGES
    splits = []
    for i, msg in enumerate(msgs):
        if (
            msg.get("role") == "assistant"
            and msg.get("tool_calls")
            and i + 1 < len(msgs)
            and msgs[i + 1].get("role") in ("tool", "system")
        ):
            splits.append(i + 1)
    return splits


def _split_at(traj_cls, pos: int):
    """Split trajectory at *pos* into ``(old_msgs, new_msgs, tools)``.

    ``old_msgs = messages[:pos]`` — the pretokenized prefix (ends with assistant turn).
    ``new_msgs`` extends through all subsequent non-assistant messages (tool/system),
    stopping before the next assistant turn.
    """
    msgs = traj_cls.MESSAGES
    end = pos
    while end < len(msgs) and msgs[end].get("role") != "assistant":
        end += 1
    return msgs[:pos], msgs[:end], traj_cls.TOOLS


_TOOL_TRAJECTORIES = [
    SingleToolTrajectory,  # 1 tool call, 1 response
    MultiTurnTrajectory,  # 2 sequential tool turns
    MultiToolSingleTurnTrajectory,  # 2 parallel tool calls (weather + date)
    ParallelToolsTrajectory,  # 3 parallel tool calls
    LongChainTrajectory,  # 3 sequential turns (weather → date → weather)
    RetrySystemTrajectory,  # tool + system retry injection mid-conversation
    IntermediateSystemTrajectory,  # system messages interleaved with tool turns
    SingleToolThinkingTrajectory,  # tool call with reasoning_content
]

_TRAJ_CASES = [
    pytest.param(traj_cls, pos, id=f"{traj_cls.__name__}-N{pos}")
    for traj_cls in _TOOL_TRAJECTORIES
    for pos in _find_tito_splits(traj_cls)
]


# ---------------------------------------------------------------------------
# TestConfig — subclass configuration smoke-checks
# ---------------------------------------------------------------------------


class TestConfig:
    """Each subclass stores the correct model-specific configuration at init."""

    def test_qwen3(self, qwen3_tito: Qwen3TITOTokenizer):
        assert qwen3_tito._assistant_start_str == "<|im_start|>assistant"
        assert qwen3_tito._newline_id in qwen3_tito.trailing_token_ids

    def test_glm47(self, glm47_tito: GLM47TITOTokenizer):
        assert glm47_tito._assistant_start_str == "<|assistant|>"
        assert glm47_tito._observation_id in glm47_tito.trailing_token_ids
        assert glm47_tito._user_id in glm47_tito.trailing_token_ids
        assert glm47_tito.max_trim_tokens == 1

    def test_default(self, default_tito: TITOTokenizer):
        assert default_tito._assistant_start_str is None
        assert default_tito.trailing_token_ids == frozenset()

    def test_comparator_inherits_trailing_ids(self, qwen3_tito: Qwen3TITOTokenizer):
        """create_comparator propagates trailing_token_ids to the comparator's trim set."""
        comp = qwen3_tito.create_comparator()
        assert comp._trim_trailing_ids == set(qwen3_tito.trailing_token_ids)


# ---------------------------------------------------------------------------
# TestMergeTokensBoundary — prefix manipulation with synthetic IDs
#
# All tests use the same trajectory (SingleToolTrajectory split at pos=3)
# to compute incremental tokens, then verify prefix manipulation with
# synthetic IDs like [100, 200, <boundary_token>].
# ---------------------------------------------------------------------------

_BND_OLD, _BND_NEW, _BND_TOOLS = _split_at(SingleToolTrajectory, 3)


class TestMergeTokensBoundary:
    """merge_tokens correctly manipulates the prefix before concatenating incremental tokens."""

    # -- Qwen3: insert \n after <|im_end|> --

    def test_qwen3_inserts_newline_after_im_end(self, qwen3_tito: Qwen3TITOTokenizer):
        """Model stops at <|im_end|> without trailing \\n; merge_tokens inserts it."""
        incremental = qwen3_tito.tokenize_additional_non_assistant(_BND_OLD, _BND_NEW, _BND_TOOLS)
        im_end = qwen3_tito._im_end_id
        nl = qwen3_tito._newline_id

        result = qwen3_tito.merge_tokens(_BND_OLD, _BND_NEW, [100, 200, im_end], _BND_TOOLS)
        assert result == [100, 200, im_end, nl] + incremental

    def test_qwen3_no_newline_otherwise(self, qwen3_tito: Qwen3TITOTokenizer):
        """No insertion when prefix does not end with <|im_end|>."""
        incremental = qwen3_tito.tokenize_additional_non_assistant(_BND_OLD, _BND_NEW, _BND_TOOLS)
        result = qwen3_tito.merge_tokens(_BND_OLD, _BND_NEW, [100, 200, 300], _BND_TOOLS)
        assert result == [100, 200, 300] + incremental

    # -- GLM47: strip ambiguous boundary tokens --

    def test_glm47_strips_observation(self, glm47_tito: GLM47TITOTokenizer):
        """Model emits <|observation|> as stop token; merge_tokens strips the duplicate."""
        incremental = glm47_tito.tokenize_additional_non_assistant(_BND_OLD, _BND_NEW, _BND_TOOLS)
        result = glm47_tito.merge_tokens(_BND_OLD, _BND_NEW, [100, 200, glm47_tito._observation_id], _BND_TOOLS)
        assert result == [100, 200] + incremental

    def test_glm47_strips_user(self, glm47_tito: GLM47TITOTokenizer):
        """<|user|> is also an ambiguous boundary — stripped the same way."""
        incremental = glm47_tito.tokenize_additional_non_assistant(_BND_OLD, _BND_NEW, _BND_TOOLS)
        result = glm47_tito.merge_tokens(_BND_OLD, _BND_NEW, [100, 200, glm47_tito._user_id], _BND_TOOLS)
        assert result == [100, 200] + incremental

    def test_glm47_no_strip_otherwise(self, glm47_tito: GLM47TITOTokenizer):
        """Non-boundary trailing token is preserved."""
        incremental = glm47_tito.tokenize_additional_non_assistant(_BND_OLD, _BND_NEW, _BND_TOOLS)
        result = glm47_tito.merge_tokens(_BND_OLD, _BND_NEW, [100, 200, 300], _BND_TOOLS)
        assert result == [100, 200, 300] + incremental

    # -- Default: no boundary handling --

    def test_default_concatenates(self, default_tito: TITOTokenizer):
        """Base class does plain concatenation without any prefix modification."""
        incremental = default_tito.tokenize_additional_non_assistant(_BND_OLD, _BND_NEW, _BND_TOOLS)
        result = default_tito.merge_tokens(_BND_OLD, _BND_NEW, [100, 200, 300], _BND_TOOLS)
        assert result == [100, 200, 300] + incremental

    # -- Edge case --

    def test_empty_prefix(self, qwen3_tito: Qwen3TITOTokenizer):
        """Empty prefix → no boundary handling, result is just incremental."""
        incremental = qwen3_tito.tokenize_additional_non_assistant(_BND_OLD, _BND_NEW, _BND_TOOLS)
        result = qwen3_tito.merge_tokens(_BND_OLD, _BND_NEW, [], _BND_TOOLS)
        assert result == incremental


# ---------------------------------------------------------------------------
# TestTokenizeAdditional — incremental tokenization via dummy-prefix diff
#
# test_produces_nonempty_incremental is the scalable core: parametrized over
# _TRAJ_CASES (trajectories × split points) × tito fixture (models).
# 8 trajectories × ~14 splits × 2 models = 28 test cases currently.
#
# Validation tests use a single trajectory since the validation logic
# (assert_messages_append_only_with_allowed_role) is model/trajectory-independent.
# ---------------------------------------------------------------------------


class TestTokenizeAdditional:
    """tokenize_additional_non_assistant produces valid incremental tokens."""

    @pytest.mark.parametrize("traj_cls, pos", _TRAJ_CASES)
    def test_produces_nonempty_incremental(self, tito: TITOTokenizer, traj_cls, pos):
        """Every valid TITO split yields non-empty incremental tokens.

        This is the primary scalability test — it runs every trajectory's
        TITO splits against every model tokenizer.
        """
        old_msgs, new_msgs, tools = _split_at(traj_cls, pos)
        incremental = tito.tokenize_additional_non_assistant(old_msgs, new_msgs, tools)
        assert len(incremental) > 0

    # -- Append-only validation (assert_messages_append_only_with_allowed_role is called internally) --

    def test_rejects_prefix_mutation(self, qwen3_tito: Qwen3TITOTokenizer):
        """Modifying an existing message in new_messages raises ValueError."""
        old_msgs, new_msgs, _ = _split_at(SingleToolTrajectory, 3)
        mutated_old = [{"role": "user", "content": "CHANGED"}] + list(old_msgs[1:])
        mutated_new = mutated_old + list(new_msgs[len(old_msgs) :])
        with pytest.raises(ValueError, match="mismatch"):
            qwen3_tito.tokenize_additional_non_assistant(old_msgs, mutated_new)

    def test_rejects_fewer_messages(self, qwen3_tito: Qwen3TITOTokenizer):
        """new_messages shorter than old_messages raises ValueError."""
        old_msgs = SingleToolTrajectory.MESSAGES[:3]
        with pytest.raises(ValueError, match="fewer"):
            qwen3_tito.tokenize_additional_non_assistant(old_msgs, old_msgs[:1])

    def test_rejects_assistant_append(self, qwen3_tito: Qwen3TITOTokenizer):
        """Appending an assistant message (not tool/system) raises ValueError."""
        old_msgs = SingleToolTrajectory.MESSAGES[:3]
        bad_new = list(old_msgs) + [{"role": "assistant", "content": "hi"}]
        with pytest.raises(ValueError, match="role"):
            qwen3_tito.tokenize_additional_non_assistant(old_msgs, bad_new)


# ---------------------------------------------------------------------------
# TestFactory — get_tito_tokenizer dispatch
# ---------------------------------------------------------------------------


class TestFactory:
    """get_tito_tokenizer creates the correct subclass from string or enum type."""

    @pytest.mark.parametrize(
        "type_str, model_id, cls",
        [
            ("qwen3", "Qwen/Qwen3-4B", Qwen3TITOTokenizer),
            ("glm47", "zai-org/GLM-4.7-Flash", GLM47TITOTokenizer),
            ("default", "Qwen/Qwen3-4B", TITOTokenizer),
        ],
    )
    def test_creates_correct_type(self, type_str, model_id, cls):
        tito = get_tito_tokenizer(_get_tokenizer(model_id), tokenizer_type=type_str)
        assert isinstance(tito, cls)

    def test_enum_input(self):
        """Enum values work the same as string values."""
        tito = get_tito_tokenizer(_get_tokenizer("Qwen/Qwen3-4B"), tokenizer_type=TITOTokenizerType.QWEN3)
        assert isinstance(tito, Qwen3TITOTokenizer)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            get_tito_tokenizer(_get_tokenizer("Qwen/Qwen3-4B"), tokenizer_type="nonexistent")

    def test_none_tokenizer_raises(self):
        with pytest.raises(ValueError, match="must not be None"):
            get_tito_tokenizer(None)
