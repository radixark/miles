"""Tests for TITOTokenizer: merge_tokens boundary logic, incremental tokenization, and factory.

## Test structure

TestConfig
    Smoke-checks that each subclass stores the correct model-specific config
    (assistant_start_str, trailing_token_ids, max_trim_tokens) and propagates
    them to the comparator.  These are NOT behavioral tests — they guard
    against accidental config regressions when modifying __init__.

TestCompletionPostprocess
    Verifies that the default completion hook is an identity operation.

TestMergeTokensBoundary
    Unit tests for the core merge_tokens boundary logic, using *synthetic*
    prefix IDs ([100, 200, ...]) so the assertions are purely about prefix
    manipulation — not about template rendering.

    Why synthetic IDs?  merge_tokens is: ``prefix + [boundary fix] + incremental``.
    The incremental part comes from tokenize_additional_messages (tested
    separately); boundary logic depends only on the last token of the prefix.
    Synthetic IDs isolate this and make failures trivially diagnosable.

    Covers three subclass behaviors:
    - Qwen3: inserts ``\\n`` when prefix ends with ``<|im_end|>`` (model stops
      at im_end without the trailing newline the template expects).
    - GLM47: strips trailing ``<|observation|>`` or ``<|user|>`` (model emits
      the stop token, but the template also emits it as the next turn's opener).
    - Default: plain concatenation (no boundary handling).

TestTokenizeAdditional
    Behavioral tests for tokenize_additional_messages — the single synthetic-prefix diff that computes incremental token IDs for the complete appended suffix.

    ``test_produces_nonempty_incremental`` is parametrized over:
      _TOOL_TRAJECTORIES (trajectory classes) × _TITO_MODELS (qwen3, glm47)
    Split points are auto-detected by _find_tito_splits from message structure,
    so adding a trajectory to _TOOL_TRAJECTORIES automatically extends coverage.

    Remaining tests cover complete-appendix rendering, generation-prompt
    timing, stored tool-call preservation, merge structure preservation, and
    append-only validation (reject prefix mutation, fewer messages, or
    forbidden roles).

TestFactory
    get_tito_tokenizer factory: string/enum dispatch, invalid input handling.
"""

from __future__ import annotations

import json
from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from transformers import AutoTokenizer

from miles.utils.chat_template_utils import (
    TEMPLATE_DIR,
    MismatchType,
    apply_chat_template,
    apply_chat_template_from_str,
    resolve_fixed_chat_template,
)
from miles.utils.chat_template_utils.tito_tokenizer import (
    ALL_APPEND_ROLES,
    DeepSeekV4TITOTokenizer,
    DeepSeekV32TITOTokenizer,
    FixedTemplate,
    GLM47TITOTokenizer,
    GLM53TITOTokenizer,
    InklingTITOTokenizer,
    Kimi25TITOTokenizer,
    Kimi26TITOTokenizer,
    Nemotron3TITOTokenizer,
    Qwen3TITOTokenizer,
    Qwen35TITOTokenizer,
    Qwen36TITOTokenizer,
    Qwen38SmallTITOTokenizer,
    QwenNextTITOTokenizer,
    TITOTokenizer,
    TITOTokenizerType,
    _build_dummy_assistant,
    extract_template_args,
    get_tito_tokenizer,
)
from miles.utils.processing_utils import load_tokenizer
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

_TOK_CACHE: dict[tuple[str, str | None], AutoTokenizer] = {}


def _get_tokenizer(model_id: str, tito_type: TITOTokenizerType | None = None) -> AutoTokenizer:
    chat_template_path = resolve_fixed_chat_template(tito_type)[0] if tito_type is not None else None
    cache_key = (model_id, chat_template_path)
    if cache_key not in _TOK_CACHE:
        _TOK_CACHE[cache_key] = load_tokenizer(
            model_id,
            chat_template_path=chat_template_path,
            trust_remote_code=True,
        )
    return _TOK_CACHE[cache_key]


# ---------------------------------------------------------------------------
# Fixtures — model-specific TITO tokenizers
#
# `tito` is parametrized over all supported models; use it for tests that
# should run against every model.  Named fixtures (qwen3_tito, etc.) are
# for tests specific to one subclass's boundary logic.
# ---------------------------------------------------------------------------

_TITO_MODELS: dict[str, tuple[str, type[TITOTokenizer], TITOTokenizerType]] = {
    "qwen3": ("Qwen/Qwen3-4B", Qwen3TITOTokenizer, TITOTokenizerType.QWEN3),
    "glm47": ("zai-org/GLM-4.7-Flash", GLM47TITOTokenizer, TITOTokenizerType.GLM47),
}


@pytest.fixture(params=list(_TITO_MODELS.keys()))
def tito(request) -> TITOTokenizer:
    model_id, cls, tito_type = _TITO_MODELS[request.param]
    return cls(
        _get_tokenizer(model_id, tito_type),
        chat_template_kwargs={"clear_thinking": False},
    )


@pytest.fixture
def qwen3_tito() -> Qwen3TITOTokenizer:
    return Qwen3TITOTokenizer(
        _get_tokenizer("Qwen/Qwen3-4B", TITOTokenizerType.QWEN3),
        chat_template_kwargs={"clear_thinking": False},
    )


@pytest.fixture
def glm47_tito() -> GLM47TITOTokenizer:
    return GLM47TITOTokenizer(
        _get_tokenizer("zai-org/GLM-4.7-Flash", TITOTokenizerType.GLM47),
        chat_template_kwargs={"clear_thinking": False},
    )


@pytest.fixture
def default_tito() -> TITOTokenizer:
    return TITOTokenizer(_get_tokenizer("Qwen/Qwen3-4B"))


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
    ``new_msgs`` extends through all subsequent non-assistant messages
    (tool/user/system), stopping before the next assistant turn.
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

    @pytest.mark.parametrize(
        "chat_template_kwargs, expected",
        [
            pytest.param({}, True, id="default-thinking"),
            pytest.param({"enable_thinking": False}, False, id="disable-via-miles-kwarg"),
            pytest.param({"thinking": False}, False, id="disable-via-sglang-kwarg"),
            pytest.param(
                {"enable_thinking": False, "thinking": True},
                False,
                id="miles-kwarg-precedes-sglang-kwarg",
            ),
            pytest.param(
                {"thinking_mode": "thinking", "thinking": False},
                True,
                id="explicit-mode-precedes-sglang-kwarg",
            ),
            pytest.param({"thinking_mode": "chat"}, False, id="explicit-chat-mode"),
        ],
    )
    def test_deepseek_v32_forwards_effective_thinking_mode(self, chat_template_kwargs, expected):
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.side_effect = [1, 2]

        tito = DeepSeekV32TITOTokenizer(tokenizer, chat_template_kwargs=chat_template_kwargs)

        assert tito.chat_template_kwargs["thinking"] is expected

    @pytest.mark.parametrize("tito_cls", [DeepSeekV32TITOTokenizer, DeepSeekV4TITOTokenizer])
    @pytest.mark.parametrize(
        "startup_thinking, request_kwargs, expected",
        [
            pytest.param(False, {"thinking": True}, True, id="enable-via-thinking"),
            pytest.param(True, {"enable_thinking": False}, False, id="disable-via-enable-thinking"),
            pytest.param(True, {"thinking_mode": "chat"}, False, id="disable-via-mode"),
            pytest.param(False, {}, False, id="omitted-alias-inherits"),
            pytest.param(False, {"enable_thinking": None}, True, id="null-enable-thinking-replaces-base"),
            pytest.param(False, {"thinking": None}, True, id="null-thinking-replaces-base"),
            pytest.param(True, {"thinking_mode": None}, False, id="null-mode-replaces-base"),
            pytest.param(
                True,
                {"enable_thinking": False, "thinking": True},
                False,
                id="request-enable-thinking-precedes-thinking",
            ),
            pytest.param(
                False, {"thinking_mode": "thinking", "enable_thinking": False}, True, id="request-mode-precedes-toggle"
            ),
            pytest.param(
                True, {"enable_thinking": None, "thinking": False}, False, id="null-toggle-falls-through-to-thinking"
            ),
        ],
    )
    def test_deepseek_request_aliases_override_startup_mode(
        self, tito_cls, startup_thinking, request_kwargs, expected
    ):
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.return_value = 1
        startup_tito = tito_cls(
            tokenizer, chat_template_kwargs={"enable_thinking": startup_thinking, "custom_option": "launch"}
        )
        original_kwargs = dict(request_kwargs)

        request_args = startup_tito.resolve_request_args({"chat_template_kwargs": request_kwargs}, turn_args=None)

        assert request_args == {
            "chat_template_kwargs": {"drop_thinking": False, "thinking": expected, "custom_option": "launch"},
            "tools": None,
        }
        assert startup_tito.chat_template_kwargs == {
            "drop_thinking": False,
            "thinking": startup_thinking,
            "custom_option": "launch",
        }
        assert request_kwargs == original_kwargs

    def test_comparator_inherits_trailing_ids(self, qwen3_tito: Qwen3TITOTokenizer):
        """create_comparator propagates trailing_token_ids to the comparator's trim set."""
        comp = qwen3_tito.create_comparator()
        assert comp._trim_trailing_ids == set(qwen3_tito.trailing_token_ids)


class TestResolveRequestArgs:
    LAUNCH = {"enable_thinking": False}
    TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
    OTHER_TOOLS = [{"type": "function", "function": {"name": "get_time"}}]

    def test_new_root_merges_request_kwargs_over_the_launch_and_takes_the_request_tools(self):
        launch = TITOTokenizer(MagicMock(), chat_template_kwargs=self.LAUNCH)

        args = launch.resolve_request_args(
            {"chat_template_kwargs": {"enable_thinking": True}, "tools": self.TOOLS}, turn_args=None
        )

        assert args == {"chat_template_kwargs": {"enable_thinking": True}, "tools": self.TOOLS}

    def test_request_without_kwargs_or_tools_renders_like_the_launch(self):
        launch = TITOTokenizer(MagicMock(), chat_template_kwargs=self.LAUNCH)
        assert launch.resolve_request_args({"messages": []}, turn_args=None) == {
            "messages": [],
            "chat_template_kwargs": self.LAUNCH,
            "tools": None,
        }
        assert launch.resolve_request_args({"chat_template_kwargs": None, "tools": []}, turn_args=None) == {
            "chat_template_kwargs": self.LAUNCH,
            "tools": None,
        }

    def test_continued_turn_inherits_omitted_kwargs_and_request_overrides_history(self):
        launch = TITOTokenizer(MagicMock(), chat_template_kwargs=self.LAUNCH)
        recorded = {"chat_template_kwargs": {"enable_thinking": True}}

        assert launch.resolve_request_args({}, turn_args=recorded) == {**recorded, "tools": None}
        same = launch.resolve_request_args({"chat_template_kwargs": {"enable_thinking": True}}, turn_args=recorded)
        assert same == {**recorded, "tools": None}
        assert launch.resolve_request_args(
            {"chat_template_kwargs": {"enable_thinking": False}}, turn_args=recorded
        ) == {"chat_template_kwargs": {"enable_thinking": False}, "tools": None}
        assert recorded == {"chat_template_kwargs": {"enable_thinking": True}}

    def test_empty_history_keeps_defaults_but_does_not_allow_new_tools(self):
        launch = TITOTokenizer(MagicMock(), chat_template_kwargs=self.LAUNCH)
        assert launch.resolve_request_args({}, turn_args={}) == {"chat_template_kwargs": self.LAUNCH, "tools": None}
        assert launch.resolve_request_args({"tools": self.TOOLS}, turn_args=None)["tools"] == self.TOOLS
        with pytest.raises(ValueError, match="tools changed"):
            launch.resolve_request_args({"tools": self.TOOLS}, turn_args={})

    def test_continued_turn_tools_inherit_pass_or_are_refused(self):
        launch = TITOTokenizer(MagicMock())
        recorded = {"tools": self.TOOLS}

        assert launch.resolve_request_args({}, turn_args=recorded) == {"chat_template_kwargs": {}, "tools": self.TOOLS}
        bare = [tool["function"] for tool in self.TOOLS]  # the same tools in the other accepted spelling
        assert launch.resolve_request_args({"tools": bare}, turn_args=recorded) == {
            "chat_template_kwargs": {},
            "tools": bare,
        }
        with pytest.raises(ValueError, match="tools changed on a continued turn"):
            launch.resolve_request_args({"tools": self.OTHER_TOOLS}, turn_args=recorded)
        with pytest.raises(ValueError, match="tools changed on a continued turn"):
            launch.resolve_request_args({"tools": self.TOOLS}, turn_args={})  # the turn had none

    def test_a_family_may_allow_tools_to_change_mid_session(self):
        class _ToolsAtTheTailTITOTokenizer(TITOTokenizer):
            def resolve_tools(self, request_args, *, request_source, turn_args):
                tools = request_source.get("tools") or (turn_args or {}).get("tools")
                if tools:
                    request_args["tools"] = deepcopy(tools)

        family = _ToolsAtTheTailTITOTokenizer(MagicMock())
        args = family.resolve_request_args({"tools": self.OTHER_TOOLS}, turn_args={"tools": self.TOOLS})
        assert args == {"chat_template_kwargs": {}, "tools": self.OTHER_TOOLS}

    def test_family_constants_override_requested_values(self, qwen3_tito: Qwen3TITOTokenizer):
        args = qwen3_tito.resolve_request_args({"chat_template_kwargs": {"enable_thinking": True}}, turn_args=None)
        assert args == {"chat_template_kwargs": {"clear_thinking": False, "enable_thinking": True}, "tools": None}
        assert qwen3_tito.resolve_request_args({"chat_template_kwargs": {"clear_thinking": True}}, turn_args=None) == {
            "chat_template_kwargs": {"clear_thinking": False},
            "tools": None,
        }

    @pytest.mark.parametrize("tito_cls", [DeepSeekV32TITOTokenizer, DeepSeekV4TITOTokenizer])
    def test_deepseek_thinking_rule_uses_resolved_kwargs(self, tito_cls):
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.return_value = 1
        model = tito_cls(tokenizer, chat_template_kwargs={"thinking": False})
        request_args = {
            "chat_template_kwargs": {"thinking_mode": "thinking", "custom_option": "resolved"},
            "temperature": 0.7,
        }

        model.resolve_thinking(
            request_args,
            request_source={"chat_template_kwargs": {"enable_thinking": False}},
            turn_args={"chat_template_kwargs": {"thinking": False}},
        )

        assert request_args == {
            "chat_template_kwargs": {"thinking": True, "custom_option": "resolved"},
            "temperature": 0.7,
        }

    @pytest.mark.parametrize("tito_cls", [DeepSeekV32TITOTokenizer, DeepSeekV4TITOTokenizer])
    def test_deepseek_recorded_mode_overrides_request_aliases(self, tito_cls):
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.return_value = 1
        launch = tito_cls(tokenizer, chat_template_kwargs={"enable_thinking": False})
        assert launch.chat_template_kwargs == {"drop_thinking": False, "thinking": False}

        recorded = launch.resolve_request_args({"chat_template_kwargs": {"thinking": True}}, turn_args=None)
        assert recorded == {"chat_template_kwargs": {"drop_thinking": False, "thinking": True}, "tools": None}
        same = launch.resolve_request_args({"chat_template_kwargs": {"enable_thinking": True}}, turn_args=recorded)
        assert same == recorded
        assert (
            launch.resolve_request_args({"chat_template_kwargs": {"thinking_mode": "chat"}}, turn_args=recorded)
            == recorded
        )

    def test_returns_the_same_full_request_without_inheriting_sampling_fields(self):
        launch = TITOTokenizer(MagicMock())
        turn_args = {"temperature": 0.2, "seed": 42, "chat_template_kwargs": {"enable_thinking": True}}
        request_args = {"temperature": 0.8, "model": "m"}
        result = launch.resolve_request_args(request_args, turn_args=turn_args)
        assert result is request_args
        assert result == {
            "temperature": 0.8,
            "model": "m",
            "chat_template_kwargs": {"enable_thinking": True},
            "tools": None,
        }
        assert turn_args["temperature"] == 0.2

    def test_inherited_nested_values_are_owned_by_the_working_request(self):
        launch = TITOTokenizer(MagicMock(), chat_template_kwargs={"options": {"labels": ["launch"]}})
        request_args = launch.resolve_request_args({}, turn_args=None)
        request_args["chat_template_kwargs"]["options"]["labels"].append("request")
        assert launch.chat_template_kwargs == {"options": {"labels": ["launch"]}}

        turn_args = {"chat_template_kwargs": {"options": {"labels": ["turn"]}}, "tools": self.TOOLS}
        request_args = launch.resolve_request_args({}, turn_args=turn_args)
        request_args["chat_template_kwargs"]["options"]["labels"].append("next")
        request_args["tools"][0]["function"]["name"] = "changed"
        assert turn_args["chat_template_kwargs"] == {"options": {"labels": ["turn"]}}
        assert turn_args["tools"][0]["function"]["name"] == "get_weather"

    def test_appended_rules_receive_original_sources_and_can_change_full_request(self):
        class Model(TITOTokenizer):
            def request_arg_rules(self):
                rules = super().request_arg_rules()
                rules.append(self.resolve_custom_args)
                return rules

            def resolve_custom_args(self, request_args, *, request_source, turn_args):
                if "temperature" not in request_source:
                    return
                assert request_source["chat_template_kwargs"] == {"choice": "request"}
                assert request_args["chat_template_kwargs"] == {"choice": "request", "from_launch": True}
                request_args["temperature"] = 0.25
                request_args["tool_choice"] = "auto"
                request_args["tools"][0]["function"]["name"] = "rewritten"
                request_args["chat_template_kwargs"]["choice"] = "model"
                assert request_source["tools"][0]["function"]["name"] == "get_weather"

        launch = {"choice": "launch", "from_launch": True}
        model = Model(MagicMock(), chat_template_kwargs=launch)
        original = {"temperature": 0.8, "tools": self.TOOLS, "chat_template_kwargs": {"choice": "request"}}
        request_args = deepcopy(original)
        assert model.resolve_request_args(request_args, turn_args=None) is request_args
        assert request_args["temperature"] == 0.25
        assert request_args["tool_choice"] == "auto"
        assert request_args["tools"][0]["function"]["name"] == "rewritten"
        assert request_args["chat_template_kwargs"]["choice"] == "model"
        assert launch == {"choice": "launch", "from_launch": True}
        assert len(TITOTokenizer(MagicMock()).request_arg_rules()) == 4
        assert len(model.request_arg_rules()) == 5

    @pytest.mark.parametrize("recorded_present", [True, False], ids=["recorded-value", "recorded-absence"])
    @pytest.mark.parametrize(
        "tito_cls, key, recorded_value, requested_value",
        [
            (Qwen35TITOTokenizer, "add_vision_id", True, False),
            (Qwen36TITOTokenizer, "add_vision_id", True, False),
            (Qwen38SmallTITOTokenizer, "add_vision_id", True, False),
            (Qwen38SmallTITOTokenizer, "enable_thinking", True, False),
            (Qwen38SmallTITOTokenizer, "reasoning_effort", "low", "medium"),
            (GLM53TITOTokenizer, "reasoning_effort", "max", "low"),
            (Nemotron3TITOTokenizer, "low_effort", True, False),
            (Kimi25TITOTokenizer, "thinking", True, False),
            (Kimi25TITOTokenizer, "tools_ts_str", "old declaration", "new declaration"),
            (Kimi26TITOTokenizer, "tools_ts_str", "old declaration", "new declaration"),
            (DeepSeekV32TITOTokenizer, "add_default_bos_token", True, False),
            (DeepSeekV32TITOTokenizer, "context", [{"role": "system", "content": "old"}], []),
            (DeepSeekV4TITOTokenizer, "reasoning_effort", "max", "high"),
            (DeepSeekV4TITOTokenizer, "add_default_bos_token", True, False),
            (DeepSeekV4TITOTokenizer, "context", [{"role": "system", "content": "old"}], []),
            (InklingTITOTokenizer, "reasoning_effort", "high", "low"),
        ],
    )
    def test_model_rules_keep_fields_that_affect_the_reused_prefix(
        self, tito_cls, key, recorded_value, requested_value, recorded_present
    ):
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1]
        tokenizer.convert_tokens_to_ids.return_value = 1
        model = tito_cls(tokenizer, chat_template_kwargs={key: requested_value})
        launch = deepcopy(model.chat_template_kwargs)
        turn_kwargs = deepcopy(launch)
        turn_kwargs.pop(key, None)
        if recorded_present:
            turn_kwargs[key] = deepcopy(recorded_value)
        turn_args = {"chat_template_kwargs": turn_kwargs}
        original_turn = deepcopy(turn_args)
        request_args = {"chat_template_kwargs": {key: requested_value, "custom_option": "new"}}

        result = model.resolve_request_args(request_args, turn_args=turn_args)

        assert result is request_args
        if recorded_present:
            assert result["chat_template_kwargs"][key] == recorded_value
            if key == "context":
                result["chat_template_kwargs"][key][0]["content"] = "changed"
        else:
            assert key not in result["chat_template_kwargs"]
        assert result["chat_template_kwargs"]["custom_option"] == "new"
        assert turn_args == original_turn
        assert model.chat_template_kwargs == launch

    @pytest.mark.parametrize(
        "tito_cls, key, old_value, new_value, content",
        [
            (InklingTITOTokenizer, "reasoning_effort", "high", "low", "hello"),
            (Qwen38SmallTITOTokenizer, "enable_thinking", True, False, "hello"),
            (Qwen38SmallTITOTokenizer, "reasoning_effort", "low", "medium", "hello"),
            (
                Qwen35TITOTokenizer,
                "add_vision_id",
                True,
                False,
                [{"type": "image", "image": "image"}, {"type": "text", "text": "describe"}],
            ),
        ],
    )
    def test_model_rules_preserve_rendered_prefix_when_request_changes_a_locked_field(
        self, tito_cls, key, old_value, new_value, content
    ):
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1]
        tokenizer.convert_tokens_to_ids.return_value = 1
        model = tito_cls(tokenizer)
        turn_args = model.resolve_request_args({"chat_template_kwargs": {key: old_value}}, turn_args=None)
        requested = {"chat_template_kwargs": {key: new_value}}
        resolved = model.resolve_request_args(requested, turn_args=turn_args)
        template_text = (TEMPLATE_DIR / model.FIXED_TEMPLATE.template).read_text()
        messages = [{"role": "user", "content": content}]
        old_kwargs = extract_template_args(turn_args)
        old_render = apply_chat_template_from_str(template_text, messages, add_generation_prompt=False, **old_kwargs)
        changed_render = apply_chat_template_from_str(
            template_text, messages, add_generation_prompt=False, **{**old_kwargs, key: new_value}
        )
        assert changed_render != old_render
        assert (
            apply_chat_template_from_str(
                template_text, messages, add_generation_prompt=False, **extract_template_args(resolved)
            )
            == old_render
        )

    @pytest.mark.parametrize(
        "launch_kwargs, request_kwargs, expected_effort",
        [
            ({}, {}, None),
            ({"reasoning_effort": "low"}, {}, "low"),
            ({"reasoning_effort": "low"}, {"reasoning_effort": "medium"}, "medium"),
            ({"reasoning_effort": "medium"}, {"reasoning_effort": "low"}, "low"),
            ({"reasoning_effort": "low"}, {"reasoning_effort": "xhigh"}, "xhigh"),
        ],
    )
    def test_qwen38_new_root_selects_effort(self, launch_kwargs, request_kwargs, expected_effort):
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1]
        model = Qwen38SmallTITOTokenizer(tokenizer, chat_template_kwargs=launch_kwargs)
        result = model.resolve_request_args(
            {"chat_template_kwargs": {**request_kwargs, "preserve_thinking": False}}, turn_args=None
        )
        expected = {"preserve_thinking": True}
        if expected_effort is not None:
            expected["reasoning_effort"] = expected_effort
        assert result["chat_template_kwargs"] == expected

    @pytest.mark.parametrize("turn_args", [None, {}, {"chat_template_kwargs": {"reasoning_effort": "low"}}])
    def test_qwen38_effort_follows_history(self, turn_args):
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1]
        tokenizer.convert_tokens_to_ids.return_value = 1
        model = Qwen38SmallTITOTokenizer(tokenizer, chat_template_kwargs={"reasoning_effort": "xhigh"})
        original_turn = deepcopy(turn_args)
        request_args = {"chat_template_kwargs": {"reasoning_effort": "medium"}}

        assert model.resolve_request_args(request_args, turn_args=turn_args) is request_args

        kwargs = request_args["chat_template_kwargs"]
        if turn_args is None:
            assert kwargs["reasoning_effort"] == "medium"
        elif turn_args:
            assert kwargs["reasoning_effort"] == "low"
        else:
            assert "reasoning_effort" not in kwargs
        assert turn_args == original_turn

    def test_qwen38_omitted_effort_uses_template_default(self):
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1]
        model = Qwen38SmallTITOTokenizer(tokenizer)
        result = model.resolve_request_args({}, turn_args=None)
        kwargs = extract_template_args(result)
        assert "reasoning_effort" not in kwargs
        template_text = (TEMPLATE_DIR / model.FIXED_TEMPLATE.template).read_text()
        messages = [{"role": "user", "content": "hello"}]
        rendered = apply_chat_template_from_str(template_text, messages, add_generation_prompt=True, **kwargs)
        assert rendered == apply_chat_template_from_str(
            template_text, messages, add_generation_prompt=True, **{**kwargs, "reasoning_effort": "xhigh"}
        )


class TestCompletionPostprocess:
    def test_default_returns_upstream_message_unchanged(self):
        tito = TITOTokenizer(MagicMock())
        assistant_message = {"role": "assistant", "content": "upstream"}
        choice = {
            "message": assistant_message,
            "finish_reason": "stop",
            "meta_info": {"existing": True},
        }

        stored_message = tito.postprocess_completion(
            choice=choice,
            assistant_message=assistant_message,
            completion_token_ids=[1, 2, 3],
        )

        assert stored_message is assistant_message
        assert choice == {
            "message": assistant_message,
            "finish_reason": "stop",
            "meta_info": {"existing": True},
        }


class TestInklingComparatorBoundaries:
    @staticmethod
    def _build_comparator():
        token_ids = {
            "<|message_user|>": 200000,
            "<|message_model|>": 200001,
            "<|message_system|>": 200002,
            "<|message_tool|>": 200003,
        }
        token_text = {token_id: token for token, token_id in token_ids.items()}
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.side_effect = token_ids.__getitem__
        tokenizer.encode.side_effect = lambda text, add_special_tokens=False: [ord(char) for char in text]
        tokenizer.decode.side_effect = lambda ids, skip_special_tokens=False: "".join(
            token_text.get(token_id, chr(token_id)) for token_id in ids
        )
        comparator = InklingTITOTokenizer(tokenizer).create_comparator()
        return token_ids, tokenizer, comparator

    def test_uses_exact_message_role_boundaries(self):
        token_ids, _, comparator = self._build_comparator()

        assert comparator._special_ids == set(token_ids.values())

    @pytest.mark.parametrize("appended_role", ["user", "system", "tool"])
    def test_appended_non_assistant_content_is_not_assistant_text(self, appended_role):
        token_ids, tokenizer, comparator = self._build_comparator()
        assistant = [token_ids["<|message_model|>"]] + tokenizer.encode("assistant")
        role_token = token_ids[f"<|message_{appended_role}|>"]
        expected = assistant + [role_token] + tokenizer.encode("expected")
        actual = assistant + [role_token] + tokenizer.encode("changed")

        mismatches = comparator.compare_sequences(expected, actual)

        assert [mismatch.type for mismatch in mismatches] == [MismatchType.NON_ASSISTANT_TEXT]

    def test_assistant_content_remains_soft_mismatch(self):
        token_ids, tokenizer, comparator = self._build_comparator()
        assistant_role = [token_ids["<|message_model|>"]]
        expected = assistant_role + tokenizer.encode("expected")
        actual = assistant_role + tokenizer.encode("changed")

        mismatches = comparator.compare_sequences(expected, actual)

        assert [mismatch.type for mismatch in mismatches] == [MismatchType.ASSISTANT_TEXT]


class TestInklingFixedTemplate:
    @staticmethod
    def _render(messages):
        template = (TEMPLATE_DIR / "inkling_fixed.jinja").read_text()
        return apply_chat_template_from_str(template, messages, add_generation_prompt=False)

    def test_tool_call_only_turn_skips_empty_text_block(self):
        messages = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "check",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"location": "Beijing"}},
                    }
                ],
            },
        ]

        rendered = self._render(messages)

        assert "<|message_model|><|content_text|><|end_message|>" not in rendered
        assert (
            "<|message_model|><|content_thinking|>check<|end_message|>"
            '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather",'
            '"args":{"location":"Beijing"}}<|end_message|><|content_model_end_sampling|>'
        ) in rendered

    def test_empty_assistant_turn_skips_bare_terminator(self):
        rendered = self._render(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": ""},
            ]
        )

        assert "<|message_model|>" not in rendered
        assert "<|content_model_end_sampling|>" not in rendered

    def test_nonempty_reasoning_and_final_text_are_preserved(self):
        rendered = self._render(
            [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "done",
                    "reasoning_content": "think",
                },
            ]
        )

        assert (
            "<|message_model|><|content_thinking|>think<|end_message|>"
            "<|message_model|><|content_text|>done<|end_message|>"
            "<|content_model_end_sampling|>"
        ) in rendered

    def test_ordered_blocks_preserve_thinking_tool_call_and_text_order(self):
        raw_json = '{ "args": {"command": "pwd"}, "name": "bash_command" }'
        rendered = self._render(
            [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "flattened-text",
                    "reasoning_content": "flattened-thinking",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "bash_command", "arguments": {"command": "pwd"}},
                        }
                    ],
                    "content_blocks": [
                        {"type": "thinking", "text": "think-1"},
                        {"type": "text", "text": ""},
                        {
                            "type": "tool_call",
                            "recipient": "bash",
                            "name": "bash_command",
                            "arguments": {"command": "pwd"},
                            "raw_json": raw_json,
                        },
                        {"type": "thinking", "text": "think-2"},
                        {"type": "text", "text": "done"},
                    ],
                },
            ]
        )

        assert (
            "<|message_model|><|content_thinking|>think-1<|end_message|>"
            "<|message_model|><|content_text|><|end_message|>"
            f"<|message_model|>bash<|content_invoke_tool_json|>{raw_json}<|end_message|>"
            "<|message_model|><|content_thinking|>think-2<|end_message|>"
            "<|message_model|><|content_text|>done<|end_message|>"
            "<|content_model_end_sampling|>"
        ) in rendered
        assert "flattened-text" not in rendered
        assert "flattened-thinking" not in rendered
        assert rendered.count("<|content_invoke_tool_json|>") == 1

    def test_empty_ordered_block_sequence_keeps_sampling_terminator(self):
        rendered = self._render(
            [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "",
                    "content_blocks": [],
                },
            ]
        )

        assert rendered.endswith("<|content_model_end_sampling|>")


class TestDeepSeekV32IncrementalAppend:
    """V3.2 rides the default synthetic-prefix suffix diff; with the family's
    pinned ``drop_thinking=False`` the vendored encoder renders every turn
    position-independently, so the synthetic-prefix incremental must equal the
    real-history render suffix for both tool and user appends."""

    class _CharTokenizer:
        def __init__(self, name_or_path: str):
            self.name_or_path = name_or_path

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [ord(c) for c in text]

        def convert_tokens_to_ids(self, token):
            return {"<｜User｜>": 1, "<｜Assistant｜>": 2}[token]

    @pytest.mark.parametrize(
        "appended",
        [
            [{"role": "user", "content": "next question"}],
            [{"role": "tool", "content": "out", "tool_call_id": "c0"}],
            [{"role": "assistant", "content": "injected"}, {"role": "user", "content": "next"}],
            [
                {"role": "assistant", "content": "first injected"},
                {"role": "assistant", "content": "second injected"},
                {"role": "user", "content": "next"},
            ],
        ],
        ids=["user", "tool", "assistant_then_user", "consecutive_assistants_then_user"],
    )
    def test_incremental_equals_real_history_suffix(self, tmp_path, appended):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v32"}), encoding="utf-8")
        tokenizer = self._CharTokenizer(str(tmp_path))
        tito = DeepSeekV32TITOTokenizer(
            tokenizer,
            chat_template_kwargs={"drop_thinking": False},
        )
        old = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "r",
                "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}],
            },
        ]
        new = old + appended

        incremental = tito.tokenize_additional_messages(old, new)

        text_old = tito.apply_chat_template(old, add_generation_prompt=False)
        text_new = tito.apply_chat_template(new, add_generation_prompt=True)
        assert text_new.startswith(text_old)
        assert incremental == tokenizer.encode(text_new[len(text_old) :])


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
        incremental = qwen3_tito.tokenize_additional_messages(
            _BND_OLD, _BND_NEW, template_args=qwen3_tito.default_template_args(_BND_TOOLS)
        )
        im_end = qwen3_tito._im_end_id
        nl = qwen3_tito._newline_id

        result = qwen3_tito.merge_tokens(
            _BND_OLD, _BND_NEW, [100, 200, im_end], template_args=qwen3_tito.default_template_args(_BND_TOOLS)
        )
        assert result == [100, 200, im_end, nl] + incremental

    def test_qwen3_no_newline_otherwise(self, qwen3_tito: Qwen3TITOTokenizer):
        """No insertion when prefix does not end with <|im_end|>."""
        incremental = qwen3_tito.tokenize_additional_messages(
            _BND_OLD, _BND_NEW, template_args=qwen3_tito.default_template_args(_BND_TOOLS)
        )
        result = qwen3_tito.merge_tokens(
            _BND_OLD, _BND_NEW, [100, 200, 300], template_args=qwen3_tito.default_template_args(_BND_TOOLS)
        )
        assert result == [100, 200, 300] + incremental

    # -- GLM47: strip ambiguous boundary tokens --

    def test_glm47_strips_observation(self, glm47_tito: GLM47TITOTokenizer):
        """Model emits <|observation|> as stop token; merge_tokens strips the duplicate."""
        incremental = glm47_tito.tokenize_additional_messages(
            _BND_OLD, _BND_NEW, template_args=glm47_tito.default_template_args(_BND_TOOLS)
        )
        result = glm47_tito.merge_tokens(
            _BND_OLD,
            _BND_NEW,
            [100, 200, glm47_tito._observation_id],
            template_args=glm47_tito.default_template_args(_BND_TOOLS),
        )
        assert result == [100, 200] + incremental

    def test_glm47_strips_user(self, glm47_tito: GLM47TITOTokenizer):
        """<|user|> is also an ambiguous boundary — stripped the same way."""
        incremental = glm47_tito.tokenize_additional_messages(
            _BND_OLD, _BND_NEW, template_args=glm47_tito.default_template_args(_BND_TOOLS)
        )
        result = glm47_tito.merge_tokens(
            _BND_OLD,
            _BND_NEW,
            [100, 200, glm47_tito._user_id],
            template_args=glm47_tito.default_template_args(_BND_TOOLS),
        )
        assert result == [100, 200] + incremental

    def test_glm47_no_strip_otherwise(self, glm47_tito: GLM47TITOTokenizer):
        """Non-boundary trailing token is preserved."""
        incremental = glm47_tito.tokenize_additional_messages(
            _BND_OLD, _BND_NEW, template_args=glm47_tito.default_template_args(_BND_TOOLS)
        )
        result = glm47_tito.merge_tokens(
            _BND_OLD, _BND_NEW, [100, 200, 300], template_args=glm47_tito.default_template_args(_BND_TOOLS)
        )
        assert result == [100, 200, 300] + incremental

    # -- Default: no boundary handling --

    def test_default_concatenates(self, default_tito: TITOTokenizer):
        """Base class does plain concatenation without any prefix modification."""
        incremental = default_tito.tokenize_additional_messages(
            _BND_OLD, _BND_NEW, template_args=default_tito.default_template_args(_BND_TOOLS)
        )
        result = default_tito.merge_tokens(
            _BND_OLD, _BND_NEW, [100, 200, 300], template_args=default_tito.default_template_args(_BND_TOOLS)
        )
        assert result == [100, 200, 300] + incremental

    # -- Edge case --

    def test_empty_prefix(self, qwen3_tito: Qwen3TITOTokenizer):
        """Empty prefix → no boundary handling, result is just incremental."""
        incremental = qwen3_tito.tokenize_additional_messages(
            _BND_OLD, _BND_NEW, template_args=qwen3_tito.default_template_args(_BND_TOOLS)
        )
        result = qwen3_tito.merge_tokens(
            _BND_OLD, _BND_NEW, [], template_args=qwen3_tito.default_template_args(_BND_TOOLS)
        )
        assert result == incremental


# ---------------------------------------------------------------------------
# TestTokenizeAdditional — one synthetic diff for the complete appended suffix
#
# test_produces_nonempty_incremental is the scalable core: parametrized over
# _TRAJ_CASES (trajectories × split points) × tito fixture (models).
# 8 trajectories × ~14 splits × 2 models = 28 test cases currently.
#
# Validation tests use a single trajectory since the validation logic
# (assert_messages_append_only_with_allowed_role) is model/trajectory-independent.
# ---------------------------------------------------------------------------


class TestTokenizeAdditional:
    """tokenize_additional_messages produces valid incremental tokens."""

    @pytest.mark.parametrize("traj_cls, pos", _TRAJ_CASES)
    def test_produces_nonempty_incremental(self, tito: TITOTokenizer, traj_cls, pos):
        """Every valid TITO split yields non-empty incremental tokens.

        This is the primary scalability test — it runs every trajectory's
        TITO splits against every model tokenizer.
        """
        old_msgs, new_msgs, tools = _split_at(traj_cls, pos)
        incremental = tito.tokenize_additional_messages(
            old_msgs, new_msgs, template_args=tito.default_template_args(tools)
        )
        assert len(incremental) > 0

    def test_complete_appendix_reaches_renderer_and_its_error_propagates(
        self, qwen3_tito: Qwen3TITOTokenizer, monkeypatch
    ):
        old_msgs = list(SingleToolTrajectory.MESSAGES[:3])
        appended = [
            SingleToolTrajectory.MESSAGES[3],
            {"role": "user", "content": "Try another route."},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": false}'},
        ]
        calls = []

        def reject_invalid_order(base_messages, appended_messages, *, template_args=None, add_generation_prompt=False):
            calls.append((base_messages, appended_messages, template_args, add_generation_prompt))
            if [message["role"] for message in appended_messages] == ["tool", "user", "tool"]:
                raise ValueError("invalid tool ordering")
            return [1]

        monkeypatch.setattr(qwen3_tito, "_tokenize_rendered_suffix", reject_invalid_order)

        with pytest.raises(ValueError, match="invalid tool ordering"):
            qwen3_tito.tokenize_additional_messages(
                old_msgs,
                old_msgs + appended,
                template_args=qwen3_tito.default_template_args(SingleToolTrajectory.TOOLS),
            )

        assert len(calls) == 1
        base_messages, rendered_appendix, template_args, add_generation_prompt = calls[0]
        assert [message["role"] for message in base_messages] == ["system", "assistant"]
        assert base_messages[-1]["tool_calls"] == old_msgs[-1]["tool_calls"]
        assert rendered_appendix == appended
        assert template_args == qwen3_tito.default_template_args(SingleToolTrajectory.TOOLS)
        assert add_generation_prompt is True

    def test_generation_prompt_is_appended_once_for_full_suffix(self, qwen3_tito: Qwen3TITOTokenizer):
        old_msgs = list(SingleToolThinkingTrajectory.MESSAGES[:3])
        new_msgs = old_msgs + [
            SingleToolThinkingTrajectory.MESSAGES[3],
            {"role": "user", "content": "Now check Shanghai too."},
        ]
        tools = SingleToolThinkingTrajectory.TOOLS

        incremental = qwen3_tito.tokenize_additional_messages(
            old_msgs, new_msgs, template_args=qwen3_tito.default_template_args(tools)
        )
        decoded = qwen3_tito.tokenizer.decode(incremental)
        assert decoded.count(qwen3_tito._assistant_start_str) == 1
        assert decoded.endswith(
            qwen3_tito.tokenizer.decode(
                qwen3_tito._tokenize_rendered_suffix(
                    new_msgs, [], template_args=qwen3_tito.default_template_args(tools), add_generation_prompt=True
                )
            )
        )

    def test_dummy_assistant_preserves_parallel_tool_calls(self):
        stored_assistant = MultiToolSingleTurnTrajectory.MESSAGES[2]
        dummy_assistant = _build_dummy_assistant(stored_assistant)

        assert dummy_assistant["content"] == ""
        assert dummy_assistant["reasoning_content"] == " "
        assert dummy_assistant["tool_calls"] == stored_assistant["tool_calls"]
        assert [call["id"] for call in dummy_assistant["tool_calls"]] == ["call_1", "call_2"]

    @pytest.mark.parametrize(
        "traj_cls, pos",
        [
            pytest.param(SingleToolTrajectory, 3, id="single-tool"),
            pytest.param(RetrySystemTrajectory, 3, id="tool-plus-system"),
            pytest.param(IntermediateSystemTrajectory, 3, id="intermediate-system"),
        ],
    )
    def test_qwen3_merge_preserves_non_assistant_structure(self, qwen3_tito: Qwen3TITOTokenizer, traj_cls, pos):
        """Merged tokens may differ in assistant text, but not in tool/system structure."""
        old_msgs, new_msgs, tools = _split_at(traj_cls, pos)
        pretokenized = apply_chat_template(
            old_msgs,
            tokenizer=qwen3_tito.tokenizer,
            tokenize=True,
            add_generation_prompt=False,
            tools=tools,
        )
        merged = qwen3_tito.merge_tokens(
            old_msgs, new_msgs, pretokenized, template_args=qwen3_tito.default_template_args(tools)
        )
        expected = apply_chat_template(
            new_msgs,
            tokenizer=qwen3_tito.tokenizer,
            tokenize=True,
            add_generation_prompt=True,
            tools=tools,
        )
        mismatches = qwen3_tito.create_comparator().compare_sequences(expected, merged)
        assert all(m.type == MismatchType.ASSISTANT_TEXT for m in mismatches)

    # -- Append-only validation (assert_messages_append_only_with_allowed_role is called internally) --

    def test_rejects_prefix_mutation(self, qwen3_tito: Qwen3TITOTokenizer):
        """Modifying an existing message in new_messages raises ValueError."""
        old_msgs, new_msgs, _ = _split_at(SingleToolTrajectory, 3)
        mutated_old = [{"role": "user", "content": "CHANGED"}] + list(old_msgs[1:])
        mutated_new = mutated_old + list(new_msgs[len(old_msgs) :])
        with pytest.raises(ValueError, match="mismatch"):
            qwen3_tito.tokenize_additional_messages(old_msgs, mutated_new)

    def test_rejects_fewer_messages(self, qwen3_tito: Qwen3TITOTokenizer):
        """new_messages shorter than old_messages raises ValueError."""
        old_msgs = SingleToolTrajectory.MESSAGES[:3]
        with pytest.raises(ValueError, match="fewer"):
            qwen3_tito.tokenize_additional_messages(old_msgs, old_msgs[:1])

    def test_restricted_template_rejects_unsupported_role(self, qwen3_tito: Qwen3TITOTokenizer):
        """A template with an explicit narrow capability rejects other roles."""

        class _ToolOnlyQwen3TITOTokenizer(Qwen3TITOTokenizer):
            FIXED_TEMPLATE = FixedTemplate(
                template=Qwen3TITOTokenizer.FIXED_TEMPLATE.template,
                extra_kwargs=dict(Qwen3TITOTokenizer.FIXED_TEMPLATE.extra_kwargs),
                allowed_append_roles=frozenset({"tool"}),
            )

        restricted = _ToolOnlyQwen3TITOTokenizer(
            qwen3_tito.tokenizer,
            chat_template_kwargs={"clear_thinking": False},
        )
        old_msgs = SingleToolTrajectory.MESSAGES[:3]
        bad_new = list(old_msgs) + [{"role": "assistant", "content": "hi"}]
        with pytest.raises(ValueError, match="role"):
            restricted.tokenize_additional_messages(old_msgs, bad_new)


# ---------------------------------------------------------------------------
# TestFactory — get_tito_tokenizer dispatch
# ---------------------------------------------------------------------------


class TestFactory:
    """get_tito_tokenizer creates the correct subclass from string or enum type."""

    @pytest.mark.parametrize(
        "type_str, model_id, cls",
        [
            ("qwen3", "Qwen/Qwen3-4B", Qwen3TITOTokenizer),
            ("qwen35", "Qwen/Qwen3-4B", Qwen35TITOTokenizer),
            ("qwen36", "Qwen/Qwen3-4B", Qwen36TITOTokenizer),
            ("qwen38small", "Qwen/Qwen3-4B", Qwen38SmallTITOTokenizer),
            ("qwen4exp", "Qwen/Qwen3-4B", Qwen38SmallTITOTokenizer),
            ("qwennext", "Qwen/Qwen3-4B", QwenNextTITOTokenizer),
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
        assert tito.allowed_append_roles == ALL_APPEND_ROLES

    @pytest.mark.parametrize(
        "type_str, cls",
        [
            ("qwen35", Qwen35TITOTokenizer),
            ("qwen36", Qwen36TITOTokenizer),
            ("qwen38small", Qwen38SmallTITOTokenizer),
            ("qwen4exp", Qwen38SmallTITOTokenizer),
            ("qwennext", QwenNextTITOTokenizer),
        ],
    )
    def test_qwen_variant_inherits_qwen3_boundary_logic(self, type_str, cls):
        """Qwen3.5 / Qwen3-Next reuse Qwen3's boundary handling via inheritance.
        The named subclass owns its FixedTemplate contract, while token-level
        merge behavior remains identical to Qwen3."""
        tito = get_tito_tokenizer(_get_tokenizer("Qwen/Qwen3-4B"), tokenizer_type=type_str)
        assert isinstance(tito, cls)
        assert isinstance(tito, Qwen3TITOTokenizer)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            get_tito_tokenizer(_get_tokenizer("Qwen/Qwen3-4B"), tokenizer_type="nonexistent")

    def test_none_tokenizer_raises(self):
        with pytest.raises(ValueError, match="must not be None"):
            get_tito_tokenizer(None)


class TestParserBinding:
    """Each TITO subclass binds sglang ``--reasoning-parser`` and
    ``--tool-call-parser`` values; ``resolve_reasoning_and_tool_call_parser``
    enforces user-supplied values agree with the bindings (or returns the
    bound values when the user didn't pass one).  The two parsers are
    resolved independently — a missing binding on one doesn't suppress the
    assert on the other."""

    @pytest.mark.parametrize(
        "tito_model, expected_reasoning, expected_tool_call",
        [
            (TITOTokenizerType.QWEN3, "qwen3", "qwen25"),
            (TITOTokenizerType.QWEN35, "qwen3", "qwen3_coder"),
            (TITOTokenizerType.QWEN36, "qwen3", "qwen3_coder"),
            (TITOTokenizerType.QWEN38_SMALL, "qwen3", "qwen3_coder"),
            (TITOTokenizerType.QWEN4_EXP, "qwen3", "qwen3_coder"),
            (TITOTokenizerType.QWENNEXT, "qwen3", "qwen25"),
            (TITOTokenizerType.GLM47, "glm45", "glm47"),
            (TITOTokenizerType.NEMOTRON3, "nemotron_3", "qwen3_coder"),
            (TITOTokenizerType.KIMI25, None, None),
            (TITOTokenizerType.KIMI26, "kimi_k2", "kimi_k2_raw_id"),
            (TITOTokenizerType.MINIMAX_M25, "minimax-append-think", "minimax-m2"),
            (TITOTokenizerType.MINIMAX_M27, "minimax-append-think", "minimax-m2"),
            (TITOTokenizerType.DEEPSEEKV32, "deepseek-v3", "deepseekv32"),
            (TITOTokenizerType.DEEPSEEKV4, "deepseek-v4", "deepseekv4"),
            (TITOTokenizerType.INKLING, None, None),
            (TITOTokenizerType.DEFAULT, None, None),
        ],
    )
    def test_subclass_binding(self, tito_model, expected_reasoning, expected_tool_call):
        cls = TITOTokenizerType.get_tokenizer_class(tito_model)
        assert cls.reasoning_parser == expected_reasoning
        assert cls.tool_call_parser == expected_tool_call

    def test_resolve_returns_binding_when_user_omits(self):
        from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.QWEN3) == ("qwen3", "qwen25")
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.QWEN35) == ("qwen3", "qwen3_coder")
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.QWEN36) == ("qwen3", "qwen3_coder")
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.QWEN38_SMALL) == ("qwen3", "qwen3_coder")
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.QWEN4_EXP) == ("qwen3", "qwen3_coder")
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.GLM47) == ("glm45", "glm47")
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.GLM53) == ("glm45", "glm47")
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.DEEPSEEKV4) == ("deepseek-v4", "deepseekv4")
        # DEFAULT family has no binding for either parser; both come back None.
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.DEFAULT) == (None, None)

    def test_resolve_accepts_matching_user_value(self):
        from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

        assert resolve_reasoning_and_tool_call_parser("qwen3", "qwen3", "qwen25") == ("qwen3", "qwen25")
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.QWEN35, "qwen3", "qwen3_coder") == (
            "qwen3",
            "qwen3_coder",
        )

    def test_resolve_raises_on_reasoning_mismatch(self):
        from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

        with pytest.raises(ValueError, match="--reasoning-parser='glm45' disagrees"):
            resolve_reasoning_and_tool_call_parser(TITOTokenizerType.QWEN3, user_reasoning_parser="glm45")

    def test_resolve_raises_on_tool_call_mismatch(self):
        from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

        with pytest.raises(ValueError, match="--tool-call-parser='glm47' disagrees"):
            resolve_reasoning_and_tool_call_parser(TITOTokenizerType.QWEN3, user_tool_call_parser="glm47")

    def test_resolve_accepts_user_value_when_family_unbound(self):
        # DEFAULT family has no binding for either parser; user-provided wins
        # (for families that haven't been wired up to a sglang parser yet).
        from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

        assert resolve_reasoning_and_tool_call_parser(
            TITOTokenizerType.DEFAULT, "custom_reasoning", "custom_tool_call"
        ) == ("custom_reasoning", "custom_tool_call")

    def test_resolve_partial_user_input(self):
        # User can pass only one of the two; the other auto-resolves from
        # the family binding independently.
        from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

        # User passes reasoning only — tool_call comes from binding.
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.QWEN3, user_reasoning_parser="qwen3") == (
            "qwen3",
            "qwen25",
        )
        # User passes tool_call only — reasoning comes from binding.
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.GLM47, user_tool_call_parser="glm47") == (
            "glm45",
            "glm47",
        )
