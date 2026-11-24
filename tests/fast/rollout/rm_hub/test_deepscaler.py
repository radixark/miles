import pytest

from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward, get_gemma_math_reward
from miles.utils.reasoning import REASONING_DELIMITERS

_EXPECTED_DELIMITERS = {
    "kimi_k3": ("<|open|>think<|sep|>", "<|open|>response<|sep|>"),
    "harmony": ("<|channel|>analysis<|message|>", "<|channel|>final<|message|>"),
    "inkling": ("<|content_thinking|>", "<|content_text|>"),
    "muse_spark": ("to=self<|message|>", "to=user<|message|>"),
    "gemma4": ("<|channel>", "<channel|>"),
    "kimi_vl": ("◁think▷", "◁/think▷"),
    "think": ("<think>", "</think>"),
    "legacy_response": (None, "###Response"),
}

# Wire markers that open a model answer segment in each family's decoded response.
_ANSWER_MARKERS = (
    "<|open|>response<|sep|>",
    "<|channel|>final<|message|>",
    "<|message_model|><|content_text|>",
    "to=user<|message|>",
    "<channel|>",
    "◁/think▷",
    "</think>",
    "###Response",
)


class TestGetDeepscalerRuleBasedReward:
    @pytest.mark.parametrize(
        "response,label,expected",
        [
            (r"Let me analyze...</think>The answer is \boxed{42}", "42", 1),
            (r"Thinking...</think>The answer is \boxed{wrong}", "42", 0),
            (r"###Response\boxed{42}", "42", 1),
            (r"###Response\boxed{wrong}", "42", 0),
            (r"The answer is \boxed{42}", "42", 0),
            (r"</think>The answer is 42", "42", 0),
            (r"</think>\boxed{42}", "", 0),
            (r"</think>\boxed{42}", r"\boxed{42}", 1),
            (r"</think>\fbox{42}", r"Therefore, $\fbox{42}$.", 1),
            (r"</think>\boxed{123}", 123, 1),
            (r"</think>\boxed{3.14}", 3.14, 1),
            (r"</think>\boxed{1/2}", "0.5", 1),
            (r"</think>\boxed{\frac{1}{2}}", "0.5", 1),
            (r"First thought</think>Second thought</think>\boxed{42}", "42", 1),
        ],
    )
    def test_get_deepscaler_rule_based_reward(self, response, label, expected):
        assert get_deepscaler_rule_based_reward(response, label) == expected

    def test_expected_delimiters_cover_ssot(self):
        assert REASONING_DELIMITERS == _EXPECTED_DELIMITERS

    @pytest.mark.parametrize("marker", _ANSWER_MARKERS)
    def test_all_delimiters_grade_only_post_delimiter_answer(self, marker):
        assert get_deepscaler_rule_based_reward(rf"thought {marker} \boxed{{42}}", "42") == 1
        assert get_deepscaler_rule_based_reward(rf"thought {marker} \boxed{{wrong}}", "42") == 0
        # Correct box before the boundary with no box after must score 0 (proves slicing ran).
        assert get_deepscaler_rule_based_reward(rf"\boxed{{42}} {marker} no final box", "42") == 0
        assert get_deepscaler_rule_based_reward(rf"\boxed{{42}} {marker} \boxed{{wrong}}", "42") == 0
        assert get_deepscaler_rule_based_reward(rf"\boxed{{wrong}} {marker} \boxed{{42}}", "42") == 1
        # Repeated markers grade only the final answer segment.
        assert get_deepscaler_rule_based_reward(rf"t1{marker}\boxed{{wrong}}t2{marker}\boxed{{42}}", "42") == 1
        assert get_deepscaler_rule_based_reward(rf"t1{marker}\boxed{{42}}t2{marker}\boxed{{wrong}}", "42") == 0

    def test_inkling_multi_block_and_tool_observations(self):
        # Direct text turn starting at index 0 (right after prompt's <|message_model|>).
        assert (
            get_deepscaler_rule_based_reward(
                r"<|content_text|>\boxed{42}<|end_message|><|content_model_end_sampling|>", "42"
            )
            == 1
        )
        # A thinking block emitted after a text block must not leak into the graded answer.
        text_then_think = (
            r"<|content_text|>\boxed{1}<|end_message|>"
            r"<|message_model|><|content_thinking|>\boxed{42}<|end_message|><|content_model_end_sampling|>"
        )
        assert get_deepscaler_rule_based_reward(text_then_think, "1") == 1
        assert get_deepscaler_rule_based_reward(text_then_think, "42") == 0
        # Multiple text blocks in the final turn concatenate with "" (matching InklingResponseParser).
        two_text_blocks = (
            r"<|content_thinking|>work<|end_message|>"
            r"<|message_model|><|content_text|>\boxed{42}<|end_message|>"
            r"<|message_model|><|content_text|> units<|end_message|><|content_model_end_sampling|>"
        )
        assert get_deepscaler_rule_based_reward(two_text_blocks, "42") == 1
        split_boxed_blocks = (
            r"<|content_text|>\box<|end_message|>"
            r"<|message_model|><|content_text|>ed{42}<|end_message|><|content_model_end_sampling|>"
        )
        assert get_deepscaler_rule_based_reward(split_boxed_blocks, "42") == 1
        # Consecutive assistant turns separated by <|content_model_end_sampling|> grade only the final turn.
        two_turns_no_final_box = (
            r"<|content_text|>\boxed{42}<|end_message|><|content_model_end_sampling|>"
            r"<|message_model|><|content_text|>no final box<|end_message|><|content_model_end_sampling|>"
        )
        assert get_deepscaler_rule_based_reward(two_turns_no_final_box, "42") == 0
        two_turns_unclosed_think = (
            r"<|content_text|>\boxed{42}<|end_message|><|content_model_end_sampling|>"
            r"<|message_model|><|content_thinking|>still thinking"
        )
        assert get_deepscaler_rule_based_reward(two_turns_unclosed_think, "42") == 0
        # Empty first Inkling turn (<|content_model_end_sampling|>) followed by prior text and unclosed final turn.
        empty_turn1_prior_text = (
            r"<|content_model_end_sampling|><|message_model|><|content_text|>\boxed{7}<|end_message|>"
            r"<|content_model_end_sampling|><|message_model|><|content_thinking|>still thinking"
        )
        assert get_deepscaler_rule_based_reward(empty_turn1_prior_text, "7") == 0
        # Direct Inkling tool call right after prompt's <|message_model|> routes through Inkling parsing.
        direct_tool_call_then_answer = (
            r'calc<|content_invoke_tool_json|>{"x":1}<|end_message|><|content_model_end_sampling|>'
            r"<|message_tool|>calc<|content_text|>\boxed{wrong}<|end_message|>"
            r"<|message_model|><|content_text|>\boxed{42}<|end_message|><|content_model_end_sampling|>"
        )
        assert get_deepscaler_rule_based_reward(direct_tool_call_then_answer, "42") == 1
        # Mid-prose mention of <|message_tool|> inside model text is not a framed tool role.
        prose_tool_token = (
            r"<|content_text|>Inkling uses <|message_tool|> for tools. \boxed{42}"
            r"<|end_message|><|content_model_end_sampling|>"
        )
        assert get_deepscaler_rule_based_reward(prose_tool_token, "42") == 1
        # Tool observation carrying <|content_text|> must not count as a model answer when final turn stops in thinking.
        tool_then_unclosed_think = (
            r"<|content_thinking|>call tool<|end_message|>"
            r"<|message_model|>calc<|content_invoke_tool_json|>{}<|end_message|><|content_model_end_sampling|>"
            r"<|message_tool|>calc<|content_text|>\boxed{42}<|end_message|>"
            r"<|message_model|><|content_thinking|>still thinking \boxed{42}"
        )
        assert get_deepscaler_rule_based_reward(tool_then_unclosed_think, "42") == 0
        # Tool observation followed by a real model answer grades the final model turn.
        tool_then_answer = (
            r"<|message_tool|>calc<|content_text|>\boxed{wrong}<|end_message|>"
            r"<|message_model|><|content_thinking|>check<|end_message|>"
            r"<|message_model|><|content_text|>\boxed{42}<|end_message|><|content_model_end_sampling|>"
        )
        assert get_deepscaler_rule_based_reward(tool_then_answer, "42") == 1

    @pytest.mark.parametrize(
        "response",
        [
            r"<|open|>think<|sep|>work \boxed{42}<|close|>think<|sep|>",
            r"<|channel|>analysis<|message|>work \boxed{42}<|end|>",
            r"<|content_thinking|>work \boxed{42}<|end_message|>",
            r" to=self<|message|>work \boxed{42}<|eom|>",
            "<|channel>thought\n" + r"work \boxed{42}",
            r"◁think▷work \boxed{42}",
            r"<think>work \boxed{42}",
            "",
            None,
        ],
    )
    def test_unclosed_reasoning_returns_zero(self, response):
        assert get_deepscaler_rule_based_reward(response, "42") == 0

    @pytest.mark.parametrize(
        "response,expected",
        [
            (r"<think>###Response \boxed{wrong}</think>\boxed{42}", 1),
            (r"<think>###Response \boxed{42}</think>\boxed{wrong}", 0),
            # Foreign markers quoted inside <think> must not shadow the real </think> closer.
            (r"<think>quote <|content_text|> \boxed{42}</think>no final box", 0),
            (r"<think>quote <|message_model|><|content_text|> \boxed{42}</think>no final box", 0),
            (r"<think>quote <|message_model|><|content_text|> \boxed{42}<|end_message|></think>no final box", 0),
            (r"<think>quote <|channel|>final<|message|> \boxed{42}</think>no final box", 0),
            (r"<think>quote to=user<|message|> \boxed{42}<|eot|></think>no final box", 0),
            (
                r"<think>Example: <|message_model|><|content_text|>\boxed{wrong}<|end_message|>"
                r"<|message_model|><|content_thinking|>work</think>\boxed{42}<|im_end|>",
                1,
            ),
            (r"<think>Example: to=user<|message|>\boxed{wrong}<|eot|></think>\boxed{42}. The marker is <|eot|>", 1),
            (r"Example: to=user<|message|>\boxed{wrong}<|eot|></think>\boxed{42}. The marker is <|eot|>", 1),
            # Harmony tool call (<|start|>assistant to=...<|channel|>...) grades the final channel.
            (
                r'<|start|>assistant to=functions.calc<|channel|>commentary json<|message|>{"x":1}<|call|>'
                r"<|start|>functions.calc to=assistant<|channel|>commentary<|message|>\boxed{wrong}<|end|>"
                r"<|start|>assistant<|channel|>final<|message|>\boxed{42}<|return|>",
                1,
            ),
            (
                r' to=functions.calc<|channel|>commentary json<|message|>{"x":1}<|call|>'
                r"<|start|>functions.calc to=assistant<|channel|>commentary<|message|>\boxed{wrong}<|end|>"
                r"<|start|>assistant<|channel|>final<|message|>\boxed{42}<|return|>",
                1,
            ),
            # Multi-turn Harmony / Muse Spark traces must scope to the final turn so an unclosed turn never rewards Turn 1.
            (
                r"<|start|>assistant<|channel|>final<|message|>\boxed{42}<|end|>"
                r"<|start|>user<|message|>next<|end|>"
                r"<|start|>assistant<|channel|>analysis<|message|>\boxed{7}",
                0,
            ),
            (
                r"<|start|>assistant<|channel|>final<|message|>\boxed{42}<|return|>"
                r"<|start|>user<|message|>next<|end|>"
                r"<|start|>assistant<|channel|>analysis<|message|>\boxed{7}",
                0,
            ),
            (
                r"<|start|>assistant<|channel|>final<|message|>\boxed{42}<|end|>"
                r"<|start|>functions.calc<|message|>ok<|end|>"
                r"<|start|>assistant<|channel|>analysis<|message|>\boxed{7}",
                0,
            ),
            (
                r"<|start|>assistant<|channel|>final<|message|>\boxed{7}<|end|>"
                r"<|start|>user<|message|>next<|end|>"
                r"<|start|>assistant<|channel|>analysis<|message|>think<|end|>"
                r"<|start|>assistant<|channel|>final<|message|>\boxed{42}<|return|>",
                1,
            ),
            (
                r"to=user<|message|>\boxed{42}<|eot|><|start|>user<|message|>next<|eot|>"
                r"<|start|>assistant to=self<|message|>work",
                0,
            ),
            (
                r"to=user<|message|>\boxed{7}<|eot|><|start|>user<|message|>next<|eot|>"
                r"<|start|>assistant to=self<|message|>think<|eom|>"
                r"<|start|>assistant to=user<|message|>\boxed{42}<|eot|>",
                1,
            ),
            # Muse Spark subsequent thinking block after to=user<|eom|> must not leak into answer, while mid-prose <|eot|> stays intact.
            (r" to=user<|message|>\boxed{1}<|eom|><|start|>assistant to=self<|message|>\boxed{42}<|eom|>", 0),
            (r"to=user<|message|>The marker <|eot|> ends a turn. \boxed{42}<|eot|>", 1),
            (
                r"<|content_thinking|>###Response </think>\boxed{42}<|end_message|>"
                r"<|message_model|><|content_text|>\boxed{wrong}<|end_message|>",
                0,
            ),
            (
                r"<|content_thinking|>###Response </think>\boxed{wrong}<|end_message|>"
                r"<|message_model|><|content_text|>\boxed{42}<|end_message|>",
                1,
            ),
            (
                r"<|channel|>analysis<|message|></think>###Response\boxed{42}<|end|>"
                r"<|start|>assistant<|channel|>final<|message|>\boxed{wrong}<|return|>",
                0,
            ),
            (
                r"<|channel|>analysis<|message|></think>###Response\boxed{wrong}<|end|>"
                r"<|start|>assistant<|channel|>final<|message|>\boxed{42}<|return|>",
                1,
            ),
        ],
    )
    def test_rightmost_structural_delimiter_precedence(self, response, expected):
        assert get_deepscaler_rule_based_reward(response, "42") == expected

    @pytest.mark.parametrize("family", ["kimi_k3", "harmony", "muse_spark", "gemma4", "think"])
    def test_trailing_whitespace_run_is_linear(self, family):
        _, delimiter = REASONING_DELIMITERS[family]
        response = f"{delimiter}\\boxed{{42}}" + " " * 30_000
        assert get_deepscaler_rule_based_reward(response, "42") == 1


class TestGetGemmaMathReward:
    @pytest.mark.parametrize(
        "response,label,expected",
        [
            (r"thinking<channel|>The answer is \boxed{42}", "42", 1),
            (r"thinking<channel|>The answer is \boxed{wrong}", "42", 0),
            (r"\boxed{42} thinking<channel|>no box here", "42", 0),
            (r"first<channel|>\boxed{wrong}second<channel|>\boxed{42}", "42", 1),
            # Permissive fallback when <channel|> is absent stays compatible with non-thinking outputs.
            (r"The answer is \boxed{42}", "42", 1),
            (r"The answer is \boxed{wrong}", "42", 0),
        ],
    )
    def test_get_gemma_math_reward(self, response, label, expected):
        assert get_gemma_math_reward(response, label) == expected
