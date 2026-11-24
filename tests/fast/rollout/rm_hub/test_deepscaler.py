import pytest

from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward, get_gemma_math_reward
from miles.utils.reasoning import REASONING_DELIMITERS

_EXPECTED_DELIMITERS = (
    "<|open|>response<|sep|>",
    "<|channel|>final<|message|>",
    "<|content_text|>",
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
        assert tuple(delimiter for _, delimiter in REASONING_DELIMITERS.values()) == _EXPECTED_DELIMITERS

    @pytest.mark.parametrize("delimiter", _EXPECTED_DELIMITERS)
    def test_all_delimiters_grade_only_post_delimiter_answer(self, delimiter):
        assert get_deepscaler_rule_based_reward(rf"thought {delimiter} \boxed{{42}}", "42") == 1
        assert get_deepscaler_rule_based_reward(rf"thought {delimiter} \boxed{{wrong}}", "42") == 0
        # Correct box inside reasoning must not pass when the final answer is wrong.
        assert get_deepscaler_rule_based_reward(rf"\boxed{{42}} {delimiter} \boxed{{wrong}}", "42") == 0
        # Wrong box inside reasoning must not override a correct final answer.
        assert get_deepscaler_rule_based_reward(rf"\boxed{{wrong}} {delimiter} \boxed{{42}}", "42") == 1
        # Repeated delimiters grade only text after the last occurrence.
        assert get_deepscaler_rule_based_reward(rf"t1{delimiter}\boxed{{wrong}}t2{delimiter}\boxed{{42}}", "42") == 1
        assert get_deepscaler_rule_based_reward(rf"t1{delimiter}\boxed{{42}}t2{delimiter}\boxed{{wrong}}", "42") == 0

    @pytest.mark.parametrize(
        "response",
        [
            r"<|open|>think<|sep|>work \boxed{42}<|close|>think",
            r"<|channel|>analysis<|message|>work \boxed{42}<|end|>",
            r"<|content_thinking|>work \boxed{42}<|end_message|>",
            r" to=self<|message|>work \boxed{42}<|eom|>",
            r"<|thought>work \boxed{42}",
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
            (r"<|content_thinking|>###Response </think>\boxed{42}<|end_message|><|content_text|>\boxed{wrong}", 0),
            (r"<|content_thinking|>###Response </think>\boxed{wrong}<|end_message|><|content_text|>\boxed{42}", 1),
            (
                r"<|channel|>analysis<|message|></think>###Response\boxed{42}<|end|>"
                r"<|start|>assistant<|channel|>final<|message|>\boxed{wrong}",
                0,
            ),
            (
                r"<|channel|>analysis<|message|></think>###Response\boxed{wrong}<|end|>"
                r"<|start|>assistant<|channel|>final<|message|>\boxed{42}",
                1,
            ),
        ],
    )
    def test_model_special_delimiter_precedence_over_generic_markers(self, response, expected):
        assert get_deepscaler_rule_based_reward(response, "42") == expected


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
