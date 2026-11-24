import pytest

from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward


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
            (r"draft \boxed{7}<|content_text|>The answer is \boxed{42}", "42", 1),
            (r"draft \boxed{42}<|content_text|>The answer is \boxed{7}", "42", 0),
        ],
    )
    def test_get_deepscaler_rule_based_reward(self, response, label, expected):
        assert get_deepscaler_rule_based_reward(response, label) == expected
