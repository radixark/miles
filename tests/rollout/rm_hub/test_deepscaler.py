import pytest

from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward


class TestGetDeepscalerRuleBasedReward:
    def test_with_think_tag_correct(self):
        response = "Let me analyze...</think>The answer is \\boxed{42}"
        assert get_deepscaler_rule_based_reward(response, "42") == 1

    def test_with_think_tag_incorrect(self):
        response = "Thinking...</think>The answer is \\boxed{wrong}"
        assert get_deepscaler_rule_based_reward(response, "42") == 0

    def test_with_response_tag_correct(self):
        response = "###Response\\boxed{42}"
        assert get_deepscaler_rule_based_reward(response, "42") == 1

    def test_with_response_tag_incorrect(self):
        response = "###Response\\boxed{wrong}"
        assert get_deepscaler_rule_based_reward(response, "42") == 0

    def test_no_delimiter(self):
        response = "The answer is \\boxed{42}"
        assert get_deepscaler_rule_based_reward(response, "42") == 0

    def test_no_boxed_answer(self):
        response = "</think>The answer is 42"
        assert get_deepscaler_rule_based_reward(response, "42") == 0

    def test_empty_label(self):
        response = "</think>\\boxed{42}"
        assert get_deepscaler_rule_based_reward(response, "") == 0

    def test_boxed_label(self):
        response = "</think>\\boxed{42}"
        assert get_deepscaler_rule_based_reward(response, "\\boxed{42}") == 1

    def test_numeric_label(self):
        response = "</think>\\boxed{123}"
        assert get_deepscaler_rule_based_reward(response, 123) == 1

    def test_float_label(self):
        response = "</think>\\boxed{3.14}"
        assert get_deepscaler_rule_based_reward(response, 3.14) == 1

    def test_fraction_equivalence(self):
        response = "</think>\\boxed{1/2}"
        assert get_deepscaler_rule_based_reward(response, "0.5") == 1

    def test_latex_fraction(self):
        response = "</think>\\boxed{\\frac{1}{2}}"
        assert get_deepscaler_rule_based_reward(response, "0.5") == 1

    def test_multiple_think_tags(self):
        response = "First thought</think>Second thought</think>\\boxed{42}"
        assert get_deepscaler_rule_based_reward(response, "42") == 1
