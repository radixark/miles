import pytest

from miles.rollout.rm_hub.gpqa import (
    _extract_letter_from_response,
    _normalize_text,
    _strip_chain_of_thought,
    compute_gpqa_reward,
)

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


class TestStripChainOfThought:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Let me think...</think>The answer is A", "The answer is A"),
            ("The answer is A", "The answer is A"),
            ("<think>The answer is A", "<think>The answer is A"),
            ("The answer is A</think>", ""),
            ("", ""),
            (None, ""),
        ],
    )
    def test_strip_chain_of_thought(self, text, expected):
        assert _strip_chain_of_thought(text) == expected

    @pytest.mark.parametrize("marker", _ANSWER_MARKERS)
    def test_strip_all_reasoning_delimiters(self, marker):
        assert _strip_chain_of_thought(f"Answer: B {marker}The answer is A") == "The answer is A"
        assert _strip_chain_of_thought(f"Answer: A{marker}") == ""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("<think>###Response Answer: B</think>The answer is A", "The answer is A"),
            ("<think>quote <|content_text|> Answer: B</think>Answer: A", "Answer: A"),
            ("<think>quote <|message_model|><|content_text|> Answer: B</think>Answer: A", "Answer: A"),
            ("<think>quote <|channel|>final<|message|> Answer: B</think>Answer: A", "Answer: A"),
            (
                "<|content_thinking|>###Response </think>Answer: B<|end_message|>"
                "<|message_model|><|content_text|>The answer is A<|end_message|><|content_model_end_sampling|>",
                "The answer is A",
            ),
            (
                "<|channel|>analysis<|message|></think>###Response Answer: B<|end|>"
                "<|start|>assistant<|channel|>final<|message|>The answer is A<|return|>",
                "The answer is A",
            ),
            # Mid-prose end token does not truncate the answer; only the trailing turn-end token is stripped.
            (
                "to=user<|message|>The marker <|eot|> ends a turn. Answer: A.<|eot|>",
                "The marker <|eot|> ends a turn. Answer: A.",
            ),
            # Consecutive Inkling assistant turns grade only the final turn.
            (
                "<|content_text|>Answer: B<|end_message|><|content_model_end_sampling|>"
                "<|message_model|><|content_text|>Answer: A<|end_message|><|content_model_end_sampling|>",
                "Answer: A",
            ),
            # Multi-turn Harmony / Muse Spark traces whose final turn stops in reasoning do not extract Turn 1's answer.
            (
                "<|start|>assistant<|channel|>final<|message|>A<|end|>"
                "<|start|>user<|message|>next<|end|>"
                "<|start|>assistant<|channel|>analysis<|message|>work",
                "<|start|>assistant<|channel|>final<|message|>A<|end|>"
                "<|start|>user<|message|>next<|end|>"
                "<|start|>assistant<|channel|>analysis<|message|>work",
            ),
            (
                "to=user<|message|>A<|eot|><|start|>user<|message|>next<|eot|>"
                "<|start|>assistant to=self<|message|>work",
                "to=user<|message|>A<|eot|><|start|>user<|message|>next<|eot|>"
                "<|start|>assistant to=self<|message|>work",
            ),
            # Muse Spark subsequent thinking block after to=user<|eom|> stops at <|eom|><|start|>.
            (
                " to=user<|message|>A<|eom|><|start|>assistant to=self<|message|>Answer: B<|eom|>",
                "A",
            ),
            # Quoted Inkling / Muse Spark frames inside <think> never override the real </think>.
            (
                "<think>Example: <|message_model|><|content_text|>Answer: B<|end_message|>"
                "<|message_model|><|content_thinking|>work</think>Answer: A<|im_end|>",
                "Answer: A",
            ),
            (
                "<think>Example: to=user<|message|>Answer: B<|eot|></think>Answer: A. The marker is <|eot|>",
                "Answer: A. The marker is <|eot|>",
            ),
            (
                "Example: to=user<|message|>Answer: B<|eot|></think>Answer: A. The marker is <|eot|>",
                "Answer: A. The marker is <|eot|>",
            ),
            # Harmony tool calls grade the final channel.
            (
                '<|start|>assistant to=functions.calc<|channel|>commentary json<|message|>{"x":1}<|call|>'
                "<|start|>functions.calc to=assistant<|channel|>commentary<|message|>Answer: B<|end|>"
                "<|start|>assistant<|channel|>final<|message|>Answer: A<|return|>",
                "Answer: A",
            ),
            (
                ' to=functions.calc<|channel|>commentary json<|message|>{"x":1}<|call|>'
                "<|start|>functions.calc to=assistant<|channel|>commentary<|message|>Answer: B<|end|>"
                "<|start|>assistant<|channel|>final<|message|>Answer: A<|return|>",
                "Answer: A",
            ),
            # Inkling direct tool calls with empty or `_` recipient route through multi-block Inkling parsing.
            (
                '<|content_invoke_tool_json|>{"x":1}<|end_message|><|content_model_end_sampling|>'
                "<|message_tool|><|content_text|>ok<|end_message|>"
                "<|message_model|><|content_text|>Answer: A<|end_message|>"
                "<|message_model|><|content_text|>.<|end_message|><|content_model_end_sampling|>",
                "Answer: A.",
            ),
            (
                '_<|content_invoke_tool_json|>{"x":1}<|end_message|><|content_model_end_sampling|>'
                "<|message_tool|>_<|content_text|>ok<|end_message|>"
                "<|message_model|><|content_text|>Answer: A<|end_message|>"
                "<|message_model|><|content_text|>.<|end_message|><|content_model_end_sampling|>",
                "Answer: A.",
            ),
        ],
    )
    def test_rightmost_structural_delimiter_precedence(self, text, expected):
        assert _strip_chain_of_thought(text) == expected


class TestNormalizeText:
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("Hello World", "hello world"),
            ("Test-123", "test 123"),
            ("A, B, C", "a b c"),
            ("", ""),
        ],
    )
    def test_normalize_text(self, input_str, expected):
        assert _normalize_text(input_str) == expected


class TestExtractLetterFromResponse:
    @pytest.mark.parametrize(
        "response,expected",
        [
            ("The answer is A", "A"),
            ("answer: B", "B"),
            ("I think C is correct", "C"),
            ("final answer: D", "D"),
            ("Option A is the best choice", "A"),
            ("</think>The answer is B", "B"),
            ("After analysis, my choice is C", "C"),
            ("A B C D", None),
            ("The answer is definitely A", "A"),
            ("The answer is clearly B.", "B"),
            ("The answer is (B). A quick check confirms it.", "B"),
            ("The answer depends on the solvent; I pick B", "B"),
            ("Therefore, B.", "B"),
            ("Long reasoning about the options.\n\nC", "C"),
            ("The answer is not A, it is B", "B"),
            ("The final answer is not C", None),
            ("A or B", None),
            ("(A) (B) (C) (D)", None),
            ("A. B. C. D.", None),
            ("The answer is: B", "B"),
            ("The final answer is **B**", "B"),
            ("The answer is \\boxed{C}", "C"),
            ("The answer is $\\textbf{D}$.", "D"),
            ("\\text{C}", "C"),
            ("C. 42 eV", "C"),
            ("(C) 42 eV", "C"),
            ("Comparing options A and C, the answer is C.", "C"),
            ("The choices B and D both fit, but the answer is D", "D"),
            ("A. Incorrect because of sterics.\nB. Too slow.\nTherefore, C.", "C"),
            ("A. Incorrect because of sterics.\nB. Too slow.", None),
            ("\\[\\boxed{B}\\]", "B"),
            ("The answer is \\(\\boxed{B}\\)", "B"),
            ("The answer is *B*", "B"),
            ("The answer is `C`", "C"),
            ("A quick check shows nothing conclusive", None),
            ("(C)", "C"),
            ("No valid letter here", None),
            ("", None),
            (None, None),
            ("The answer is Z", None),
        ],
    )
    def test_extract_letter(self, response, expected):
        assert _extract_letter_from_response(response, "ABCD") == expected


class TestComputeGpqaReward:
    @pytest.mark.parametrize(
        "response,label,metadata,expected",
        [
            ("Answer: A", "A", None, 1.0),
            ("Answer: A", "B", None, 0.0),
            (None, "A", None, 0.0),
            ("Answer: B", "ignored", {"correct_letter": "B"}, 1.0),
            ("Answer: A", "ignored", {"correct_letter": "B"}, 0.0),
            ("Answer: A", 0, {"choices": ["Option 1", "Option 2", "Option 3", "Option 4"]}, 1.0),
            ("Answer: B", 1, {"choices": ["Option 1", "Option 2", "Option 3", "Option 4"]}, 1.0),
            ("Answer: X", "X", {"valid_letters": ["X", "Y", "Z"]}, 1.0),
            ("Answer: A", "X", {"valid_letters": ["X", "Y", "Z"]}, 0.0),
            (
                "I believe the answer is Paris",
                "",
                {"choices": ["Paris", "London", "Berlin", "Rome"], "correct_letter": "A"},
                1.0,
            ),
            ("Answer: A", "", {"choices": {"A": "Paris", "B": "London"}, "correct_letter": "A"}, 1.0),
            ("The answer is Paris", "Paris", {"choices": ["Paris", "London", "Berlin", "Rome"]}, 1.0),
            ("Let me think step by step...</think>The answer is A", "A", None, 1.0),
            ("The answer is definitely A", "A", None, 1.0),
        ],
    )
    def test_compute_gpqa_reward(self, response, label, metadata, expected):
        assert compute_gpqa_reward(response, label, metadata=metadata) == expected

    @pytest.mark.parametrize("label", ["A", "B", "C", "D"])
    def test_no_committed_letter_scores_zero(self, label):
        assert compute_gpqa_reward("abcd", label) == 0.0
        assert (
            compute_gpqa_reward("The product depends on which hydride shifts; both pathways look plausible.", label)
            == 0.0
        )

    @pytest.mark.parametrize(
        "response,label,metadata,expected",
        [
            (
                "the candidates are paris, london, rome and berlin.",
                0,
                {"choices": ["Paris", "London", "Rome", "Berlin"]},
                0.0,
            ),
            (
                "the candidates are paris, london, rome and berlin.",
                3,
                {"choices": ["Paris", "London", "Rome", "Berlin"]},
                0.0,
            ),
            ("It is not Rome, it is Paris", 0, {"choices": ["Paris", "London", "Rome", "Berlin"]}, 0.0),
            ("Parisian food is great, the city is Berlin", 3, {"choices": ["Paris", "London", "Rome", "Berlin"]}, 1.0),
            (
                "The product is (R)-2-butanol.",
                1,
                {"choices": ["2-butanol", "(R)-2-butanol", "1-butanol", "butanal"]},
                1.0,
            ),
            (
                "Either 2-butanol or (R)-2-butanol.",
                1,
                {"choices": ["2-butanol", "(R)-2-butanol", "1-butanol", "butanal"]},
                0.0,
            ),
            ("The answer is 4", "4", None, 1.0),
            ("The answer is 14", "4", None, 0.0),
        ],
    )
    def test_listing_several_choices_is_a_hedge(self, response, label, metadata, expected):
        assert compute_gpqa_reward(response, label, metadata=metadata) == expected

    @pytest.mark.parametrize("marker", _ANSWER_MARKERS)
    def test_reasoning_delimiters_ignore_distractor_in_thought(self, marker):
        response = f"First guess: Answer: B.{marker}The answer is A."
        assert compute_gpqa_reward(response, "A") == 1.0
        assert compute_gpqa_reward(response, "B") == 0.0

    @pytest.mark.parametrize(
        "response",
        [
            "<|content_thinking|>Answer: B<|end_message|><|message_model|><|content_text|>A<|end_message|><|content_model_end_sampling|>",
            "<|content_text|>A<|end_message|><|content_model_end_sampling|>",
            " to=self<|message|>Answer: B<|eom|><|start|>assistant to=user<|message|>A<|eot|>",
            "<|channel|>analysis<|message|>Answer: B<|end|><|start|>assistant<|channel|>final<|message|>A<|return|>",
            "<|open|>think<|sep|>Answer: B<|close|>think<|sep|><|open|>response<|sep|>A<|close|>response<|sep|><|close|>message<|sep|>",
            "<|channel>thought\nAnswer: B<channel|>A<turn|>",
            "<think>\nAnswer: B\n</think>\n\nA<|im_end|>",
            "<think>\nAnswer: B\n</think>\n\nA<｜end▁of▁sentence｜>",
            "<think>\nAnswer: B\n</think>\n\nA[e~[",
            "to=user<|message|>The marker <|eot|> ends a turn. Answer: A.<|eot|>",
            "<|content_text|>A.<|end_message|><|message_model|><|content_text|> 42 eV<|end_message|><|content_model_end_sampling|>",
        ],
    )
    def test_framed_bare_letter_with_end_tokens_scores_one(self, response):
        assert compute_gpqa_reward(response, "A") == 1.0
        assert compute_gpqa_reward(response, "B") == 0.0

    def test_unclosed_reasoning_stays_permissive(self):
        assert compute_gpqa_reward("<think>The answer is A", "A") == 1.0
        assert compute_gpqa_reward("<|content_thinking|>The answer is A", "A") == 1.0
        # Inkling tool observations carrying <|content_text|> must not trigger splitting when the final turn is unclosed.
        tool_then_unclosed = (
            "<|message_tool|>search<|content_text|>irrelevant<|end_message|>"
            "<|message_model|><|content_thinking|>The answer is A"
        )
        assert compute_gpqa_reward(tool_then_unclosed, "A") == 1.0
        # Unclosed final Harmony / Muse Spark turn must not grade an earlier turn's bare-letter answer.
        harmony_stale = (
            "<|start|>assistant<|channel|>final<|message|>A<|end|>"
            "<|start|>user<|message|>next<|end|>"
            "<|start|>assistant<|channel|>analysis<|message|>work"
        )
        muse_stale = (
            "to=user<|message|>A<|eot|><|start|>user<|message|>next<|eot|><|start|>assistant to=self<|message|>work"
        )
        assert compute_gpqa_reward(harmony_stale, "A") == 0.0
        assert compute_gpqa_reward(muse_stale, "A") == 0.0
