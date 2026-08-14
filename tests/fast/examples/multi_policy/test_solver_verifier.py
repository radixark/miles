from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from examples.multi_policy import solver_verifier
from examples.multi_policy.solver_verifier import _Verdict
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample

SOLVER_URL = "http://solver-host:1111/generate"
VERIFIER_URL = "http://verifier-host:2222/generate"


@dataclass
class _FakeGenerate:
    responses: dict[str, str]
    aborted_urls: frozenset[str] = frozenset()
    calls: list[tuple[str, Sample]] = field(default_factory=list)

    async def __call__(self, input: GenerateFnInput, url: str | None = None) -> GenerateFnOutput:
        sample = input.sample
        self.calls.append((url, sample))
        sample.response = self.responses[url]
        sample.status = Sample.Status.ABORTED if url in self.aborted_urls else Sample.Status.COMPLETED
        return GenerateFnOutput(samples=sample)


def _make_input(*, prompt: str | list[dict[str, str]], label: str) -> GenerateFnInput:
    args = Namespace(
        megatron_config=encode_megatron_config("solver", "verifier"),
        use_critic=False,
        sglang_model_routers={"solver": ("solver-host", 1111), "verifier": ("verifier-host", 2222)},
    )
    sample = Sample(group_index=3, index=7, prompt=prompt, label=label)
    return GenerateFnInput(state=SimpleNamespace(args=args), sample=sample, sampling_params={}, evaluation=False)


@dataclass(frozen=True)
class _RunResult:
    fake: _FakeGenerate
    samples: list[Sample]


async def _run(monkeypatch, *, solver_response: str, verifier_response: str, abort_solver: bool = False) -> _RunResult:
    fake = _FakeGenerate(
        responses={SOLVER_URL: solver_response, VERIFIER_URL: verifier_response},
        aborted_urls=frozenset({SOLVER_URL}) if abort_solver else frozenset(),
    )
    monkeypatch.setattr(solver_verifier, "single_turn_generate", fake)
    output = await solver_verifier.generate(
        _make_input(prompt=[dict(role="user", content="What is 9 + 9?")], label="#### 18")
    )
    return _RunResult(fake=fake, samples=output.samples)


class TestComputeVerifierReward:
    @pytest.mark.parametrize("verifier_correct", [False, True])
    def test_agreeing_with_a_right_solver_is_the_only_full_credit_case(self, verifier_correct):
        """The solver was right and the verifier said so, so its own answer never enters the score."""
        assert solver_verifier._compute_verifier_reward(
            solver_correct=True, verdict=_Verdict.AGREE, verifier_correct=verifier_correct
        ) == 1.0

    @pytest.mark.parametrize("verifier_correct", [False, True])
    def test_calling_a_right_solver_wrong_scores_zero(self, verifier_correct):
        """A false accusation is worthless however good the verifier's replacement answer is."""
        assert solver_verifier._compute_verifier_reward(
            solver_correct=True, verdict=_Verdict.WRONG, verifier_correct=verifier_correct
        ) == 0.0

    @pytest.mark.parametrize("verifier_correct", [False, True])
    def test_agreeing_with_a_wrong_solver_scores_zero(self, verifier_correct):
        """Endorsing a wrong solution is the failure the verifier exists to avoid."""
        assert solver_verifier._compute_verifier_reward(
            solver_correct=False, verdict=_Verdict.AGREE, verifier_correct=verifier_correct
        ) == 0.0

    def test_catching_a_wrong_solver_without_fixing_it_scores_half(self):
        """Spotting the error is worth partial credit even when the replacement answer is wrong."""
        assert solver_verifier._compute_verifier_reward(
            solver_correct=False, verdict=_Verdict.WRONG, verifier_correct=False
        ) == 0.5

    def test_catching_a_wrong_solver_and_fixing_it_scores_full(self):
        """Both halves of the verifier's job were done."""
        assert solver_verifier._compute_verifier_reward(
            solver_correct=False, verdict=_Verdict.WRONG, verifier_correct=True
        ) == 1.0

    @pytest.mark.parametrize("solver_correct", [False, True])
    @pytest.mark.parametrize("verifier_correct", [False, True])
    def test_an_unparseable_verdict_scores_zero(self, solver_correct, verifier_correct):
        """A verdict nobody can read teaches the solver nothing, whatever the verifier meant."""
        assert solver_verifier._compute_verifier_reward(
            solver_correct=solver_correct, verdict=None, verifier_correct=verifier_correct
        ) == 0.0


class TestParseVerdict:
    def test_the_verdict_line_the_prompt_asks_for_is_read(self):
        """The happy path the verifier prompt asks for."""
        assert solver_verifier._parse_verdict("The arithmetic checks out.\nVERDICT: AGREE") is _Verdict.AGREE

    def test_a_marker_inside_the_reasoning_is_not_a_verdict(self):
        """Otherwise 'I cannot decide: AGREE or WRONG' scores as if the verifier had ruled."""
        assert solver_verifier._parse_verdict("I would AGREE, but the sum is off.\nWRONG\n#### 18") is None

    def test_two_verdict_lines_are_unparseable(self):
        """A reply that rules twice never made up its mind, and picking one of them invents a decision."""
        response = "VERDICT: AGREE\nOn reflection:\nVERDICT: WRONG"

        assert solver_verifier._parse_verdict(response) is None

    def test_a_lowercase_verdict_is_not_a_verdict(self):
        """Strict markers keep prose such as 'I agree with the setup' from scoring."""
        assert solver_verifier._parse_verdict("verdict: agree") is None

    def test_a_longer_word_containing_the_marker_is_not_a_verdict(self):
        """The verdict line holds the marker alone, so 'AGREEMENT' cannot be read as one."""
        assert solver_verifier._parse_verdict("VERDICT: AGREEMENT") is None

    def test_a_response_without_any_marker_is_unparseable(self):
        """An empty or rambling reply has no verdict to score."""
        assert solver_verifier._parse_verdict("") is None

    def test_the_wrong_verdict_line_is_read(self):
        """The other half of the protocol, and the only path that lets the verifier earn credit on a bad solution."""
        assert solver_verifier._parse_verdict("The sum is off.\nVERDICT: WRONG\n#### 18") is _Verdict.WRONG

    @pytest.mark.parametrize("response", ["VERDICT:AGREE", "VERDICT:   AGREE   "])
    def test_spacing_around_the_marker_is_tolerated(self, response):
        """Models are inconsistent about the space after the colon, and that is not a decision."""
        assert solver_verifier._parse_verdict(response) is _Verdict.AGREE

    def test_an_indented_verdict_line_is_not_a_verdict(self):
        """The prompt asks for a line of its own, so an indented one is quoted text rather than a ruling."""
        assert solver_verifier._parse_verdict("Example:\n    VERDICT: AGREE") is None

    def test_trailing_text_on_the_verdict_line_is_not_a_verdict(self):
        """'VERDICT: AGREE with reservations' is a sentence, and reading it as a ruling invents certainty."""
        assert solver_verifier._parse_verdict("VERDICT: AGREE with reservations") is None

    def test_the_same_verdict_twice_is_still_unparseable(self):
        """Counting lines, not distinct values, keeps a repeated ruling from looking more decided than it is."""
        assert solver_verifier._parse_verdict("VERDICT: AGREE\nVERDICT: AGREE") is None


class TestExtractAnswer:
    def test_the_gsm8k_marker_is_read(self):
        """Both the dataset label and the prompted reply end with '#### <answer>'."""
        assert solver_verifier._extract_answer("Half of 36 is 18.\n#### 18") == "18"

    def test_the_last_marker_wins(self):
        """A reply that reconsiders itself is scored on its final answer."""
        assert solver_verifier._extract_answer("#### 17\nOn reflection:\n#### 18") == "18"

    def test_a_marked_answer_is_normalized(self):
        """Currency, thousand separators and a trailing period are formatting, not the answer."""
        assert solver_verifier._extract_answer("#### $1,234.") == "1234"

    def test_a_reply_without_the_marker_falls_back_to_its_last_number(self):
        """The solver prompt comes from the dataset, so it need not ask for the marker."""
        assert solver_verifier._extract_answer("First 9, then 9, so the total is 18") == "18"

    def test_a_reply_without_any_number_has_no_answer(self):
        """Nothing to compare against the ground truth."""
        assert solver_verifier._extract_answer("I cannot tell") is None

    def test_a_negative_marked_answer_keeps_its_sign(self):
        """gsm8k answers can be negative, and dropping the sign would score a wrong reply as right."""
        assert solver_verifier._extract_answer("#### -5") == "-5"

    def test_the_fallback_reads_the_last_number_not_the_first(self):
        """An unmarked reply states its result last, after restating the operands."""
        assert solver_verifier._extract_answer("From 20 we subtract 2, so 18") == "18"

    def test_the_marker_wins_over_a_later_bare_number(self):
        """Trailing prose such as a page reference must not overwrite the answer the model marked."""
        assert solver_verifier._extract_answer("#### 18\nsee step 3") == "18"


class TestIsCorrect:
    def test_a_label_without_an_answer_never_matches(self):
        """A dataset row with no parseable answer must fail closed rather than reward everything."""
        assert not solver_verifier._is_correct("#### 18", ground_truth=None)

    def test_both_sides_are_normalized_before_comparing(self):
        """The label and the reply format the same number differently, and formatting is not a wrong answer."""
        assert solver_verifier._is_correct("#### $1,234.", ground_truth="1234")


class TestBuildVerifierSample:
    def test_the_verifier_sample_keeps_the_identity_of_the_solver_sample(self):
        """Both samples belong to one trajectory, and the group is what advantage is computed over."""
        solver = Sample(group_index=3, index=7, rollout_id=2, prompt=[dict(role="user", content="q")], label="#### 18")
        solver.response = "#### 18"

        verifier = solver_verifier._build_verifier_sample(solver)

        assert (verifier.group_index, verifier.index, verifier.rollout_id) == (3, 7, 2)
        assert verifier.label == "#### 18"

    def test_the_metadata_is_copied_rather_than_shared(self):
        """The two samples are scored and dumped separately, so one must not write into the other's metadata."""
        solver = Sample(prompt=[dict(role="user", content="q")], metadata={"source": "gsm8k"})
        solver.response = "#### 18"

        verifier = solver_verifier._build_verifier_sample(solver)
        verifier.metadata["source"] = "mutated"

        assert solver.metadata == {"source": "gsm8k"}

    def test_the_routing_key_is_carried_over(self):
        """Consistent hashing routes both samples of a trajectory to the same engine."""
        solver = Sample(prompt=[dict(role="user", content="q")], routing_key="key-1")
        solver.response = "#### 18"

        assert solver_verifier._build_verifier_sample(solver).routing_key == "key-1"


class TestExtractQuestion:
    def test_a_system_message_is_not_quoted_as_the_question(self):
        """Only the user turn holds the problem; quoting the system prompt would confuse the verifier."""
        prompt = [dict(role="system", content="Be brief."), dict(role="user", content="What is 9 + 9?")]

        assert solver_verifier._extract_question(prompt) == "What is 9 + 9?"

    def test_several_user_messages_are_refused(self):
        """This example assumes one question per sample, and silently picking one would mis-state the task."""
        prompt = [dict(role="user", content="a"), dict(role="user", content="b")]

        with pytest.raises(ValueError):
            solver_verifier._extract_question(prompt)


class TestComputeRouterUrl:
    def test_the_url_points_at_the_router_of_that_policy(self):
        """Nothing else routes by model id, so a wrong url would silently train one policy on the other's tokens."""
        args = Namespace(sglang_model_routers={"solver": ("solver-host", 1111)})

        assert solver_verifier._compute_router_url(args, model_id="solver") == SOLVER_URL


class TestGenerate:
    async def test_the_verifier_prompt_quotes_the_question_and_the_solver_answer(self, monkeypatch):
        """The verifier only sees the solver's work through the prompt this function assembles."""
        result = await _run(monkeypatch, solver_response="It is 18.\n#### 18", verifier_response="VERDICT: AGREE")

        verifier_prompt = result.fake.calls[1][1].prompt
        assert isinstance(verifier_prompt, list)
        assert verifier_prompt[0]["role"] == "user"
        assert "What is 9 + 9?" in verifier_prompt[0]["content"]
        assert "It is 18.\n#### 18" in verifier_prompt[0]["content"]

    async def test_a_raw_string_prompt_is_refused(self, monkeypatch):
        """A string prompt may already be chat templated, so quoting it as the question would leak tokens."""
        fake = _FakeGenerate(responses={SOLVER_URL: "#### 18", VERIFIER_URL: "VERDICT: AGREE"})
        monkeypatch.setattr(solver_verifier, "single_turn_generate", fake)

        with pytest.raises(AssertionError, match="chat templated"):
            await solver_verifier.generate(_make_input(prompt="What is 9 + 9?", label="#### 18"))

    async def test_each_policy_is_generated_against_its_own_router(self, monkeypatch):
        """Nothing routes by trainer_model_id, so the generate function picks the url itself."""
        result = await _run(monkeypatch, solver_response="#### 18", verifier_response="VERDICT: AGREE")

        assert [url for url, _ in result.fake.calls] == [SOLVER_URL, VERIFIER_URL]

    async def test_both_samples_are_returned_bound_to_their_own_policy(self, monkeypatch):
        """trainer_model_id is filled on return, and it is what sends each sample to its trainer."""
        result = await _run(monkeypatch, solver_response="#### 18", verifier_response="VERDICT: AGREE")

        solver_sample, verifier_sample = result.samples
        assert solver_sample.trainer_model_id == "solver"
        assert verifier_sample.trainer_model_id == "verifier"

    async def test_a_right_solver_endorsed_by_the_verifier_rewards_both(self, monkeypatch):
        """The end to end path of the full credit row of the reward matrix."""
        result = await _run(monkeypatch, solver_response="#### 18", verifier_response="Checks out.\nVERDICT: AGREE")

        solver_sample, verifier_sample = result.samples
        assert solver_sample.reward == 1.0
        assert verifier_sample.reward == 1.0

    async def test_a_wrong_solver_corrected_by_the_verifier_rewards_only_the_verifier(self, monkeypatch):
        """The solver is scored against the label, the verifier against what it did about the solver."""
        result = await _run(monkeypatch, solver_response="#### 17", verifier_response="VERDICT: WRONG\n#### 18")

        solver_sample, verifier_sample = result.samples
        assert solver_sample.reward == 0.0
        assert verifier_sample.reward == 1.0

    async def test_a_wrong_solver_caught_but_not_fixed_rewards_half(self, monkeypatch):
        """The verifier's own answer is graded against the same ground truth."""
        result = await _run(monkeypatch, solver_response="#### 17", verifier_response="VERDICT: WRONG\n#### 16")

        assert result.samples[1].reward == 0.5

    async def test_a_run_naming_one_policy_is_refused(self, monkeypatch):
        """This example needs a solver and a verifier, and it must not silently train one of them twice."""
        fake = _FakeGenerate(responses={})
        monkeypatch.setattr(solver_verifier, "single_turn_generate", fake)
        input = _make_input(prompt=[dict(role="user", content="What is 9 + 9?")], label="#### 18")
        input.args.megatron_config = encode_megatron_config("solver")

        with pytest.raises(AssertionError, match="pairs one solver policy with one verifier policy"):
            await solver_verifier.generate(input)


class TestAbortedSolver:
    async def test_an_aborted_solver_is_not_handed_to_the_verifier(self, monkeypatch):
        """Judging a truncated attempt trains the verifier on work the solver never finished."""
        result = await _run(
            monkeypatch, solver_response="#### 18", verifier_response="VERDICT: AGREE", abort_solver=True
        )

        assert [url for url, _ in result.fake.calls] == [SOLVER_URL]
        [solver_sample] = result.samples
        assert solver_sample.trainer_model_id == "solver"

    async def test_an_aborted_solver_is_not_rewarded(self, monkeypatch):
        """Scoring a truncated attempt against the label rewards luck, and its group has no verifier to pair with."""
        result = await _run(
            monkeypatch, solver_response="#### 18", verifier_response="VERDICT: AGREE", abort_solver=True
        )

        assert result.samples[0].reward is None


class TestTheLauncherLeavesThePromptAsMessages:
    def test_the_e2e_test_does_not_ask_the_dataset_to_apply_the_chat_template(self):
        """--apply-chat-template renders the messages into one templated string at dataset build time.
        This example quotes the question into a second prompt, which a string carrying special tokens
        cannot be used for, and a list prompt is chat templated at generation anyway. Getting this wrong
        costs an 8-GPU run to notice."""
        source = (
            Path(__file__).resolve().parents[4] / "examples/multi_policy/run_solver_verifier_gsm8k.py"
        ).read_text()
        # the comment saying why the flag is absent names the flag, so read only what is handed to the run
        launched = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))

        assert "--custom-generate-function-path examples.multi_policy.solver_verifier.generate" in launched
        assert "--apply-chat-template" not in launched

    def test_a_templated_string_prompt_is_refused_rather_than_quoted(self):
        """The refusal is what keeps a prompt full of special tokens out of the verifier's question."""
        with pytest.raises(AssertionError, match="the dataset must use messages"):
            solver_verifier._extract_question("<|im_start|>user\nWhat is 9 + 9?<|im_end|>\n")

    def test_a_message_prompt_gives_up_its_question(self):
        """The other half: the shape the launcher now preserves is the one this reads."""
        question = solver_verifier._extract_question(
            [dict(role="system", content="be terse"), dict(role="user", content="What is 9 + 9?")]
        )

        assert question == "What is 9 + 9?"
