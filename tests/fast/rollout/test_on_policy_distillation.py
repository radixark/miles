import math
from argparse import Namespace

import pytest
from tests.ci.ci_register import register_cpu_ci

from miles.rollout import on_policy_distillation
from miles.rollout.on_policy_distillation import (
    _assert_context_placed,
    _compute_topk_reverse_kl,
    _generation_tail,
    _per_position_ids,
    _privileged_context,
    _score_payload,
    _score_sampled_tokens,
    _teacher_input_ids,
    _teacher_prompt_text,
    _teacher_url_for_sample,
    parse_teacher_urls,
)
from miles.utils.types import Sample

register_cpu_ci(est_time=60, suite="stage-a-cpu")


def _entry(prob: float, token_id: int):
    return [math.log(prob), token_id]


def _args(strategy: str, weight_mode: str = "student_p"):
    return Namespace(
        opd_top_k_strategy=strategy,
        opd_reward_weight_mode=weight_mode,
    )


def _sample():
    return Sample(
        tokens=[10, 11, 12],
        response_length=2,
        metadata={
            "opd_student_top_logprobs": [
                [_entry(0.6, 1), _entry(0.4, 2)],
                [_entry(0.7, 4), _entry(0.3, 5)],
            ]
        },
    )


def _teacher_payload():
    return {
        "teacher": {
            "meta_info": {
                "input_top_logprobs": [
                    None,
                    [_entry(0.5, 2), _entry(0.5, 3)],
                    [_entry(0.8, 4), _entry(0.2, 6)],
                ],
                "input_token_ids_logprobs": [
                    None,
                    [_entry(0.3, 1), _entry(0.7, 2)],
                    [_entry(0.4, 4), _entry(0.6, 5)],
                ],
            }
        },
        "student_on_teacher": {
            "meta_info": {
                "input_token_ids_logprobs": [
                    None,
                    [_entry(0.4, 2), _entry(0.2, 3)],
                    [_entry(0.7, 4), _entry(0.1, 6)],
                ]
            }
        },
    }


def test_topk_only_student_uses_student_probability_weights():
    reverse_kl = _compute_topk_reverse_kl(_args("only-student"), _sample(), _teacher_payload())

    expected_0 = 0.6 * math.log(0.6 / 0.3) + 0.4 * math.log(0.4 / 0.7)
    expected_1 = 0.7 * math.log(0.7 / 0.4) + 0.3 * math.log(0.3 / 0.6)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_topk_intersection_uses_overlap_only():
    reverse_kl = _compute_topk_reverse_kl(_args("intersection", "none"), _sample(), _teacher_payload())

    assert reverse_kl.tolist() == pytest.approx(
        [
            math.log(0.4 / 0.5),
            math.log(0.7 / 0.8),
        ]
    )


def test_topk_only_teacher_does_not_need_student_top_logprobs():
    sample = Sample(tokens=[10, 11, 12], response_length=2)

    reverse_kl = _compute_topk_reverse_kl(_args("only-teacher"), sample, _teacher_payload())

    expected_0 = (2 / 3) * math.log(0.4 / 0.5) + (1 / 3) * math.log(0.2 / 0.5)
    expected_1 = (7 / 8) * math.log(0.7 / 0.8) + (1 / 8) * math.log(0.1 / 0.2)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_topk_xor_uses_symmetric_difference_without_normalization():
    reverse_kl = _compute_topk_reverse_kl(_args("xor", "none"), _sample(), _teacher_payload())

    expected_0 = math.log(0.6 / 0.3) + math.log(0.2 / 0.5)
    expected_1 = math.log(0.3 / 0.6) + math.log(0.1 / 0.2)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_per_position_ids_pads_prompt_and_keeps_response_order():
    # Two response positions, each with two top-k entries [logprob, token_id].
    student_top = [[_entry(0.6, 5), _entry(0.4, 7)], [_entry(0.7, 9), _entry(0.3, 11)]]
    per_pos = _per_position_ids(student_top, prompt_len=3)
    # 3 empty prompt slots, then response positions with their own token ids.
    assert per_pos == [[], [], [], [5, 7], [9, 11]]
    # Aligns with the existing _trim_input_field extraction values[1:][-R:]: for a
    # length-5 response, indices 3,4 are the response positions.
    values = list(range(5))
    assert values[1:][-2:] == [3, 4]
    assert per_pos[3] == [5, 7] and per_pos[4] == [9, 11]


def test_score_payload_routes_per_position_vs_flat():
    flat = _score_payload([1, 2, 3], token_ids=[5, 7])
    assert flat["token_ids_logprob"] == [5, 7]
    assert "token_ids_logprob_positions" not in flat

    per_pos = _score_payload([1, 2, 3], token_ids_positions=[[], [5, 7], [9, 11]])
    assert per_pos["token_ids_logprob_positions"] == [[], [5, 7], [9, 11]]
    assert "token_ids_logprob" not in per_pos


# ---------------------------------------------------------------------------
# Multi-teacher routing (--opd-teacher-urls)
# ---------------------------------------------------------------------------


def _routing_args(urls=None, key="opd_teacher", rm_url="http://single-teacher/generate"):
    return Namespace(opd_teacher_urls=urls, opd_teacher_key=key, rm_url=rm_url)


def _tagged_sample(metadata=None):
    return Sample(tokens=[1, 2, 3], response_length=2, metadata=metadata or {})


def test_parse_teacher_urls_parses_names_and_keeps_equals_in_url():
    url_map = parse_teacher_urls(["math=http://h1:30001/generate", "code=http://h2:30002/generate?tag=a=b"])
    assert url_map == {
        "math": "http://h1:30001/generate",
        "code": "http://h2:30002/generate?tag=a=b",
    }


def test_parse_teacher_urls_empty_or_none_gives_empty_map():
    assert parse_teacher_urls(None) == {}
    assert parse_teacher_urls([]) == {}


@pytest.mark.parametrize("bad", ["math", "=http://h1/generate", "math=", "  =  "])
def test_parse_teacher_urls_rejects_malformed_entries(bad):
    with pytest.raises(ValueError, match="expected NAME=URL"):
        parse_teacher_urls([bad])


def test_parse_teacher_urls_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate teacher name"):
        parse_teacher_urls(["math=http://h1/generate", "math=http://h2/generate"])


def test_routing_unset_map_falls_back_to_rm_url():
    args = _routing_args(urls=None)
    sample = _tagged_sample({"opd_teacher": "math"})
    assert _teacher_url_for_sample(args, sample) == "http://single-teacher/generate"


def test_routing_by_metadata_name():
    args = _routing_args(urls=["math=http://h1/generate", "code=http://h2/generate"])
    assert _teacher_url_for_sample(args, _tagged_sample({"opd_teacher": "math"})) == "http://h1/generate"
    assert _teacher_url_for_sample(args, _tagged_sample({"opd_teacher": "code"})) == "http://h2/generate"


def test_routing_respects_custom_metadata_key():
    args = _routing_args(urls=["math=http://h1/generate"], key="task")
    assert _teacher_url_for_sample(args, _tagged_sample({"task": "math"})) == "http://h1/generate"


def test_routing_missing_name_uses_default_entry():
    args = _routing_args(urls=["math=http://h1/generate", "default=http://h3/generate"])
    assert _teacher_url_for_sample(args, _tagged_sample({})) == "http://h3/generate"


def test_routing_unknown_name_uses_default_entry():
    args = _routing_args(urls=["math=http://h1/generate", "default=http://h3/generate"])
    assert _teacher_url_for_sample(args, _tagged_sample({"opd_teacher": "physics"})) == "http://h3/generate"


def test_routing_unknown_name_without_default_raises():
    args = _routing_args(urls=["math=http://h1/generate"])
    with pytest.raises(ValueError, match="matches no --opd-teacher-urls name"):
        _teacher_url_for_sample(args, _tagged_sample({"opd_teacher": "physics"}))


def test_routing_missing_name_without_default_raises():
    args = _routing_args(urls=["math=http://h1/generate"])
    with pytest.raises(ValueError, match="missing teacher key"):
        _teacher_url_for_sample(args, _tagged_sample({}))


# ---------------------------------------------------------------------------
# Privileged context (--opd-privileged-context-key)
# ---------------------------------------------------------------------------

PRIVILEGED_KEY = "opd_privileged_context"
HINT = "\n\nThe verified answer is 4."
TAIL = "<eot><gen>"
RESPONSE_IDS = [90, 91]


class _StubTokenizer:
    """ChatML-ish stand-in whose tail is TAIL, so the probe can derive it."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None):
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return f"{body}{TAIL}" if add_generation_prompt else body

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class _StubChatTemplateUtils:
    @staticmethod
    def apply_chat_template(messages, *, tokenizer, tools=None, tokenize=False, add_generation_prompt=False, **kw):
        return tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, tools=tools
        )


@pytest.fixture
def _patch_env(monkeypatch):
    import miles.utils

    # The real helper imports sglang; the local import inside _teacher_prompt_text
    # resolves this attribute at call time.
    monkeypatch.setattr(miles.utils, "chat_template_utils", _StubChatTemplateUtils, raising=False)
    monkeypatch.setattr(
        on_policy_distillation,
        "_render_chat",
        lambda args, messages, tokenizer, tools=None: tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ),
    )
    monkeypatch.setattr(on_policy_distillation, "_opd_tokenizer", lambda args: _StubTokenizer())
    on_policy_distillation._GENERATION_TAIL_CACHE.clear()


def _privileged_args(key=PRIVILEGED_KEY, max_context=None, apply_chat_template=True):
    return Namespace(
        opd_privileged_context_key=key,
        rollout_max_context_len=max_context,
        apply_chat_template_kwargs=None,
        apply_chat_template=apply_chat_template,
        hf_checkpoint="stub",
        chat_template_path=None,
    )


MESSAGES = [{"role": "user", "content": "Q"}]
RENDERED = f"<user>Q{TAIL}"
SPLICED_IDS = [ord(c) for c in f"<user>Q{HINT}{TAIL}"]


def _privileged_sample(context=HINT, prompt=RENDERED, response=RESPONSE_IDS):
    return Sample(
        prompt=prompt,
        tokens=[1, 2, *response],
        response_length=len(response),
        metadata={} if context is None else {PRIVILEGED_KEY: context},
    )


def test_privileged_context_absent_when_key_unset(_patch_env):
    args = Namespace(opd_privileged_context_key=None)
    assert _privileged_context(args, _privileged_sample()) is None


def test_privileged_context_absent_when_sample_lacks_key(_patch_env):
    # Lets one dataset mix privileged and plain samples.
    assert _privileged_context(_privileged_args(), _privileged_sample(context=None)) is None


@pytest.mark.parametrize("bad", ["", "   ", 42, []])
def test_privileged_context_rejects_blank_or_non_string(_patch_env, bad):
    with pytest.raises(ValueError, match="non-empty string"):
        _privileged_context(_privileged_args(), _privileged_sample(context=bad))


def test_generation_tail_is_derived_from_the_tokenizer(_patch_env):
    assert _generation_tail(_privileged_args(), _StubTokenizer()) == TAIL


def test_generation_tail_raises_when_the_template_rewrites_content(_patch_env):
    class _Trimming:
        # Gemma-2 trims message content, which swallowed a whitespace-padded probe.
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None):
            return "<user>" + messages[-1]["content"].replace(on_policy_distillation.PROBE_MARKER, "") + TAIL

    with pytest.raises(ValueError, match="Cannot derive the chat template"):
        _generation_tail(_privileged_args(), _Trimming())


def test_generation_tail_raises_when_the_template_echoes_content(_patch_env):
    class _Echoing:
        # A second copy of the content inside the tail would be duplicated on every splice.
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None):
            c = messages[-1]["content"]
            return f"<sys>{c}</sys><user>{c}{TAIL}"

    with pytest.raises(ValueError, match="rendered the probe marker 2 times"):
        _generation_tail(_privileged_args(), _Echoing())


def test_splice_is_unaffected_by_a_prompt_containing_the_marker(_patch_env):
    """The probe renders only the marker, so user text matching it cannot collide."""
    marker = on_policy_distillation.PROBE_MARKER
    sample = _privileged_sample(prompt=f"<user>Ask about {marker} please{TAIL}")
    out = _teacher_prompt_text(_privileged_args(), sample, HINT, _StubTokenizer())
    assert out == f"<user>Ask about {marker} please{HINT}{TAIL}"


def test_rendered_string_and_message_list_prompts_agree(_patch_env):
    """The whole point: one chosen key lands identically in both --apply-chat-template modes."""
    args = _privileged_args()
    from_string = _teacher_prompt_text(args, _privileged_sample(prompt=RENDERED), HINT, _StubTokenizer())
    from_messages = _teacher_prompt_text(args, _privileged_sample(prompt=MESSAGES), HINT, _StubTokenizer())
    assert from_string == from_messages == f"<user>Q{HINT}{TAIL}"


def test_rendered_prompt_missing_the_tail_raises(_patch_env):
    """With --apply-chat-template on, a prompt that does not end in the tail means the
    probe and the renderer disagree -- appending anyway would land past the generation
    prompt. Must raise rather than silently mis-place."""
    sample = _privileged_sample(prompt="<user>Q<something-else>")
    with pytest.raises(ValueError, match="does not end with this template"):
        _teacher_prompt_text(_privileged_args(), sample, HINT, _StubTokenizer())


def test_message_list_not_ending_in_a_user_turn_raises(_patch_env):
    sample = _privileged_sample(prompt=[{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])
    with pytest.raises(ValueError, match="message list ending in a text user message"):
        _teacher_prompt_text(_privileged_args(), sample, HINT, _StubTokenizer())


def test_teacher_input_ids_are_the_spliced_prompt_then_the_response(_patch_env):
    ids = _teacher_input_ids(_privileged_args(), _privileged_sample(), HINT, _StubTokenizer())
    assert ids == [ord(c) for c in f"<user>Q{HINT}{TAIL}"] + RESPONSE_IDS


def test_teacher_input_ids_appends_nothing_when_response_is_empty(_patch_env):
    # Guards the tokens[-response_length:] trap, which returns the whole list at 0.
    ids = _teacher_input_ids(_privileged_args(), _privileged_sample(response=[]), HINT, _StubTokenizer())
    assert ids == [ord(c) for c in f"<user>Q{HINT}{TAIL}"]


def _teacher_response(scored_ids):
    # _trim_input_field drops the leading placeholder, then takes the response tail.
    return {"meta_info": {"input_token_logprobs": [[None, 2], *([-0.1, i] for i in scored_ids)]}}


def test_assert_context_placed_accepts_context_that_survived_rendering():
    _assert_context_placed(f"<user>Q{HINT}{TAIL}", HINT)


def test_message_list_path_raises_when_the_template_drops_the_context(_patch_env, monkeypatch):
    """Reached for real: Gemma-2 applies `content | trim`, so a template can eat it."""
    monkeypatch.setattr(
        on_policy_distillation, "_render_chat", lambda args, messages, tokenizer, tools=None: f"<user>Q{TAIL}"
    )
    with pytest.raises(ValueError, match="did not survive rendering"):
        _teacher_prompt_text(_privileged_args(), _privileged_sample(prompt=MESSAGES), HINT, _StubTokenizer())


def test_zero_matching_samples_is_logged_as_a_warning(_patch_env, caplog):
    """The headline misconfiguration: a typo'd key means every sample takes the
    no-context branch, so the counter must still run there or nothing is ever logged."""
    args = _privileged_args(key="typo_not_in_metadata")
    on_policy_distillation._PRIVILEGED_SEEN[:] = [0, 0]
    with caplog.at_level("WARNING"):
        for _ in range(16):
            on_policy_distillation._teacher_scoring_tokens(args, _privileged_sample())
    assert "0/16 samples scored" in caplog.text


def test_prerendered_prompt_is_spliced_even_with_the_flag_off(_patch_env):
    """--apply-chat-template off does not mean untemplated: a pre-rendered prompt column
    ends in the generation tail, and appending past it would make the context the
    assistant's opening tokens."""
    args = _privileged_args(apply_chat_template=False)
    sample = _privileged_sample(prompt=RENDERED)
    assert _teacher_prompt_text(args, sample, HINT, _StubTokenizer()) == f"<user>Q{HINT}{TAIL}"


def test_untemplated_string_prompt_gets_context_appended(_patch_env):
    """--apply-chat-template off with a plain-text prompt column yields a raw string."""
    args = _privileged_args(apply_chat_template=False)
    sample = _privileged_sample(prompt="What is 2+2?")
    assert _teacher_prompt_text(args, sample, HINT, _StubTokenizer()) == f"What is 2+2?{HINT}"


def test_context_overflow_degrades_instead_of_killing_the_run(_patch_env):
    """A per-sample condition must not raise: it propagates out of async_rm and kills the job."""
    args = _privileged_args(max_context=4)
    assert _teacher_input_ids(args, _privileged_sample(), HINT, _StubTokenizer()) is None


@pytest.mark.asyncio
async def test_score_sampled_tokens_without_context_scores_the_student_sequence(_patch_env, monkeypatch):
    seen = {}

    async def fake_post(url, payload, timeout_secs=None):
        seen["payload"] = payload
        return _teacher_response(RESPONSE_IDS)

    monkeypatch.setattr(on_policy_distillation, "_post_json", fake_post)
    sample = _privileged_sample(context=None)
    await _score_sampled_tokens(_privileged_args(), sample, "http://teacher/generate", None)
    assert seen["payload"]["input_ids"] == sample.tokens


@pytest.mark.asyncio
async def test_score_sampled_tokens_with_context_scores_the_spliced_prompt(_patch_env, monkeypatch):
    seen = {}

    async def fake_post(url, payload, timeout_secs=None):
        seen["payload"] = payload
        return _teacher_response(RESPONSE_IDS)

    monkeypatch.setattr(on_policy_distillation, "_post_json", fake_post)
    await _score_sampled_tokens(_privileged_args(), _privileged_sample(), "http://teacher/generate", None)
    assert seen["payload"]["input_ids"] == [ord(c) for c in f"<user>Q{HINT}{TAIL}"] + RESPONSE_IDS


def _topk_privileged_args(strategy, per_position=False, key=PRIVILEGED_KEY):
    return Namespace(
        opd_privileged_context_key=key,
        apply_chat_template_kwargs=None,
        apply_chat_template=True,
        hf_checkpoint="stub",
        chat_template_path=None,
        rollout_max_context_len=None,
        opd_log_prob_top_k=2,
        opd_top_k_strategy=strategy,
        opd_topk_per_position=per_position,
        opd_teacher_urls=None,
        opd_teacher_key="opd_teacher",
        rm_url="http://teacher/generate",
        sglang_router_ip="student-host",
        sglang_router_port=1234,
        sglang_router_request_timeout_secs=None,
    )


def _topk_privileged_sample(context=HINT):
    sample = _privileged_sample(context=context)
    sample.metadata["opd_student_top_logprobs"] = [
        [_entry(0.6, 90), _entry(0.4, 7)],
        [_entry(0.7, 91), _entry(0.3, 8)],
    ]
    return sample


async def _capture_topk_payloads(monkeypatch, args, sample):
    """Run reward_func against stub transport, returning payloads keyed by URL."""
    payloads = {}

    async def fake_post(url, payload, timeout_secs=None):
        payloads[url] = payload
        return {
            "meta_info": {
                "input_token_logprobs": [[None, 2], *([-0.1, i] for i in RESPONSE_IDS)],
                "input_top_logprobs": [None, [_entry(0.5, 90)], [_entry(0.5, 91)]],
                "input_token_ids_logprobs": [None, [_entry(0.5, 90)], [_entry(0.5, 91)]],
            }
        }

    monkeypatch.setattr(on_policy_distillation, "_post_json", fake_post)
    await on_policy_distillation.reward_func(args, sample)
    return payloads


@pytest.mark.asyncio
async def test_topk_teacher_scores_the_augmented_sequence(_patch_env, monkeypatch):
    payloads = await _capture_topk_payloads(
        monkeypatch, _topk_privileged_args("only-student"), _topk_privileged_sample()
    )
    assert payloads["http://teacher/generate"]["input_ids"] == SPLICED_IDS + RESPONSE_IDS


@pytest.mark.asyncio
async def test_topk_student_rescoring_still_uses_the_public_sequence(_patch_env, monkeypatch):
    # only-teacher re-scores the student on teacher-proposed ids; the student must be
    # conditioned on the prompt it actually saw, not the privileged one.
    sample = _topk_privileged_sample()
    payloads = await _capture_topk_payloads(monkeypatch, _topk_privileged_args("only-teacher"), sample)
    assert payloads["http://student-host:1234/generate"]["input_ids"] == sample.tokens
    assert payloads["http://teacher/generate"]["input_ids"] == SPLICED_IDS + RESPONSE_IDS


@pytest.mark.asyncio
async def test_topk_per_position_pads_each_side_with_its_own_prompt_length(_patch_env, monkeypatch):
    # The teacher's id-lists must be offset by the *teacher* prompt length and the
    # student's by the student's, or both sides read the wrong positions.
    sample = _topk_privileged_sample()
    payloads = await _capture_topk_payloads(monkeypatch, _topk_privileged_args("union", per_position=True), sample)

    teacher_positions = payloads["http://teacher/generate"]["token_ids_logprob_positions"]
    student_positions = payloads["http://student-host:1234/generate"]["token_ids_logprob_positions"]
    assert len(teacher_positions) == len(SPLICED_IDS) + sample.response_length
    assert teacher_positions[: len(SPLICED_IDS)] == [[]] * len(SPLICED_IDS)
    assert teacher_positions[len(SPLICED_IDS) :] == [[90, 7], [91, 8]]

    student_prompt_len = len(sample.tokens) - sample.response_length
    assert len(student_positions) == student_prompt_len + sample.response_length
    assert student_positions[:student_prompt_len] == [[]] * student_prompt_len


@pytest.mark.asyncio
async def test_topk_without_privileged_context_scores_the_student_sequence(_patch_env, monkeypatch):
    args = _topk_privileged_args("only-student", key=None)
    sample = _topk_privileged_sample(context=None)
    payloads = await _capture_topk_payloads(monkeypatch, args, sample)
    assert payloads["http://teacher/generate"]["input_ids"] == sample.tokens


# ---------------------------------------------------------------------------
# Real tokenizers: the stub above agrees with itself by construction, so the
# equivalence of the two prompt modes has to be checked against actual templates.
# ---------------------------------------------------------------------------

# Qwen3-0.6B is already loaded unconditionally by miles/utils/test_utils/mock_tools.py,
# so it is proven present in this CI and must NOT be allowed to skip.
_REQUIRED_MODEL = "Qwen/Qwen3-0.6B"
_REAL_MODELS = [_REQUIRED_MODEL, "Qwen/Qwen3-4B"]


def _real_tokenizer(model_id):
    from miles.utils.processing_utils import load_tokenizer

    try:
        return load_tokenizer(model_id, trust_remote_code=True)
    except (OSError, ConnectionError) as exc:  # offline / rate-limited hub
        if model_id == _REQUIRED_MODEL:
            raise
        pytest.skip(f"tokenizer {model_id} unavailable: {type(exc).__name__}")


REAL_CASES = {
    "single user turn": [{"role": "user", "content": "What is 2+2?"}],
    "system + user": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "What is 2+2?"}],
    "multi turn": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "What is 2+2?"},
    ],
    "content containing the eot literal": [{"role": "user", "content": "Explain <|im_end|> tokens"}],
    # Documented contract: a non-user final turn renders identically under ChatML and the
    # context is appended to that final turn. Pinned so a template change cannot silently
    # invalidate the docs.
    "tool-final": [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "checking"},
        {"role": "tool", "content": "sunny"},
    ],
}


@pytest.mark.parametrize("model_id", _REAL_MODELS)
@pytest.mark.parametrize("case", sorted(REAL_CASES))
def test_both_prompt_modes_agree_on_a_real_template(monkeypatch, model_id, case):
    """Splicing a rendered prompt must equal appending to the message list and rendering."""
    chat_template_utils = pytest.importorskip("miles.utils.chat_template_utils")

    tok = _real_tokenizer(model_id)
    messages = REAL_CASES[case]
    args = Namespace(
        opd_privileged_context_key=PRIVILEGED_KEY,
        rollout_max_context_len=None,
        apply_chat_template_kwargs=None,
        hf_checkpoint=model_id,
        chat_template_path=None,
        apply_chat_template=True,
    )
    monkeypatch.setattr(on_policy_distillation, "_opd_tokenizer", lambda a: tok)
    on_policy_distillation._GENERATION_TAIL_CACHE.clear()

    rendered = chat_template_utils.apply_chat_template(
        messages, tokenizer=tok, tools=None, tokenize=False, add_generation_prompt=True
    )
    from_string = _teacher_prompt_text(args, _privileged_sample(prompt=rendered), HINT, tok)
    if messages[-1]["role"] == "user":
        from_messages = _teacher_prompt_text(args, _privileged_sample(prompt=messages), HINT, tok)
        assert from_string == from_messages
    else:
        # The list path deliberately rejects a non-user final turn; the rendered path
        # cannot tell the difference and appends to that turn, as documented.
        with pytest.raises(ValueError, match="message list ending in a text user message"):
            _teacher_prompt_text(args, _privileged_sample(prompt=messages), HINT, tok)
    # the hint must precede the TRAILING generation prompt; the same tail recurs
    # mid-conversation in multi-turn prompts, so .index() would find the wrong one
    tail = _generation_tail(args, tok)
    assert from_string.endswith(tail)
    assert from_string.index(HINT.strip()) < len(from_string) - len(tail)
