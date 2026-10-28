import asyncio
import math
import threading
from argparse import Namespace

import pytest
import torch
from aiohttp import web
from tests.ci.ci_register import register_cpu_ci

import miles.rollout.on_policy_distillation as on_policy_distillation
from miles.rollout.on_policy_distillation import (
    TOP_K_STRATEGIES,
    _compute_topk_reverse_kl,
    _per_position_ids,
    _post_json,
    _score_payload,
    _scoring_session,
    _teacher_url_for_sample,
    _trim_input_field,
    close_scoring_sessions,
    parse_teacher_urls,
    post_process_rewards,
    reward_func,
)
from miles.utils.types import Sample

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["rollout"])


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


class TestScorePayloadExtraKey:
    def test_no_extra_key_leaves_the_payload_without_one(self):
        """Without an extra key the scoring payload has no extra_key field at all."""
        assert "extra_key" not in _score_payload([1, 2, 3])

    def test_an_extra_key_reaches_the_payload_verbatim(self):
        """The extra key is forwarded unchanged to the scoring request."""
        assert _score_payload([1, 2, 3], extra_key="train:-:7")["extra_key"] == "train:-:7"


class TestRewardFuncExtraKey:
    TEACHER_URL = "http://teacher/generate"
    STUDENT_URL = "http://student:30000/generate"

    def _args(self) -> Namespace:
        return Namespace(
            opd_log_prob_top_k=2,
            opd_top_k_strategy="only-teacher",
            opd_teacher_urls=None,
            opd_teacher_key="opd_teacher",
            rm_url=self.TEACHER_URL,
            sglang_router_ip="student",
            sglang_router_port=30000,
            sglang_router_request_timeout_secs=14400,
            opd_reward_weight_mode="student_p",
            opd_topk_per_position=False,
        )

    @staticmethod
    def _capture_posts(monkeypatch) -> list[tuple[str, dict]]:
        posted: list[tuple[str, dict]] = []

        async def fake_post_json(url, payload, timeout_secs=None):
            posted.append((url, payload))
            rows = [None, [_entry(0.5, 2)], [_entry(0.8, 4)]]
            return {"meta_info": {"input_top_logprobs": rows, "input_token_ids_logprobs": rows}}

        monkeypatch.setattr(on_policy_distillation, "_post_json", fake_post_json)
        return posted

    @pytest.mark.parametrize(argnames="opd_topk_per_position", argvalues=[True, False])
    async def test_only_the_student_score_request_carries_the_sample_key(
        self, monkeypatch: pytest.MonkeyPatch, opd_topk_per_position: bool
    ) -> None:
        """The live student reads the sample's own radix namespace; the frozen teacher needs no partition."""
        posted = self._capture_posts(monkeypatch)
        sample = _sample()
        sample.kv_cache_namespace = "train:-:7"
        args = self._args()
        args.opd_topk_per_position = opd_topk_per_position

        await reward_func(args=args, sample=sample)

        assert [url for url, _ in posted] == [self.TEACHER_URL, self.STUDENT_URL]
        assert "extra_key" not in posted[0][1]
        assert posted[1][1]["extra_key"] == "train:-:7"

    async def test_an_unstarted_sample_scores_the_student_without_a_key(self, monkeypatch):
        """A sample that never entered generate has no namespace to score under."""
        posted = self._capture_posts(monkeypatch)

        await reward_func(self._args(), _sample())

        assert "extra_key" not in posted[1][1]


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
# Response-only scoring (logprob_start_len = prompt_len - 1)
# ---------------------------------------------------------------------------


class _FakeScorer:
    """A /generate scoring endpoint that lays out input logprobs the way SGLang does.

    Rows start at logprob_start_len with a None placeholder, row i scoring the token at
    position logprob_start_len + i; top-k and requested-id rows share that layout, and
    token_ids_logprob_positions is read by absolute input position (sgl-project/sglang#27421).
    """

    VOCAB_SIZE = 32

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def logprob(self, position: int, token_id: int) -> float:
        # a shared base keeps the two models' top-k overlapping; the seed term tells them apart
        base = ((position + 1) * 31 + (token_id + 1) * 17) % 89 + 1
        return -(base + self.seed * ((position + token_id) % 3)) / 16

    def top(self, position: int, k: int) -> list[list]:
        ranked = sorted(range(self.VOCAB_SIZE), key=lambda token_id: -self.logprob(position, token_id))
        return [[self.logprob(position, token_id), token_id, None] for token_id in ranked[:k]]

    def __call__(self, payload: dict) -> dict:
        input_ids = payload["input_ids"]
        positions = range(payload["logprob_start_len"] + 1, len(input_ids))

        def rows(row_of) -> list:
            return [None] + [row_of(position) for position in positions]

        def requested(position: int, token_ids: list[int]) -> list[list]:
            return [[self.logprob(position, token_id), token_id, None] for token_id in token_ids]

        meta_info = {"input_token_logprobs": rows(lambda p: [self.logprob(p, input_ids[p]), input_ids[p], None])}
        if (k := payload.get("top_logprobs_num", 0)) > 0:
            meta_info["input_top_logprobs"] = rows(lambda p: self.top(p, k))
        if (flat := payload.get("token_ids_logprob")) is not None:
            meta_info["input_token_ids_logprobs"] = rows(lambda p: requested(p, flat))
        if (per_position := payload.get("token_ids_logprob_positions")) is not None:
            meta_info["input_token_ids_logprobs"] = rows(lambda p: requested(p, per_position[p]))
        return {"meta_info": meta_info}


_SCORING_TOP_K = 3


class TestResponseOnlyScoring:
    """Scoring from the response boundary must extract exactly what scoring every token did."""

    TEACHER_URL = "http://teacher/generate"
    PROMPT_LEN = 5
    RESPONSE_LEN = 4

    def _args(self, *, top_k: int, strategy: str = "only-student", per_position: bool = False) -> Namespace:
        return Namespace(
            opd_log_prob_top_k=top_k,
            opd_top_k_strategy=strategy,
            opd_reward_weight_mode="student_p",
            opd_topk_per_position=per_position,
            opd_teacher_urls=None,
            opd_teacher_key="opd_teacher",
            rm_url=self.TEACHER_URL,
            sglang_router_ip="student",
            sglang_router_port=30000,
            sglang_router_request_timeout_secs=None,
            reward_key=None,
        )

    def _sample(self, student: _FakeScorer) -> Sample:
        tokens = [(7 * i + 3) % _FakeScorer.VOCAB_SIZE for i in range(self.PROMPT_LEN + self.RESPONSE_LEN)]
        student_top = [
            [entry[:2] for entry in student.top(self.PROMPT_LEN + r, _SCORING_TOP_K)] for r in range(self.RESPONSE_LEN)
        ]
        return Sample(
            tokens=tokens, response_length=self.RESPONSE_LEN, metadata={"opd_student_top_logprobs": student_top}
        )

    async def _score(self, monkeypatch, args: Namespace, *, client_start_len: int | None) -> tuple[Sample, list]:
        """Run reward_func + post_process_rewards; ``client_start_len`` replays an older client's offset."""
        teacher, student = _FakeScorer(seed=1), _FakeScorer(seed=2)
        exchanges: list[tuple[dict, dict]] = []

        async def fake_post_json(url, payload, timeout_secs=None):
            if client_start_len is not None:
                payload = {**payload, "logprob_start_len": client_start_len}
            response = (teacher if url == self.TEACHER_URL else student)(payload)
            exchanges.append((payload, response))
            return response

        monkeypatch.setattr(on_policy_distillation, "_post_json", fake_post_json)
        sample = self._sample(student)
        sample.reward = await reward_func(args, sample)
        post_process_rewards(args, [sample])
        return sample, exchanges

    @staticmethod
    def _signal(sample: Sample) -> torch.Tensor:
        return sample.opd_reverse_kl if sample.teacher_log_probs is None else sample.teacher_log_probs

    @pytest.mark.parametrize(
        argnames="top_k,strategy,per_position",
        argvalues=[(0, "only-student", False)]
        + [
            (_SCORING_TOP_K, strategy, per_position)
            for strategy in sorted(TOP_K_STRATEGIES)
            for per_position in (False, True)
        ],
    )
    async def test_the_extracted_signal_matches_scoring_from_position_zero(
        self, monkeypatch: pytest.MonkeyPatch, top_k: int, strategy: str, per_position: bool
    ) -> None:
        args = self._args(top_k=top_k, strategy=strategy, per_position=per_position)

        new, new_exchanges = await self._score(monkeypatch, args, client_start_len=None)
        old, _ = await self._score(monkeypatch, args, client_start_len=0)

        assert self._signal(new).shape == (self.RESPONSE_LEN,)
        assert torch.equal(self._signal(new), self._signal(old))
        (teacher_payload, teacher_response), *student_exchanges = new_exchanges
        assert teacher_payload["logprob_start_len"] == self.PROMPT_LEN - 1
        assert len(teacher_response["meta_info"]["input_token_logprobs"]) == self.RESPONSE_LEN + 1
        # the live student still scores from position 0, so stale prompt KV cannot leak in
        assert [payload["logprob_start_len"] for payload, _ in student_exchanges] == [0] * len(student_exchanges)

    def test_a_reply_too_short_for_the_response_is_refused(self) -> None:
        meta_info = {"input_token_logprobs": [None, [-0.5, 3, None]]}

        with pytest.raises(ValueError, match="too few for a response of 2 tokens"):
            _trim_input_field(meta_info, "input_token_logprobs", response_length=2)

    def test_a_sample_without_a_prompt_still_scores_from_position_zero(self) -> None:
        sample = Sample(tokens=[1, 2, 3], response_length=3)

        assert on_policy_distillation._response_logprob_start_len(sample) == 0


# ---------------------------------------------------------------------------
# Pooled scoring session
# ---------------------------------------------------------------------------


class _ScoringServer:
    """A local /generate endpoint recording the client socket each request came in on."""

    def __init__(self, *, delay_secs: float = 0.0) -> None:
        self.delay_secs = delay_secs
        self.peers: list = []
        self._runner: web.AppRunner | None = None

    async def __aenter__(self) -> str:
        async def handle(request: web.Request) -> web.Response:
            self.peers.append(request.transport.get_extra_info("peername"))
            await asyncio.sleep(self.delay_secs)
            return web.json_response({"meta_info": {"echo": await request.json()}})

        app = web.Application()
        app.router.add_post("/generate", handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", 0).start()
        host, port = self._runner.addresses[0][:2]
        return f"http://{host}:{port}/generate"

    async def __aexit__(self, *exc_info) -> None:
        await close_scoring_sessions()
        await self._runner.cleanup()


class TestPooledScoringSession:
    async def test_consecutive_scoring_calls_reuse_one_connection(self) -> None:
        server = _ScoringServer()
        async with server as url:
            replies = [await _post_json(url, {"call": i}) for i in range(3)]

        assert [reply["meta_info"]["echo"] for reply in replies] == [{"call": 0}, {"call": 1}, {"call": 2}]
        assert len(server.peers) == 3 and len(set(server.peers)) == 1

    async def test_the_timeout_bounds_only_the_request_that_sets_it(self) -> None:
        async with _ScoringServer(delay_secs=0.3) as url:
            with pytest.raises(asyncio.TimeoutError):
                await _post_json(url, {}, timeout_secs=0.05)
            assert await _post_json(url, {"after": "timeout"}) == {"meta_info": {"echo": {"after": "timeout"}}}

    async def test_closing_releases_the_session_and_the_next_call_opens_a_new_one(self) -> None:
        first = _scoring_session()
        assert _scoring_session() is first

        await close_scoring_sessions()
        second = _scoring_session()

        assert first.closed and not second.closed
        await close_scoring_sessions()

    async def test_closing_reaches_a_session_opened_on_another_running_loop(self) -> None:
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()

        async def open_session():
            return _scoring_session()

        try:
            other = await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(open_session(), loop))
            await close_scoring_sessions()
            assert other.closed
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            loop.close()
