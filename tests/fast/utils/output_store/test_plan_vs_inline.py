"""The store path's sampling-mask / top-k columns equal the inline path's on randomized rollouts.

Each case runs one rollout both ways. The inline path builds a ``Sample`` from SGLang's meta_info:
``update_sample_from_response`` per response, ``update_sample_with_tool_responses`` per tool
observation, ``merge_samples`` across session turns and ``strip_last_output_tokens`` for a
truncated tail. The store path builds a ``StorePlan`` from the refs of the objects SGLang would
store, applies the same edits and materializes it. The training columns must be identical.
"""

import asyncio
from argparse import Namespace
from typing import NamedTuple

import numpy as np
import pytest
import torch
from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config

from miles.rollout.generate_utils import tool_call_utils
from miles.rollout.generate_utils.generate_endpoint_utils import update_sample_from_response
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.utils.output_store.plan import (
    MASK_LENGTHS,
    MASK_SUPPORT_LOGPROBS,
    MASK_TOKEN_IDS,
    SamplingMaskSegments,
    StorePlan,
)
from miles.utils.types import Sample

MASK = SamplingMaskSegments.name
VOCAB = 64
SELECTED_MAX_SUPPORT = 6
TOPK_COLUMNS = ("rollout_topk_token_ids", "rollout_topk_log_probs")
REQUEST = {SamplingMaskSegments.request_key: True}


class Spec(NamedTuple):
    name: str
    dtype: str
    shape: tuple[int, ...]


class Ref(NamedTuple):
    key: str
    fields: tuple[Spec, ...]


class Tokenizer:
    @staticmethod
    def decode(tokens) -> str:
        return ""


class Response(NamedTuple):
    tokens: list[int]
    meta_info: dict
    arrays: dict[str, np.ndarray] | None  # what SGLang stores; None for a no-output abort


def generate(rng: np.random.Generator, *, mode: str, k: int, abort: bool) -> Response:
    """One SGLang response: output tokens, the meta_info of the inline path and the stored arrays."""
    if abort:
        return Response([], {"output_token_logprobs": [], "finish_reason": {"type": "abort"}}, None)
    max_support = k if mode == "support" else SELECTED_MAX_SUPPORT
    tokens, supports, support_logprobs = [], [], []
    for _ in range(int(rng.integers(0, 6))):
        support = rng.choice(VOCAB, size=int(rng.integers(1, max_support + 1)), replace=False).tolist()
        tokens.append(support[int(rng.integers(len(support)))])
        supports.append(support)
        # float32-exact values, so the inline float32 cast and the stored float32 array agree.
        support_logprobs.append(rng.uniform(-8.0, 0.0, len(support)).astype(np.float32).tolist())
    sampled = [row[s.index(t)] for t, s, row in zip(tokens, supports, support_logprobs, strict=True)]
    meta_info = {
        "output_token_logprobs": [(lp, t, None) for lp, t in zip(sampled, tokens, strict=True)],
        "output_token_sampling_mask": supports,
        "output_token_sampling_logprobs": support_logprobs if mode == "support" else sampled,
        "finish_reason": {"type": "stop"},
    }
    arrays = {
        MASK_LENGTHS: np.asarray([len(s) for s in supports], dtype=np.int32),
        MASK_TOKEN_IDS: np.asarray([t for s in supports for t in s], dtype=np.int32),
    }
    if mode == "support":
        arrays[MASK_SUPPORT_LOGPROBS] = np.asarray([v for row in support_logprobs for v in row], dtype=np.float32)
    return Response(tokens, meta_info, arrays)


def inline_response(sample: Sample, response: Response, *, mode: str, k: int) -> None:
    # A sampling-mask request as compute_request_payload builds it; support mode adds sampling_logprobs_mode.
    payload = {"input_ids": sample.tokens, "return_sampling_mask": True}
    if mode == "support":
        payload["sampling_logprobs_mode"] = "support"
    args = Namespace(rollout_top_logprobs_num=k, sglang=make_sglang_config(speculative_algorithm=None))
    output = {"text": "", "meta_info": response.meta_info}
    asyncio.run(update_sample_from_response(args, sample, payload, output, update_loss_mask=True))


def store_response(response: Response, *, key: str, num_tokens: int) -> tuple[StorePlan, dict]:
    if response.arrays is None:
        plan = StorePlan.from_response(
            None, request=REQUEST, num_tokens=num_tokens, response_length=0, no_output_abort=True
        )
        return plan, {}
    ref = Ref(key, tuple(Spec(name, a.dtype.name, a.shape) for name, a in response.arrays.items()))
    plan = StorePlan.from_response(ref, request=REQUEST, num_tokens=num_tokens, response_length=len(response.tokens))
    return plan, {key: response.arrays}


def tokens_of(rng: np.random.Generator, low: int, high: int) -> list[int]:
    return rng.integers(0, VOCAB, size=int(rng.integers(low, high + 1))).tolist()


def run_case(rng: np.random.Generator, *, mode: str, k: int) -> tuple[Sample, StorePlan, dict]:
    """A session of 1-3 turns, each 1-3 responses with tool observations between them."""
    num_turns = int(rng.integers(1, 4))
    abort_last = rng.random() < 0.15
    prefix = tokens_of(rng, 1, 4)
    turns, plan, objects = [], None, {}
    for turn in range(num_turns):
        between = tokens_of(rng, 1, 3) if turn else []
        sample = Sample(tokens=prefix + between)
        turn_plan = None
        num_responses = int(rng.integers(1, 4))
        for i in range(num_responses):
            if i:
                observation = tokens_of(rng, 1, 3)
                tool_call_utils.update_sample_with_tool_responses(sample, observation, Tokenizer())
                turn_plan = turn_plan.insert_tokens(len(observation))
            abort = abort_last and turn == num_turns - 1 and i == num_responses - 1
            response = generate(rng, mode=mode, k=k, abort=abort)
            inline_response(sample, response, mode=mode, k=k)
            new_plan, new_objects = store_response(response, key=f"{turn}.{i}", num_tokens=len(sample.tokens))
            objects |= new_objects
            turn_plan = new_plan if turn_plan is None else turn_plan.append_response(new_plan)
        turns.append(sample)
        plan = turn_plan if plan is None else plan.merge_turn(turn_plan, observation_tokens=len(between))
        prefix = sample.tokens

    sample = merge_samples(turns, Tokenizer())
    strip = int(rng.integers(0, sample.response_length + 1)) if rng.random() < 0.4 else 0
    sample.strip_last_output_tokens(strip, Tokenizer())
    return sample, plan.strip_tail(strip), objects


@pytest.fixture(autouse=True)
def tool_observations_are_their_tokens(monkeypatch):
    # The real update_sample_with_tool_responses runs; its "messages" are already the token ids.
    monkeypatch.setattr(tool_call_utils, "tokenize_tool_responses", lambda messages, tokenizer: list(messages))


@pytest.mark.parametrize("seed", range(300))
def test_store_columns_equal_inline_columns(seed):
    rng = np.random.default_rng(seed)
    # Requests ask for support log-probs only when top-k is collected in support launch mode
    # (configure_rollout_topk_logprobs_request); support launch mode with k = 0 sends selected requests.
    launch_mode, k = [("selected", 0), ("support", int(rng.integers(1, 5))), ("support", 0)][seed % 3]
    mode = "support" if launch_mode == "support" and k else "selected"

    sample, plan, objects = run_case(rng, mode=mode, k=k)

    plan.validate(num_tokens=len(sample.tokens), response_length=sample.response_length)
    response_tokens = np.asarray(sample.tokens[len(sample.tokens) - sample.response_length :], dtype=np.int64)
    support_topk = k if mode == "support" else 0
    columns = plan.materialize(
        objects, response_tokens=response_tokens, support_topk=support_topk, indexer_streams=None
    )

    inline_ids, inline_offsets = sample.rollout_sampling_mask._as_tensors()
    for store, inline, dtype in (
        (columns["rollout_sampling_mask_ids"], inline_ids, torch.int32),
        (columns["rollout_sampling_mask_offsets"], inline_offsets, torch.int64),
    ):
        assert store.dtype == inline.dtype == dtype
        assert torch.equal(store, inline), (store.tolist(), inline.tolist())
    if mode == "selected":
        assert set(columns).isdisjoint(TOPK_COLUMNS) and sample.rollout_topk_token_ids is None
        return
    for name, dtype in zip(TOPK_COLUMNS, (np.int32, np.float32), strict=True):
        store, inline = columns[name], getattr(sample, name)
        assert store.dtype == inline.dtype == dtype and store.shape == inline.shape == (sample.response_length, k)
        np.testing.assert_array_equal(store, inline)
