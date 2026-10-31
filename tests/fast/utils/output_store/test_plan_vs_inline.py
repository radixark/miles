"""A rollout through the output store yields the inline path's sample and training columns.

Each case runs one randomized rollout twice through the same production code:
``update_sample_from_response`` per response, ``update_sample_with_tool_responses`` per tool
observation, ``merge_samples`` across session turns and ``strip_last_output_tokens`` for a
truncated tail. One sample gets SGLang's inline meta_info; the other opts into the output store
and gets the ref of the arrays SGLang would store. Their other fields must match, and the store
sample's plan must materialize into the inline sample's arrays.
"""

import asyncio
from argparse import Namespace
from dataclasses import replace
from typing import NamedTuple

import numpy as np
import pybase64
import pytest
import torch
from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config
from tests.fast.utils.output_store.test_plan import Ref, Spec, fake_store_refs

from miles.rollout.generate_utils import tool_call_utils
from miles.rollout.generate_utils.generate_endpoint_utils import update_sample_from_response
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.utils.output_store.plan import (
    MASK_LENGTHS,
    MASK_SUPPORT_LOGPROBS,
    MASK_TOKEN_IDS,
    OUTPUT_STORE_REF_KEY,
    OUTPUT_STORE_REQUEST_KEY,
    RoutedExpertsSegments,
    StorePlanError,
)
from miles.utils.types import Sample

R3 = RoutedExpertsSegments.name
VOCAB = 64
SELECTED_MAX_SUPPORT = 6
LAYERS, EXPERTS_PER_TOKEN = 2, 3
TOPK_COLUMNS = ("rollout_topk_token_ids", "rollout_topk_log_probs")


class Tokenizer:
    @staticmethod
    def decode(tokens) -> str:
        return ""


class Response(NamedTuple):
    tokens: list[int]
    inline_meta_info: dict
    store_meta_info: dict
    arrays: dict[str, np.ndarray] | None  # what SGLang stores; None for a no-output abort


def generate(
    rng: np.random.Generator, *, key: str, num_tokens: int, mode: str, k: int, routed_experts: bool, abort: bool
) -> Response:
    """One SGLang response to ``num_tokens`` input tokens, as inline meta_info and as a stored object."""
    if abort:
        meta_info = {"output_token_logprobs": [], "finish_reason": {"type": "abort"}}
        return Response([], meta_info, dict(meta_info), None)
    max_support = k if mode == "support" else SELECTED_MAX_SUPPORT
    tokens, supports, support_logprobs = [], [], []
    for _ in range(int(rng.integers(0, 6))):
        support = rng.choice(VOCAB, size=int(rng.integers(1, max_support + 1)), replace=False).tolist()
        tokens.append(support[int(rng.integers(len(support)))])
        supports.append(support)
        # float32-exact values, so the inline float32 cast and the stored float32 array agree.
        support_logprobs.append(rng.uniform(-8.0, 0.0, len(support)).astype(np.float32).tolist())
    sampled = [row[s.index(t)] for t, s, row in zip(tokens, supports, support_logprobs, strict=True)]
    common = {
        "output_token_logprobs": [(lp, t, None) for lp, t in zip(sampled, tokens, strict=True)],
        "finish_reason": {"type": "stop"},
    }
    inline = {
        **common,
        "output_token_sampling_mask": supports,
        "output_token_sampling_logprobs": support_logprobs if mode == "support" else sampled,
    }
    # With the output store, SGLang keeps only the sampled tokens' log-probs inline.
    sampled_key = "output_token_sampling_selected_logprobs" if mode == "support" else "output_token_sampling_logprobs"
    store = {**common, sampled_key: sampled}
    arrays = {
        MASK_LENGTHS: np.asarray([len(s) for s in supports], dtype=np.int32),
        MASK_TOKEN_IDS: np.asarray([t for s in supports for t in s], dtype=np.int32),
    }
    if mode == "support":
        arrays[MASK_SUPPORT_LOGPROBS] = np.asarray([v for row in support_logprobs for v in row], dtype=np.float32)
    if routed_experts:
        # Every /generate forward covers the whole sequence: a row per position but the last.
        rows = rng.integers(1, 64, size=(num_tokens + len(tokens) - 1, LAYERS, EXPERTS_PER_TOKEN), dtype=np.int32)
        inline["routed_experts"] = pybase64.b64encode(rows.tobytes()).decode()
        arrays[R3] = rows
    store[OUTPUT_STORE_REF_KEY] = Ref(
        key, tuple(Spec(n, a.dtype.name, a.shape) for n, a in arrays.items())
    ).to_builtins()
    return Response(tokens, inline, store, arrays)


def respond(sample: Sample, meta_info: dict, *, mode: str, k: int, routed_experts: bool, via_store: bool) -> None:
    # A sampling-mask request as compute_request_payload builds it; support mode adds sampling_logprobs_mode.
    payload = {"input_ids": sample.tokens, "return_sampling_mask": True}
    if mode == "support":
        payload["sampling_logprobs_mode"] = "support"
    if routed_experts:
        payload[RoutedExpertsSegments.request_key] = True
    if via_store:
        payload[OUTPUT_STORE_REQUEST_KEY] = True
    args = Namespace(
        rollout_top_logprobs_num=k, sglang=make_sglang_config(speculative_algorithm=None), num_layers=LAYERS
    )
    output = {"text": "", "meta_info": meta_info}
    asyncio.run(update_sample_from_response(args, sample, payload, output, update_loss_mask=True))


def tokens_of(rng: np.random.Generator, low: int, high: int) -> list[int]:
    return rng.integers(0, VOCAB, size=int(rng.integers(low, high + 1))).tolist()


def run_case(
    rng: np.random.Generator, *, mode: str, k: int, routed_experts: bool, key_prefix: str = "", aborts: bool = True
) -> tuple[Sample, Sample, dict]:
    """A session of 1-3 turns, each 1-3 responses with tool observations between them."""
    num_turns = int(rng.integers(1, 4))
    # A no-output abort ends a /generate sample; a session never records one, so no merge follows
    # it (inline, its missing routed experts would stop the merge).
    abort_last = aborts and rng.random() < 0.15 and not (routed_experts and num_turns > 1)
    prefix = tokens_of(rng, 2, 4)
    inline_turns, store_turns, objects = [], [], {}
    for turn in range(num_turns):
        between = tokens_of(rng, 1, 3) if turn else []
        inline, store = Sample(tokens=prefix + between), Sample(tokens=prefix + between)
        num_responses = int(rng.integers(1, 4))
        for i in range(num_responses):
            if i:
                observation = tokens_of(rng, 1, 3)
                for sample in (inline, store):
                    tool_call_utils.update_sample_with_tool_responses(sample, observation, Tokenizer())
            abort = abort_last and turn == num_turns - 1 and i == num_responses - 1
            key = f"{key_prefix}{turn}.{i}"
            response = generate(
                rng, key=key, num_tokens=len(inline.tokens), mode=mode, k=k, routed_experts=routed_experts, abort=abort
            )
            respond(inline, response.inline_meta_info, mode=mode, k=k, routed_experts=routed_experts, via_store=False)
            respond(store, response.store_meta_info, mode=mode, k=k, routed_experts=routed_experts, via_store=True)
            if response.arrays is not None:
                objects[key] = response.arrays
        inline_turns.append(inline)
        store_turns.append(store)
        prefix = inline.tokens

    inline, store = merge_samples(inline_turns, Tokenizer()), merge_samples(store_turns, Tokenizer())
    strip = int(rng.integers(0, inline.response_length + 1)) if rng.random() < 0.4 else 0
    for sample in (inline, store):
        sample.strip_last_output_tokens(strip, Tokenizer())
    return inline, store, objects


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    # The real update_sample_with_tool_responses runs; its "messages" are already the token ids.
    monkeypatch.setattr(tool_call_utils, "tokenize_tool_responses", lambda messages, tokenizer: list(messages))
    fake_store_refs(monkeypatch)


@pytest.mark.parametrize("seed", range(300))
def test_store_sample_equals_inline_sample(seed):
    rng = np.random.default_rng(seed)
    # Requests ask for support log-probs only when top-k is collected in support launch mode
    # (configure_rollout_topk_logprobs_request); support launch mode with k = 0 sends selected requests.
    launch_mode, k = [("selected", 0), ("support", int(rng.integers(1, 5))), ("support", 0)][seed % 3]
    mode = "support" if launch_mode == "support" and k else "selected"
    routed_experts = bool(rng.random() < 0.5)

    inline, store, objects = run_case(rng, mode=mode, k=k, routed_experts=routed_experts)

    for field in ("tokens", "response_length", "loss_mask", "rollout_log_probs", "status"):
        assert getattr(store, field) == getattr(inline, field), field
    assert store.rollout_sampling_mask is None and store.rollout_routed_experts is None
    assert store.rollout_topk_token_ids is None and store.rollout_topk_log_probs is None
    plan = store.rollout_store_plan
    if routed_experts and inline.rollout_routed_experts is None:
        # A no-output abort ended the sample: inline it has no routed experts, the plan no rows.
        assert plan.fields[R3].num_rows() == 0
        plan = replace(plan, fields={name: segments for name, segments in plan.fields.items() if name != R3})
    plan.validate(num_tokens=len(store.tokens), response_length=store.response_length)
    response_tokens = np.asarray(store.tokens[len(store.tokens) - store.response_length :], dtype=np.int64)
    columns = plan.materialize(
        objects, response_tokens=response_tokens, support_topk=k if mode == "support" else 0, indexer_streams=None
    )

    inline_ids, inline_offsets = inline.rollout_sampling_mask._as_tensors()
    for store_column, inline_column, dtype in (
        (columns["rollout_sampling_mask_ids"], inline_ids, torch.int32),
        (columns["rollout_sampling_mask_offsets"], inline_offsets, torch.int64),
    ):
        assert store_column.dtype == inline_column.dtype == dtype
        assert torch.equal(store_column, inline_column), (store_column.tolist(), inline_column.tolist())
    if R3 in plan.fields:
        assert columns["rollout_routed_experts"].dtype == inline.rollout_routed_experts.dtype == np.int32
        np.testing.assert_array_equal(columns["rollout_routed_experts"], inline.rollout_routed_experts)
    if mode == "selected":
        assert set(columns).isdisjoint(TOPK_COLUMNS) and inline.rollout_topk_token_ids is None
        return
    for name, dtype in zip(TOPK_COLUMNS, (np.int32, np.float32), strict=True):
        store_column, inline_column = columns[name], getattr(inline, name)
        assert store_column.dtype == inline_column.dtype == dtype
        assert store_column.shape == inline_column.shape == (inline.response_length, k)
        np.testing.assert_array_equal(store_column, inline_column)


def test_a_response_without_its_ref_raises():
    # The router dropped the opt-in, so SGLang answered inline: the plan cannot be built.
    response = generate(
        np.random.default_rng(0), key="a", num_tokens=3, mode="selected", k=0, routed_experts=True, abort=False
    )
    with pytest.raises(StorePlanError, match="asked for routed_experts"):
        respond(
            Sample(tokens=[1, 2, 3]),
            response.inline_meta_info,
            mode="selected",
            k=0,
            routed_experts=True,
            via_store=True,
        )
