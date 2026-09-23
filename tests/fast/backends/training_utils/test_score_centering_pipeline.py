"""Exercise candidate transport through the real rollout and training consumers."""

import asyncio
from argparse import Namespace
from copy import deepcopy

import numpy as np
import pytest
import torch

from miles.backends.training_utils import parallel
from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss import compute_advantages_and_returns, loss_function
from miles.backends.training_utils.loss_hub.losses import get_loss_function
from miles.backends.training_utils.loss_hub.score_centering import score_centering_loss
from miles.backends.training_utils.parallel import GroupInfo, ParallelState
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp_raw
from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload, update_sample_from_response
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.generate_utils.score_centering import (
    append_score_centering_observations,
    append_score_centering_topk,
    configure_score_centering_request,
    validate_score_centering_sample,
)
from miles.rollout.session.samples.codec import COMPUTED_FIELDS, decode_samples_and_merge_input_sample, encode_samples
from miles.rollout.session.samples.merge import compute_samples_from_openai_records
from miles.rollout.session.types import SessionRecord
from miles.utils.score_centering import validate_score_centering_args
from miles.utils.types import Sample


class _Tokenizer:
    def decode(self, tokens: list[int]) -> str:
        return str(tokens)


def _args(**overrides: object) -> Namespace:
    values = dict(
        loss_type="score_centering",
        score_centering_top_k=3,
        score_centering_is="none",
        score_centering_tis_clip=2.0,
        score_centering_mis_low=0.5,
        score_centering_mis_high=5.0,
        rollout_temperature=0.7,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        advantage_estimator="grpo",
        rewards_normalization=False,
        reward_key=None,
        use_dynamic_global_batch_size=False,
        balance_data=False,
        qkv_format="thd",
        true_on_policy_mode=False,
        allgather_cp=False,
        log_probs_chunk_size=2,
        vocab_size=8,
        entropy_coef=0,
        observe_training_entropy=False,
        use_kl_loss=False,
    )
    return Namespace(**(values | overrides))


@pytest.fixture
def single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    singleton = GroupInfo(rank=0, size=1, group=None)
    monkeypatch.setattr(
        parallel,
        "_parallel_state",
        ParallelState(
            **{name: singleton for name in ("intra_dp", "intra_dp_cp", "cp", "tp", "pp", "ep", "etp", "indep_dp")}
        ),
    )


def _turn(prompt: list[int], output: list[int], probabilities: list[float]) -> Sample:
    logps = np.log(probabilities).tolist()
    sample = Sample(
        tokens=prompt + output,
        response_length=len(output),
        response=str(output),
        rollout_log_probs=logps,
        loss_mask=[1] * len(output),
        status=Sample.Status.COMPLETED,
        index=0,
        group_index=0,
        reward=1.0,
    )
    rows = [[(logp, token, None) for logp, token in zip(logps, output, strict=True)]] * len(output)
    append_score_centering_topk(
        sample,
        {
            "output_token_logprobs": [(logp, token, None) for logp, token in zip(logps, output, strict=True)],
            "output_top_logprobs": rows,
        },
        3,
    )
    return sample


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_multiturn_wire_dp_split_and_training_gradient(single_rank: None, mode: str) -> None:
    first = _turn([0, 1], [2, 3], [0.5, 0.25])
    second = _turn([0, 1, 2, 3, 6], [4, 5], [0.55, 0.2])
    merged = merge_samples([first, second], _Tokenizer())
    assert merged.loss_mask == [1, 1, 0, 1, 1]
    assert (merged.rollout_topk_token_ids[2] == -1).all()
    assert np.isneginf(merged.rollout_topk_log_probs[2]).all()
    merged.strip_last_output_tokens(1, _Tokenizer())
    wire = encode_samples([merged], {"test": True})
    restored = decode_samples_and_merge_input_sample(wire, Sample()).samples[0]
    np.testing.assert_array_equal(restored.rollout_topk_token_ids, [[2, 3, -1], [2, 3, -1], [-1, -1, -1], [4, 5, -1]])
    np.testing.assert_array_equal(restored.rollout_topk_log_probs, merged.rollout_topk_log_probs)
    restored.reward, restored.index = 1.0, 0
    other = deepcopy(restored)
    other.index, other.reward = 1, -0.7
    args = _args(score_centering_is=mode)
    data = convert_samples_to_train_data(args, [restored, other], {}, None, None)
    batch = split_train_data_by_dp_raw(args, data, dp_size=2)[1]
    np.testing.assert_array_equal(batch["rollout_topk_token_ids"][0], restored.rollout_topk_token_ids)
    batch["total_lengths"] = [len(restored.tokens)]
    batch["unconcat_tokens"] = [torch.tensor(restored.tokens)]
    batch["loss_masks"] = [torch.tensor(restored.loss_mask)]
    batch["rollout_log_probs"] = [torch.tensor(restored.rollout_log_probs)]
    batch["advantages"] = [torch.full((restored.response_length,), -0.7)]
    # Observation rows must not leak NaNs from unused sampled logprobs/advantages.
    batch["rollout_log_probs"][0][2] = torch.nan
    batch["advantages"][0][2] = torch.nan
    logits = torch.randn(1, len(restored.tokens), 10, generator=torch.Generator().manual_seed(9), requires_grad=True)
    reduce = get_sum_of_sample_mean(batch["total_lengths"], batch["response_lengths"], batch["loss_masks"])
    loss, metrics = get_loss_function(args)(args, batch, logits, reduce)
    actual = torch.autograd.grad(loss, logits)[0]
    reference_logits = logits.detach().clone().requires_grad_()
    logp = (reference_logits[0, 1:-1, :8] / 0.7).log_softmax(-1)
    ids = torch.tensor(restored.rollout_topk_token_ids, dtype=torch.long)
    mask = torch.tensor(restored.loss_mask, dtype=torch.bool)
    reference, _ = score_centering_loss(
        logp.gather(-1, torch.tensor(restored.tokens[2:])[:, None]).squeeze(-1),
        logp.gather(-1, ids.clamp_min(0)),
        torch.tensor(restored.rollout_log_probs),
        torch.tensor(restored.rollout_topk_log_probs),
        (ids >= 0) & mask[:, None],
        mask.float() * -0.7,
        mode=mode,
    )
    expected = torch.autograd.grad(reduce(reference), reference_logits)[0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert torch.isfinite(loss) and all(torch.isfinite(value) for value in metrics.values())
    assert (actual[0, 3] == 0).all()  # observation token
    assert (actual[..., 8:] == 0).all()  # padded vocabulary
    # Independent k3 formula checks direction and excludes the masked NaN observation.
    train_logp = logp.gather(-1, torch.tensor(restored.tokens[2:])[:, None]).squeeze(-1).detach()
    delta = (train_logp[mask] - torch.tensor(restored.rollout_log_probs)[mask]).clamp(-20, 20)
    expected_kl = (delta.exp() - 1 - delta).clamp(-10, 10).mean()
    torch.testing.assert_close(metrics["train_rollout_kl"], expected_kl)
    assert not metrics["train_rollout_kl"].requires_grad


def test_observation_padding_retry_and_disabled_wire() -> None:
    sample = _turn([0], [2, 3], [0.5, 0.25])
    append_score_centering_observations(sample, 2)
    sample.tokens.extend([6, 6])
    sample.response_length += 2
    sample.rollout_log_probs.extend([0.0, 0.0])
    sample.loss_mask.extend([0, 0])
    validate_score_centering_sample(sample, 3)
    sample.reset_for_retry()
    assert sample.rollout_topk_token_ids is None and sample.rollout_topk_log_probs is None
    old_fields = tuple(field for field in COMPUTED_FIELDS if not field.startswith("rollout_topk_"))
    assert encode_samples([Sample()], {}) == encode_samples([Sample()], {}, fields=old_fields)
    assert (
        decode_samples_and_merge_input_sample(encode_samples([Sample()], {}, fields=old_fields), Sample())
        .samples[0]
        .rollout_topk_token_ids
        is None
    )


@pytest.mark.parametrize("openai", [False, True])
def test_request_candidates_and_sampler_contract(openai: bool) -> None:
    request = {} if openai else {"sampling_params": {}}
    configure_score_centering_request(_args(score_centering_top_k=128), request, openai=openai)
    assert request["top_logprobs" if openai else "top_logprobs_num"] == 128
    sampling = request if openai else request["sampling_params"]
    assert sampling["temperature"] == 0.7
    sampling["top_p"] = 0.9
    with pytest.raises(ValueError, match="top_p"):
        configure_score_centering_request(_args(), request, openai=openai)
    original = deepcopy(request)
    configure_score_centering_request(_args(loss_type="policy_loss"), request, openai=openai)
    assert request == original


@pytest.mark.parametrize(
    "constraint",
    [
        {"tool_choice": "required"},
        {"tool_choice": {"type": "function", "function": {"name": "test"}}},
        {"tools": [{"type": "function", "function": {"name": "test", "strict": True}}]},
        {"response_format": {"type": "json_object"}},
    ],
)
def test_implicit_openai_grammar_constraints_are_rejected(constraint: dict) -> None:
    with pytest.raises(ValueError):
        configure_score_centering_request(_args(), constraint, openai=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rollout_top_p", 0.9),
        ("rollout_temperature", 0),
        ("rollout_top_k", 5),
        ("score_centering_top_k", 0),
        ("score_centering_tis_clip", float("inf")),
        ("score_centering_mis_low", 6),
        ("use_tis", True),
        ("advantage_estimator", "gspo"),
        ("recompute_logprobs_via_prefill", True),
        ("sglang_speculative_algorithm", "EAGLE"),
    ],
)
def test_invalid_options_fail_early(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        validate_score_centering_args(_args(**{field: value}))


@pytest.mark.parametrize("session", ["v1", "v2"])
@pytest.mark.parametrize("top_k", [21, 128])
def test_large_session_heads_require_miles_router(session: str, top_k: int) -> None:
    args = _args(use_session_server=session, score_centering_top_k=top_k, use_miles_router=False)
    with pytest.raises(ValueError, match="require --use-miles-router"):
        validate_score_centering_args(args)

    args.use_miles_router = True
    validate_score_centering_args(args)


@pytest.mark.parametrize("session", ["v1", "v2"])
def test_standard_openai_head_size_can_use_sglang_router(session: str) -> None:
    validate_score_centering_args(_args(use_session_server=session, score_centering_top_k=20, use_miles_router=False))


def test_large_native_heads_can_use_sglang_router() -> None:
    validate_score_centering_args(_args(use_session_server=None, score_centering_top_k=128, use_miles_router=False))


def test_other_losses_do_not_require_score_centering_router() -> None:
    validate_score_centering_args(
        _args(loss_type="policy_loss", use_session_server="v2", score_centering_top_k=128, use_miles_router=False)
    )


def test_missing_or_mismatched_probabilities_fail_before_training() -> None:
    sample = _turn([0], [2, 3], [0.5, 0.25])
    sample.rollout_log_probs[0] -= 0.1
    with pytest.raises(ValueError, match="same sampler"):
        validate_score_centering_sample(sample, 3)
    sample.rollout_topk_token_ids[1] = -1
    with pytest.raises(ValueError, match="Every trained token"):
        validate_score_centering_sample(sample, 3)
    with pytest.raises(ValueError, match="output_top_logprobs"):
        append_score_centering_topk(Sample(response_length=1), {"output_token_logprobs": [(-1.0, 2, None)]}, 3)


def _meta(output: list[int], probabilities: list[float]) -> dict:
    entries = [(float(np.log(p)), token, None) for p, token in zip(probabilities, output, strict=True)]
    return {
        "output_token_logprobs": entries,
        "output_top_logprobs": [entries] * len(output),
        "finish_reason": {"type": "stop"},
    }


def test_native_generate_producer_appends_each_call() -> None:
    args = _args(
        rollout_max_response_len=20,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        sglang_speculative_algorithm=None,
    )
    payload, status = compute_request_payload(args, [0, 1], {})
    assert status is None and payload["top_logprobs_num"] == 3 and payload["return_logprob"]
    sample = Sample()
    for output, probabilities in (([2, 3], [0.5, 0.25]), ([4, 5], [0.55, 0.2])):
        asyncio.run(
            update_sample_from_response(
                args, sample, payload, {"text": str(output), "meta_info": _meta(output, probabilities)}, True
            )
        )
    assert sample.tokens == [0, 1, 2, 3, 4, 5]
    np.testing.assert_array_equal(sample.rollout_topk_token_ids[:, 0], [2, 2, 4, 4])
    validate_score_centering_sample(sample, 3)


def test_greedy_evaluation_does_not_collect_training_candidates() -> None:
    args = _args(
        rollout_max_response_len=20,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        sglang_speculative_algorithm=None,
    )
    payload, _ = compute_request_payload(args, [0, 1], {"temperature": 0.0}, evaluation=True)
    assert payload["sampling_params"]["temperature"] == 0.0 and "top_logprobs_num" not in payload
    meta = _meta([2, 3], [0.5, 0.25])
    del meta["output_top_logprobs"]
    sample = Sample()
    asyncio.run(update_sample_from_response(args, sample, payload, {"text": "eval", "meta_info": meta}))
    assert sample.rollout_topk_token_ids is None


def test_session_producer_trims_candidates_with_tito_tokens() -> None:
    records = []
    for prompt, output, probabilities in (([0, 1], [2, 3], [0.5, 0.25]), ([0, 1, 2, 6], [4, 5], [0.55, 0.2])):
        records.append(
            SessionRecord(
                timestamp=2.0,
                request_timestamp=1.0,
                method="POST",
                path="v1/chat/completions",
                status_code=200,
                request={"input_ids": prompt, "top_logprobs": 3},
                response={"choices": [{"meta_info": _meta(output, probabilities), "finish_reason": "stop"}]},
            )
        )
    samples = compute_samples_from_openai_records(
        _args(save_debug_trajectory_data=None, sglang_speculative_algorithm=None),
        records,
        _Tokenizer(),
        accumulated_token_ids=[0, 1, 2, 6, 4, 5],
        max_trim_tokens=1,
    )
    assert samples[0].response_length == 1
    assert samples[0].rollout_topk_token_ids.shape == (1, 3)
    merged = merge_samples(samples, _Tokenizer())
    assert merged.loss_mask == [1, 0, 1, 1]
    validate_score_centering_sample(merged, 3)


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_shared_advantages_loss_scaling_and_regularization(
    single_rank: None, per_token: bool, recompute: bool
) -> None:
    args = _args(
        use_rollout_logprobs=True,
        skip_actor_forward_only=False,
        use_opd=False,
        normalize_advantages=False,
        kl_coef=0,
        entropy_coef=0.03,
        use_kl_loss=True,
        use_unbiased_kl=True,
        kl_loss_type="k2",
        kl_loss_coef=0.1,
        calculate_per_token_loss=per_token,
        recompute_loss_function=recompute,
        global_batch_size=1,
    )
    sample = _turn([0, 1], [2, 3], [0.5, 0.25])
    batch = convert_samples_to_train_data(args, [sample], {}, None, None)
    batch.update(
        total_lengths=[4],
        unconcat_tokens=[torch.tensor(sample.tokens)],
        loss_masks=[torch.tensor([1, 0])],
        rollout_mask_sums=torch.tensor([1]),
        rollout_log_probs=[torch.tensor([np.log(0.5), torch.nan], dtype=torch.float32)],
        ref_log_probs=[torch.tensor([-0.8, torch.nan])],
    )
    compute_advantages_and_returns(args, batch)
    logits = torch.randn(1, 4, 10, generator=torch.Generator().manual_seed(21), requires_grad=True)
    actual, normalizer, metrics = loss_function(args, batch, 2, logits, apply_megatron_loss_scaling=True)
    reference_logits = logits.detach().clone().requires_grad_()
    logp = (reference_logits[0, 1, :8] / 0.7).log_softmax(-1)
    pg, _ = score_centering_loss(
        logp[2:3],
        logp[torch.tensor([[2, 3]])],
        torch.tensor([np.log(0.5)]),
        torch.tensor([[np.log(0.5), np.log(0.25)]]),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1),
    )
    entropy = -(logp.exp() * logp).sum()
    kl = (logp[2] + 0.8).square() / 2 * (logp[2] - np.log(0.5)).exp()
    expected = (pg.sum() - 0.03 * entropy + 0.1 * kl) * (1 if per_token else 2)
    torch.testing.assert_close(actual, expected, check_dtype=False)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, atol=1e-6, rtol=1e-5)
    assert normalizer == 1 and "entropy_loss" in metrics["keys"] and "kl_loss" in metrics["keys"]
    assert "train_rollout_kl" in metrics["keys"]


@pytest.mark.parametrize("candidates", [[2, 3, -1], [-1, 3, 2], [-1, 2, -1], [-1, -1, -1]])
def test_candidate_validation_preserves_order_and_allows_repeated_padding(candidates: list[int]) -> None:
    sample = _turn([0], [2, 3], [0.5, 0.25])
    sample.rollout_topk_token_ids[:] = candidates
    sample.rollout_topk_log_probs[:] = [
        -np.inf if token == -1 else np.log(0.5 if token == 2 else 0.25) for token in candidates
    ]
    if all(token == -1 for token in candidates):
        sample.loss_mask = [0, 0]
    original_ids = sample.rollout_topk_token_ids.copy()
    original_logps = sample.rollout_topk_log_probs.copy()
    validate_score_centering_sample(sample, 3)
    np.testing.assert_array_equal(sample.rollout_topk_token_ids, original_ids)
    np.testing.assert_array_equal(sample.rollout_topk_log_probs, original_logps)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("token", [0, 3])
def test_candidate_validation_rejects_unsorted_duplicates(masked: bool, token: int) -> None:
    sample = _turn([0], [2, 3], [0.5, 0.25])
    sample.rollout_topk_token_ids[1] = [token, 2, token]
    sample.rollout_topk_log_probs[1] = np.log([0.25, 0.5, 0.25])
    sample.loss_mask[1] = 0 if masked else 1
    with pytest.raises(ValueError, match="Duplicate"):
        validate_score_centering_sample(sample, 3)
