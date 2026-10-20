"""Filtered rollout probabilities through candidate transport and the real loss."""

import math
from copy import deepcopy

import numpy as np
import pytest
import torch
from tests.fast.fixtures.score_centering_fixtures import _args, _Tokenizer

from miles.backends.training_utils import parallel
from miles.backends.training_utils.data.replay import materialize_replay_columns
from miles.backends.training_utils.data.rollout import DataIterator
from miles.backends.training_utils.loss.hub.score_centering_loss import score_centering_loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState
from miles.backends.training_utils.torch_native.actor import SAMPLING_MASK_KEYS, TRAIN_KEYS
from miles.ray.rollout.train_data_conversion import (
    ROLLOUT_DATA_VALUE_SPEC,
    convert_samples_to_train_data,
    split_train_data_by_dp_raw,
)
from miles.rollout.generate_utils.rollout_topk_logprobs import configure_rollout_topk_logprobs_request
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.session.samples.codec import COMPUTED_FIELDS, decode_samples_and_merge_input_sample, encode_samples
from miles.utils.object_store import RayObjectStore
from miles.utils.replay_assembly import derive_sampling_mask, derive_support_topk
from miles.utils.replay_index import ReplayDataError, ReplayIndex
from miles.utils.score_centering import validate_score_centering_args
from miles.utils.types import Sample


def _support_mask(supports: list[list[int]], probabilities: list[list[float]]) -> ReplayIndex:
    return ReplayIndex.from_meta_info(
        {
            "output_token_sampling_mask": supports,
            "output_token_sampling_logprobs": [[math.log(p) for p in row] for row in probabilities],
        },
        via_store=False,
        routed_experts_rows=None,
        num_layers=None,
        indexer_topk_rows=None,
        indexer_num_streams=None,
        return_sampling_mask=True,
        support_logprobs=True,
        num_output_tokens=len(supports),
        aborted=False,
    )


def _trainer_data(args, samples: list[Sample]) -> dict:
    """Rollout-side conversion, then the trainer's replay derivation, as one rank sees it."""
    data = convert_samples_to_train_data(args, samples, {}, None, None)
    data["total_lengths"] = [len(tokens) for tokens in data["tokens"]]
    materialize_replay_columns(args, data)
    return data


def _with_producer_candidates(sample: Sample, k: int = 3) -> Sample:
    """Record support candidates on the sample, as a custom rollout producer would."""
    response = np.asarray(sample.tokens[-sample.response_length :], dtype=np.int64)
    derived = derive_sampling_mask(sample.replay_index.sampling_mask.mask_pieces({}), response)
    sample.rollout_topk_token_ids, sample.rollout_topk_log_probs = derive_support_topk(derived, k=k)
    return sample


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


def _filtered_sample() -> Sample:
    # SGLang support mode returns the actual post-filter behavior probabilities.
    return Sample(
        tokens=[0, 1, 2],
        response_length=1,
        response="2",
        replay_index=_support_mask([[3, 2]], [[3 / 7, 4 / 7]]),
        rollout_log_probs=[math.log(4 / 7)],
        loss_mask=[1],
        status=Sample.Status.COMPLETED,
        index=0,
        group_index=0,
        reward=1.0,
    )


def _support_args(**overrides):
    return _args(rollout_sampling_logprobs_mode="support", use_sampling_support_replay=True, **overrides)


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
@pytest.mark.parametrize("ci_test", [False, True])
def test_filtered_loss_matches_dense_support_gradient(single_rank: None, mode: str, ci_test: bool) -> None:
    sample = _filtered_sample()
    args = _support_args(
        ci_test=ci_test,
        score_centering_is=mode,
        rollout_top_p=0.6,
        rollout_top_k=3,
        entropy_coef=0.03,
    )
    validate_score_centering_args(args)
    data = _trainer_data(args, [sample])
    assert data["rollout_sampling_mask_ids"][0].tolist() == [3, 2]
    assert data["rollout_topk_lengths"][0].tolist() == [2]
    np.testing.assert_allclose(np.exp(data["rollout_topk_log_probs"][0][0, :2]), [3 / 7, 4 / 7], rtol=1e-6)
    np.testing.assert_allclose(data["rollout_log_probs"][0], [math.log(4 / 7)], rtol=1e-6)
    batch = {
        "unconcat_tokens": [torch.tensor(sample.tokens)],
        "total_lengths": [len(sample.tokens)],
        "response_lengths": [sample.response_length],
        "loss_masks": [torch.tensor(sample.loss_mask)],
        "rollout_log_probs": [torch.tensor(values) for values in data["rollout_log_probs"]],
        "rollout_topk_lengths": data["rollout_topk_lengths"],
        "rollout_sampling_mask_ids": data["rollout_sampling_mask_ids"],
        "rollout_sampling_mask_offsets": data["rollout_sampling_mask_offsets"],
        "rollout_topk_log_probs": data["rollout_topk_log_probs"],
        "advantages": [torch.tensor([0.7])],
    }
    logits = torch.tensor(
        [[[0.0] * 8, [0.0, 0.0, 1.4, 0.2, 2.1, -0.3, 0.1, 0.5], [0.0] * 8]],
        requires_grad=True,
    )
    loss, metrics = score_centering_loss_function(args, batch, logits, torch.mean)
    actual = torch.autograd.grad(loss, logits)[0]

    reference_logits = logits.detach().clone().requires_grad_()
    logp = torch.log_softmax(reference_logits[0, 1, [2, 3]] / 0.7, dim=0)
    q = torch.tensor([4 / 7, 3 / 7])
    ratio = logp.detach().exp() / q
    if mode == "none":
        weight = torch.ones_like(ratio)
    elif mode == "tis":
        weight = ratio.clamp(max=2)
    else:
        weight = torch.where((ratio >= 0.5) & (ratio <= 5), ratio, 0)
    support_entropy = -(logp.exp() * logp).sum()
    reference = -0.7 * (weight[0] * logp[0] - (q * weight * logp).sum()) - 0.03 * support_entropy
    expected = torch.autograd.grad(reference, reference_logits)[0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual[0, 1, [0, 1, 4, 5, 6, 7]], torch.zeros(6), atol=1e-6, rtol=0)
    torch.testing.assert_close(metrics["sc_rollout_head_mass"], torch.ones_like(metrics["sc_rollout_head_mass"]))
    torch.testing.assert_close(metrics["entropy_loss"], support_entropy.detach())


@pytest.mark.parametrize("ci_test", [False, True])
def test_missing_support_candidate_checked_only_in_ci(ci_test: bool) -> None:
    sample = _with_producer_candidates(_filtered_sample())
    sample.rollout_topk_token_ids[0, 1] = -1
    sample.rollout_topk_log_probs[0, 1] = -np.inf
    args = _support_args(ci_test=ci_test)
    if ci_test:
        with pytest.raises(ValueError, match="must equal the sampling support"):
            _trainer_data(args, [sample])
    else:
        data = _trainer_data(args, [sample])
        np.testing.assert_array_equal(data["rollout_topk_token_ids"][0], sample.rollout_topk_token_ids)
        np.testing.assert_array_equal(data["rollout_topk_log_probs"][0], sample.rollout_topk_log_probs)


def test_support_wider_than_candidates_fails_on_the_trainer() -> None:
    sample = _filtered_sample()
    sample.replay_index = _support_mask([[2, 5, 7, 8]], [[0.4, 0.3, 0.2, 0.1]])
    with pytest.raises(ReplayDataError, match="support exceeds"):
        _trainer_data(_support_args(), [sample])


def test_missing_support_probabilities_fail_closed() -> None:
    with pytest.raises(ReplayDataError, match="output_token_sampling_logprobs"):
        ReplayIndex.from_meta_info(
            {"output_token_sampling_mask": [[2, 3]]},
            via_store=False,
            routed_experts_rows=None,
            num_layers=None,
            indexer_topk_rows=None,
            indexer_num_streams=None,
            return_sampling_mask=True,
            support_logprobs=True,
            num_output_tokens=1,
            aborted=False,
        )


@pytest.mark.parametrize("top_p,top_k", [(0.8, 4), (1.0, 4)])
def test_uncovered_support_rejected(top_p: float, top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        validate_score_centering_args(
            _args(rollout_top_p=top_p, rollout_top_k=top_k, rollout_sampling_logprobs_mode="support")
        )


def test_request_override_cannot_exceed_candidate_count() -> None:
    request = {"sampling_params": {"temperature": 0.7, "top_p": 0.6, "top_k": 4}}
    with pytest.raises(ValueError, match="top_k"):
        configure_rollout_topk_logprobs_request(_args(rollout_top_p=0.6, rollout_top_k=3), request)


@pytest.mark.parametrize("layout", ["ordered", "permuted", "gap", "mismatch", "negative_padding"])
def test_compaction_requires_ordered_equivalence(layout):
    first, second = _with_producer_candidates(_filtered_sample()), _with_producer_candidates(_filtered_sample())
    if layout == "permuted":
        second.rollout_topk_token_ids[0, :2] = [2, 3]
        second.rollout_topk_log_probs[0, :2] = second.rollout_topk_log_probs[0, :2][::-1]
    elif layout == "gap":
        second.rollout_topk_token_ids[0] = [3, -1, 2]
    elif layout == "mismatch":
        second.rollout_topk_token_ids[0, 1] = 4
    elif layout == "negative_padding":
        second.rollout_topk_token_ids[0, 2] = -2
    # Valid -inf logprobs must not be mistaken for absent candidate IDs.
    first.rollout_topk_log_probs[0, 0] = -np.inf
    first.loss_mask = [0]
    data = _trainer_data(_support_args(), [first, second])
    if layout == "ordered":
        assert "rollout_topk_token_ids" not in data
        assert [value.tolist() for value in data["rollout_topk_lengths"]] == [[2], [2]]
        assert all(value.dtype == torch.int32 for value in data["rollout_topk_lengths"])
    else:
        assert "rollout_topk_lengths" not in data
        np.testing.assert_array_equal(data["rollout_topk_token_ids"][1], second.rollout_topk_token_ids)


def test_trainer_derived_candidates_match_producer_candidates():
    derived = _trainer_data(_support_args(loss_type="policy_loss"), [_filtered_sample()])
    produced = _trainer_data(_support_args(loss_type="policy_loss"), [_with_producer_candidates(_filtered_sample())])
    for field in ("rollout_topk_token_ids", "rollout_topk_log_probs"):
        np.testing.assert_array_equal(derived[field][0], produced[field][0])
    np.testing.assert_array_equal(derived["rollout_topk_token_ids"][0], [[3, 2, -1]])


@pytest.mark.parametrize(
    "loss_type,mode,k",
    [("policy_loss", "support", 3), ("score_centering", "selected", 3), ("policy_loss", "support", 0)],
)
def test_unrelated_conversion_keeps_existing_representation(loss_type, mode, k):
    from tests.fast.fixtures.score_centering_fixtures import _turn

    # Selected mode records candidates during generation; support mode derives them on the trainer.
    sample = _turn([0, 1], [2], [0.5]) if mode == "selected" else _filtered_sample()
    data = _trainer_data(
        _args(loss_type=loss_type, rollout_sampling_logprobs_mode=mode, rollout_top_logprobs_num=k), [sample]
    )
    assert "rollout_topk_lengths" not in data
    assert ("rollout_topk_token_ids" in data) == bool(k)


@pytest.mark.parametrize("delay", [False, True])
@pytest.mark.parametrize("use_kl", [False, True])
def test_compact_session_object_store_and_iterator(single_rank, ray_local_mode, use_kl, delay):
    first = _session_turn([0, 1], [2, 3])
    second = _session_turn([0, 1, 2, 3, 6], [4, 5])
    merged = merge_samples([first, second], _Tokenizer())
    merged.strip_last_output_tokens(1, _Tokenizer())
    wire = encode_samples([merged], {}, fields=COMPUTED_FIELDS)
    restored = decode_samples_and_merge_input_sample(wire, Sample(), fields=COMPUTED_FIELDS).samples[0]
    restored.index, restored.reward = 0, 1.0
    restored.loss_mask[1] = 0
    other = deepcopy(restored)
    other.index = 1
    args = _support_args(
        ci_test=True,
        enable_delay_token_replay_assembly=delay,
        use_kl_loss=use_kl,
        use_unbiased_kl=True,
        kl_loss_type="k2",
        kl_loss_coef=0.1,
        entropy_coef=0.03,
    )
    data = convert_samples_to_train_data(args, [restored, other], {}, None, None)
    # The replay index crosses the object store only when the train rank assembles it.
    assert ("replay_index" in data) == delay
    shard = split_train_data_by_dp_raw(args, data, dp_size=2)[1]
    store = RayObjectStore(frees_objects=True)
    ref = store.put(shard, value_spec=ROLLOUT_DATA_VALUE_SPEC)
    with store.get(ref) as fetched:
        fetched = dict(fetched)
        fetched.pop("partition")
        materialize_replay_columns(args, fetched)
        assert fetched["rollout_topk_lengths"][0].tolist() == [2, 2, 0, 2]
        assert "rollout_topk_token_ids" not in fetched
        batch = DataIterator(fetched, micro_batch_size=1).get_next(TRAIN_KEYS + SAMPLING_MASK_KEYS)
    store.remove(ref)
    assert batch["rollout_topk_token_ids"] is None
    response = np.asarray(restored.tokens[-restored.response_length :], dtype=np.int64)
    pieces = restored.replay_index.sampling_mask.mask_pieces({})
    dense_ids, _ = derive_support_topk(derive_sampling_mask(pieces, response), k=3)
    batch.update(
        total_lengths=[len(restored.tokens)],
        response_lengths=[restored.response_length],
        unconcat_tokens=[torch.tensor(restored.tokens)],
        loss_masks=[torch.tensor(restored.loss_mask)],
        rollout_log_probs=[torch.tensor(restored.rollout_log_probs)],
        advantages=[torch.ones(restored.response_length)],
        ref_log_probs=[torch.full((restored.response_length,), -1.0)],
    )
    dense_batch = dict(batch, rollout_topk_token_ids=[dense_ids])
    logits = torch.randn(1, len(restored.tokens), 10, generator=torch.Generator().manual_seed(49))
    outputs = []
    mask = torch.tensor(restored.loss_mask)
    for current in (batch, dense_batch):
        scores = logits.clone().requires_grad_()
        loss, metrics = score_centering_loss_function(args, current, scores, lambda v: (v * mask).sum() / mask.sum())
        outputs.append((loss, torch.autograd.grad(loss, scores)[0], metrics))
    torch.testing.assert_close(outputs[0], outputs[1])


def _session_turn(prompt: list[int], output: list[int]) -> Sample:
    return Sample(
        tokens=prompt + output,
        response_length=len(output),
        response=str(output),
        replay_index=_support_mask([output] * len(output), [[0.5, 0.5]] * len(output)),
        rollout_log_probs=[math.log(0.5)] * len(output),
        loss_mask=[1] * len(output),
        status=Sample.Status.COMPLETED,
        index=0,
        group_index=0,
        reward=1.0,
    )
