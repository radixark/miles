"""Filtered rollout probabilities through candidate transport and the real loss."""

import math
from copy import deepcopy

import numpy as np
import pytest
import torch
from tests.fast.fixtures.score_centering_fixtures import _args, _Tokenizer, _turn

from miles.backends.training_utils import parallel
from miles.backends.training_utils.data.rollout import DataIterator
from miles.backends.training_utils.loss.hub.score_centering_loss import score_centering_loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState
from miles.backends.training_utils.torch_native.actor import SAMPLING_MASK_KEYS, TRAIN_KEYS
from miles.ray.rollout.train_data_conversion import (
    ROLLOUT_DATA_VALUE_SPEC,
    convert_samples_to_train_data,
    split_train_data_by_dp_raw,
)
from miles.rollout.generate_utils.rollout_topk_logprobs import (
    append_rollout_topk_logprobs,
    configure_rollout_topk_logprobs_request,
    validate_rollout_topk_logprobs_sample,
)
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.session.samples.codec import (
    COMPUTED_FIELDS,
    ROLLOUT_SAMPLING_MASK_FIELDS,
    decode_samples_and_merge_input_sample,
    encode_samples,
)
from miles.utils.object_store import RayObjectStore
from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.score_centering import validate_score_centering_args
from miles.utils.types import Sample


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
    sample = Sample(
        tokens=[0, 1, 2],
        response_length=1,
        response="2",
        rollout_log_probs=[math.log(4 / 7)],
        rollout_sampling_mask=RolloutSamplingMask.from_mask_list([[3, 2]]),
        loss_mask=[1],
        status=Sample.Status.COMPLETED,
        index=0,
        group_index=0,
        reward=1.0,
    )
    append_rollout_topk_logprobs(
        sample,
        {
            "output_token_logprobs": [(math.log(0.4), 2, None)],
            "output_token_sampling_mask": [[3, 2]],
            "output_token_sampling_logprobs": [[math.log(3 / 7), math.log(4 / 7)]],
        },
        3,
        sampling_logprobs_mode="support",
    )
    return sample


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
@pytest.mark.parametrize("ci_test", [False, True])
def test_filtered_loss_matches_dense_support_gradient(single_rank: None, mode: str, ci_test: bool) -> None:
    sample = _filtered_sample()
    validate_rollout_topk_logprobs_sample(sample, 3)
    np.testing.assert_array_equal(sample.rollout_topk_token_ids, [[3, 2, -1]])
    np.testing.assert_allclose(np.exp(sample.rollout_topk_log_probs[0, :2]), [3 / 7, 4 / 7])
    args = _args(
        ci_test=ci_test,
        score_centering_is=mode,
        rollout_top_p=0.6,
        rollout_top_k=3,
        rollout_sampling_logprobs_mode="support",
        use_sampling_support_replay=True,
        entropy_coef=0.03,
    )
    validate_score_centering_args(args)
    data = convert_samples_to_train_data(args, [sample], {}, None, None)
    assert data["rollout_sampling_mask_ids"][0].tolist() == [3, 2]
    batch = {
        "unconcat_tokens": [torch.tensor(sample.tokens)],
        "total_lengths": [len(sample.tokens)],
        "response_lengths": [sample.response_length],
        "loss_masks": [torch.tensor(sample.loss_mask)],
        "rollout_log_probs": [torch.tensor(sample.rollout_log_probs)],
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
    sample = _filtered_sample()
    sample.rollout_topk_token_ids[0, 1] = -1
    sample.rollout_topk_log_probs[0, 1] = -np.inf
    args = _args(ci_test=ci_test)
    if ci_test:
        with pytest.raises(ValueError, match="must equal the sampling support"):
            convert_samples_to_train_data(args, [sample], {}, None, None)
    else:
        data = convert_samples_to_train_data(args, [sample], {}, None, None)
        np.testing.assert_array_equal(data["rollout_topk_token_ids"][0], sample.rollout_topk_token_ids)
        np.testing.assert_array_equal(data["rollout_topk_log_probs"][0], sample.rollout_topk_log_probs)


def test_missing_support_candidate_fails_at_generation() -> None:
    sample = Sample(
        tokens=[0, 1, 2],
        response_length=1,
        rollout_sampling_mask=RolloutSamplingMask.from_mask_list([[2, 5, 7, 8]]),
    )
    meta = {
        "output_token_logprobs": [(math.log(0.4), 2, None)],
        "output_token_sampling_mask": [[2, 5, 7, 8]],
        "output_token_sampling_logprobs": [[math.log(0.4), math.log(0.3), math.log(0.2), math.log(0.1)]],
    }
    with pytest.raises(ValueError, match="support exceeds"):
        append_rollout_topk_logprobs(sample, meta, 3, sampling_logprobs_mode="support")


def test_missing_support_probabilities_fail_closed() -> None:
    sample = Sample(
        tokens=[0, 1, 2],
        response_length=1,
        rollout_sampling_mask=RolloutSamplingMask.from_mask_list([[2, 3]]),
    )
    meta = {"output_token_logprobs": [(math.log(0.4), 2, None)], "output_token_sampling_mask": [[2, 3]]}
    with pytest.raises(ValueError, match="output_token_sampling_logprobs"):
        append_rollout_topk_logprobs(sample, meta, 3, sampling_logprobs_mode="support")


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
    first, second = _filtered_sample(), _filtered_sample()
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
    data = convert_samples_to_train_data(
        _args(rollout_sampling_logprobs_mode="support"), [first, second], {}, None, None
    )
    if layout == "ordered":
        assert "rollout_topk_token_ids" not in data
        assert [value.tolist() for value in data["rollout_topk_lengths"]] == [[2], [2]]
        assert all(value.dtype == torch.int32 for value in data["rollout_topk_lengths"])
    else:
        assert "rollout_topk_lengths" not in data
        np.testing.assert_array_equal(data["rollout_topk_token_ids"][1], second.rollout_topk_token_ids)


@pytest.mark.parametrize(
    "loss_type,mode,k",
    [("policy_loss", "support", 3), ("score_centering", "selected", 3), ("policy_loss", "support", 0)],
)
def test_unrelated_conversion_keeps_existing_representation(loss_type, mode, k):
    data = convert_samples_to_train_data(
        _args(loss_type=loss_type, rollout_sampling_logprobs_mode=mode, rollout_top_logprobs_num=k),
        [_filtered_sample()],
        {},
        None,
        None,
    )
    assert "rollout_topk_lengths" not in data
    assert ("rollout_topk_token_ids" in data) == bool(k)


@pytest.mark.parametrize("use_kl", [False, True])
def test_compact_session_object_store_and_iterator(single_rank, ray_local_mode, use_kl):
    first = _turn([0, 1], [2, 3], [0.5, 0.5])
    second = _turn([0, 1, 2, 3, 6], [4, 5], [0.55, 0.45])
    first.rollout_sampling_mask = RolloutSamplingMask.from_mask_list([[2, 3], [2, 3]])
    second.rollout_sampling_mask = RolloutSamplingMask.from_mask_list([[4, 5], [4, 5]])
    merged = merge_samples([first, second], _Tokenizer())
    merged.strip_last_output_tokens(1, _Tokenizer())
    fields = COMPUTED_FIELDS + ROLLOUT_SAMPLING_MASK_FIELDS
    wire = encode_samples([merged], {}, fields=fields)
    restored = decode_samples_and_merge_input_sample(wire, Sample(), fields=fields).samples[0]
    restored.index, restored.reward = 0, 1.0
    restored.loss_mask[1] = 0
    other = deepcopy(restored)
    other.index = 1
    args = _args(
        rollout_sampling_logprobs_mode="support",
        use_sampling_support_replay=True,
        ci_test=True,
        use_kl_loss=use_kl,
        use_unbiased_kl=True,
        kl_loss_type="k2",
        kl_loss_coef=0.1,
        entropy_coef=0.03,
    )
    data = convert_samples_to_train_data(args, [restored, other], {}, None, None)
    assert data["rollout_topk_lengths"][0].tolist() == [2, 2, 0, 2]
    assert "rollout_topk_token_ids" not in data
    shard = split_train_data_by_dp_raw(args, data, dp_size=2)[1]
    store = RayObjectStore(frees_objects=True)
    ref = store.put(shard, value_spec=ROLLOUT_DATA_VALUE_SPEC)
    with store.get(ref) as fetched:
        batch = DataIterator(fetched, micro_batch_size=1).get_next(TRAIN_KEYS + SAMPLING_MASK_KEYS)
    store.remove(ref)
    assert batch["rollout_topk_token_ids"] is None
    batch.update(
        total_lengths=[len(restored.tokens)],
        response_lengths=[restored.response_length],
        unconcat_tokens=[torch.tensor(restored.tokens)],
        loss_masks=[torch.tensor(restored.loss_mask)],
        rollout_log_probs=[torch.tensor(restored.rollout_log_probs)],
        advantages=[torch.ones(restored.response_length)],
        ref_log_probs=[torch.full((restored.response_length,), -1.0)],
    )
    dense_batch = dict(batch, rollout_topk_token_ids=[restored.rollout_topk_token_ids])
    logits = torch.randn(1, len(restored.tokens), 10, generator=torch.Generator().manual_seed(49))
    outputs = []
    mask = torch.tensor(restored.loss_mask)
    for current in (batch, dense_batch):
        scores = logits.clone().requires_grad_()
        loss, metrics = score_centering_loss_function(args, current, scores, lambda v: (v * mask).sum() / mask.sum())
        outputs.append((loss, torch.autograd.grad(loss, scores)[0], metrics))
    torch.testing.assert_close(outputs[0], outputs[1])
