"""Exercise candidate transport through the real rollout and training consumers."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from tests.fast.fixtures.score_centering_fixtures import _args, _Tokenizer, _turn

from miles.backends.training_utils import parallel
from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss import compute_advantages_and_returns, loss_function
from miles.backends.training_utils.loss_hub.losses import get_loss_function
from miles.backends.training_utils.loss_hub.score_centering import score_centering_loss
from miles.backends.training_utils.parallel import GroupInfo, ParallelState
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp_raw
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample, encode_samples
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
