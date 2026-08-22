import pytest
import torch

from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss_hub.losses import tinker_loss_function

from .loss_test_utils import make_args, make_inputs, make_parallel_state

VOCAB = 32


def make_batch(seed=7, prompt_lens=(4, 6), response_lens=(3, 5)):
    make_parallel_state()
    args = make_args(loss_type="custom_loss")
    inputs = make_inputs(
        seed=seed,
        batch_size=len(prompt_lens),
        prompt_lens=list(prompt_lens),
        response_lens=list(response_lens),
        vocab_size=VOCAB,
        args=args,
    )
    batch = dict(
        unconcat_tokens=inputs["unconcat_tokens"],
        total_lengths=inputs["total_lens"],
        response_lengths=list(response_lens),
        loss_masks=[torch.ones(rl, dtype=torch.int32) for rl in response_lens],
        rollout_log_probs=inputs["rollout_log_probs"],
        tinker_operation_lanes=[0] * len(prompt_lens),
        tinker_loss_by_lane={0: {"loss_fn": "cross_entropy"}},
    )
    return args, batch, inputs["policy_logits"].requires_grad_(True)


def reference_log_probs(args, batch, logits):
    return get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )["log_probs"]


def run(args, batch, logits):
    loss, metrics = tinker_loss_function(args, batch, logits, sum_of_sample_mean=None)
    return loss, metrics


def test_linear_cross_entropy_is_a_plain_weighted_sum():
    args, batch, logits = make_batch()
    weights = [torch.tensor([0.5, 0.0, 2.0]), torch.tensor([1.0, 1.0, 0.0, -1.0, 0.25])]
    batch["loss_weights"] = weights

    loss, metrics = run(args, batch, logits)
    expected = sum(-(lp * w).sum() for lp, w in zip(reference_log_probs(args, batch, logits), weights, strict=True))
    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["loss"], expected)
    assert loss.requires_grad


def test_binary_mask_still_gates_tokens():
    args, batch, logits = make_batch()
    batch["loss_weights"] = [torch.ones(3), torch.ones(5)]
    batch["loss_masks"] = [torch.tensor([1, 0, 1], dtype=torch.int32), torch.zeros(5, dtype=torch.int32)]

    loss, _ = run(args, batch, logits)
    lp = reference_log_probs(args, batch, logits)
    expected = -(lp[0] * torch.tensor([1.0, 0.0, 1.0])).sum()
    assert torch.allclose(loss, expected)


def test_importance_sampling_and_ppo_clip():
    args, batch, logits = make_batch()
    advantages = [torch.tensor([1.0, -1.0, 2.0]), torch.tensor([0.5, 0.5, -0.5, 1.0, 0.0])]
    batch["advantages"] = advantages
    batch["tinker_loss_by_lane"] = {0: {"loss_fn": "importance_sampling"}}

    loss, _ = run(args, batch, logits)
    lp = reference_log_probs(args, batch, logits)
    ratios = [torch.exp(new - old) for new, old in zip(lp, batch["rollout_log_probs"], strict=True)]
    expected = sum(-(r * a).sum() for r, a in zip(ratios, advantages, strict=True))
    assert torch.allclose(loss, expected)

    batch["tinker_loss_by_lane"] = {
        0: {"loss_fn": "ppo", "loss_fn_config": {"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}}
    }
    loss_ppo, _ = run(args, batch, logits)
    expected_ppo = sum(
        -torch.minimum(r * a, r.clamp(0.9, 1.1) * a).sum() for r, a in zip(ratios, advantages, strict=True)
    )
    assert torch.allclose(loss_ppo, expected_ppo)
    # Ensure these logits exercise the clipped branch.
    assert not torch.allclose(loss_ppo, loss)


def test_mixed_lanes_dispatch_independently():
    args, batch, logits = make_batch()
    batch["tinker_operation_lanes"] = [0, 1]
    batch["loss_weights"] = [torch.ones(3), torch.zeros(5)]
    batch["advantages"] = [torch.zeros(3), torch.ones(5)]
    batch["tinker_loss_by_lane"] = {
        0: {"loss_fn": "cross_entropy"},
        1: {"loss_fn": "importance_sampling"},
    }

    loss, _ = run(args, batch, logits)
    lp = reference_log_probs(args, batch, logits)
    ratio = torch.exp(lp[1] - batch["rollout_log_probs"][1])
    expected = -(lp[0].sum()) + -(ratio.sum())
    assert torch.allclose(loss, expected)


def test_sum_reduction_is_chunk_additive():
    # The same data as one batch vs two single-sample batches must produce the
    # same total loss — the invariant that makes K forward_backward operations
    # accumulate identically to one.
    args, batch, logits = make_batch()
    batch["loss_weights"] = [torch.ones(3) * 0.5, torch.ones(5) * 1.5]
    full_loss, _ = run(args, batch, logits)

    total = 0.0
    offset = 0
    for i, total_len in enumerate(batch["total_lengths"]):
        sub_logits = logits[:, offset : offset + total_len]
        sub = dict(
            unconcat_tokens=[batch["unconcat_tokens"][i]],
            total_lengths=[total_len],
            response_lengths=[batch["response_lengths"][i]],
            loss_masks=[batch["loss_masks"][i]],
            loss_weights=[batch["loss_weights"][i]],
            tinker_operation_lanes=[0],
            tinker_loss_by_lane=batch["tinker_loss_by_lane"],
        )
        sub_loss, _ = run(args, sub, sub_logits)
        total += sub_loss
        offset += total_len
    assert torch.allclose(full_loss, total)


def test_zero_weight_padding_contributes_nothing():
    # DP padding duplicates a sample with all-zero loss_weights; the padded
    # row must not move the loss.
    args, batch, logits = make_batch()
    batch["loss_weights"] = [torch.ones(3), torch.zeros(5)]
    loss, _ = run(args, batch, logits)
    lp = reference_log_probs(args, batch, logits)
    assert torch.allclose(loss, -(lp[0].sum()))


def test_missing_channel_missing_spec_and_unknown_loss_fail_loudly():
    args, batch, logits = make_batch()
    with pytest.raises(ValueError, match="needs per-token 'loss_weights'"):
        run(args, batch, logits)

    batch["loss_weights"] = [torch.ones(3), torch.ones(5)]
    batch["tinker_operation_lanes"] = [0, 3]
    with pytest.raises(ValueError, match="no loss spec for lane 3"):
        run(args, batch, logits)

    batch["tinker_operation_lanes"] = [0, 0]
    batch["tinker_loss_by_lane"] = {0: {"loss_fn": "dro"}}
    with pytest.raises(ValueError, match="unknown loss_fn 'dro'"):
        run(args, batch, logits)


def test_collector_captures_per_datum_logprobs_in_row_order():
    args, batch, logits = make_batch()
    batch["loss_weights"] = [torch.ones(3), torch.ones(5)]
    batch["sample_indices"] = [0, 1]
    collector: dict = {}
    batch["tinker_logprob_collector"] = collector

    run(args, batch, logits)
    lp = reference_log_probs(args, batch, logits)
    assert set(collector) == {(0, 0), (0, 1)}
    assert collector[(0, 0)] == pytest.approx(lp[0].tolist())
    assert collector[(0, 1)] == pytest.approx(lp[1].tolist())


def test_forward_only_batch_collects_logprobs_without_client_loss_terms():
    # Homogeneous selections: an all-forward batch never mixes with backward
    # rows; it needs no channels, fills the collector, and its dummy loss is
    # never backwarded (the executor runs forward_only=True).
    args, batch, logits = make_batch()
    batch["tinker_operation_lanes"] = [0, 1]
    batch["tinker_loss_by_lane"] = {}
    batch["tinker_forward_only"] = True
    batch["sample_indices"] = [0, 0]
    collector: dict = {}
    batch["tinker_logprob_collector"] = collector

    loss, metrics = run(args, batch, logits)
    lp = reference_log_probs(args, batch, logits)
    assert loss.item() == 0.0 and metrics["loss"].item() == 0.0
    assert collector[(0, 0)] == pytest.approx(lp[0].tolist())
    assert collector[(1, 0)] == pytest.approx(lp[1].tolist())
