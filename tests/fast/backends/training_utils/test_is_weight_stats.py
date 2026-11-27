import math
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.utils.checkpoint

from miles.backends.training_utils.loss import is_weight_stats
from miles.backends.training_utils.loss.hub import losses as loss_utils
from miles.backends.training_utils.loss.is_weight_stats import IsWeightStats


def _single_rank_all_reduce(tensor, groups, op):
    assert op in (dist.ReduceOp.SUM, dist.ReduceOp.MAX)


def _reduce(stats: IsWeightStats) -> dict[str, float]:
    return stats.reduce_and_reset([], _single_rank_all_reduce)


def _exact_fresh_weights(seed: int = 0):
    """Fresh tokens carry weight exactly 1; stale ones drift, with one large outlier."""
    rng = np.random.default_rng(seed)
    fresh = torch.ones(1000)
    stale = torch.from_numpy(np.exp(rng.normal(0.0, 0.05, size=999))).float()
    stale = torch.cat([stale, torch.tensor([50.0])])
    raw = torch.cat([fresh, stale])
    fresh_mask = torch.cat([torch.ones(1000), torch.zeros(1000)])
    return raw, fresh_mask, 1 - fresh_mask, stale


def test_untruncated_weights_report_their_tail_split_by_version():
    raw, fresh, stale, stale_weights = _exact_fresh_weights()
    stats = IsWeightStats()
    stats.add(raw=raw, active=torch.ones_like(raw), fresh=fresh, stale=stale)
    metrics = _reduce(stats)

    assert metrics["is/all_count"] == 2000
    assert metrics["is/fresh_count"] == 1000
    assert metrics["is/fresh_max"] == 1.0
    assert metrics["is/fresh_p99_9"] == 1.0
    assert metrics["is/fresh_ess_ratio_raw"] == pytest.approx(1.0)
    assert metrics["is/stale_max"] == pytest.approx(50.0)
    for name, q in is_weight_stats.QUANTILES.items():
        expected = float(np.quantile(stale_weights.numpy(), q, method="inverted_cdf"))
        assert metrics[f"is/stale_{name}"] == pytest.approx(expected, rel=0.006)
    w = stale_weights.double()
    assert metrics["is/stale_ess_ratio_raw"] == pytest.approx(float(w.sum() ** 2 / (len(w) * (w * w).sum())))
    # nothing was clipped, so the applied weights are the raw ones
    assert metrics["is/all_clipfrac"] == 0.0
    assert metrics["is/stale_ess_ratio"] == metrics["is/stale_ess_ratio_raw"]


def test_inactive_tokens_and_a_second_microbatch():
    stats = IsWeightStats()
    stats.add(raw=torch.tensor([3.0, 100.0]), active=torch.tensor([1.0, 0.0]))
    stats.add(raw=torch.tensor([0.5]), active=torch.tensor([1.0]))
    metrics = _reduce(stats)
    assert metrics["is/all_count"] == 2
    assert metrics["is/all_max"] == pytest.approx(3.0)
    assert metrics["is/fresh_count"] == 0
    assert "is/fresh_max" not in metrics
    # the collector starts over after a reduce
    assert _reduce(stats)["is/all_count"] == 0


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_a_checkpointed_loss_counts_each_microbatch_once(use_reentrant):
    """--recompute-loss-function reruns the loss inside backward; its rerun must not add the microbatch again."""
    stats = IsWeightStats()
    x = torch.tensor([0.5, 2.0, 1.0], requires_grad=True)

    def loss(x):
        stats.add(raw=torch.exp(x.detach() - 1.0), active=torch.ones(3))
        return (x * x).sum()

    torch.utils.checkpoint.checkpoint(loss, x, use_reentrant=use_reentrant).backward()
    assert _reduce(stats)["is/all_count"] == 3


def test_clipping_shows_in_clipfrac_and_applied_ess():
    raw = torch.tensor([0.5, 1.0, 1.5, 4.0])
    applied = raw.clamp(min=0.0, max=2.0)
    stats = IsWeightStats()
    stats.add(raw=raw, active=torch.ones(4), applied=applied)
    metrics = _reduce(stats)
    assert metrics["is/all_clipfrac"] == 0.25
    a = applied.double()
    assert metrics["is/all_ess_ratio"] == pytest.approx(float(a.sum() ** 2 / (4 * (a * a).sum())))
    assert metrics["is/all_ess_ratio"] > metrics["is/all_ess_ratio_raw"]


def _loss_args(**overrides) -> Namespace:
    args = Namespace(
        use_rollout_logprobs=False,
        use_sampling_support_replay=False,
        skip_actor_forward_only=True,
        use_opsm=False,
        advantage_estimator="grpo",
        get_mismatch_metrics=False,
        use_tis=True,
        tis_clip=math.inf,
        tis_clip_low=0.0,
        eps_clip=0.2,
        eps_clip_high=0.28,
        custom_tis_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        calculate_per_token_loss=False,
        qkv_format="thd",
        entropy_coef=0.0,
        use_kl_loss=False,
        use_unbiased_kl=False,
        kl_loss_type="k1",
        kl_loss_coef=0.0,
        rollout_temperature=1.0,
        log_probs_chunk_size=-1,
        true_on_policy_mode=True,
        allgather_cp=False,
        observe_training_entropy=False,
        eps_clip_c=None,
        dump_details=None,
        fully_async=False,
        loss_type="policy_loss",
        train_backend="megatron",
        backend=Namespace(calculate_per_token_loss=False, vocab_size=None),
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _patch_loss_rows(monkeypatch, train):
    """One sample, no CP: the response layout's rows are the sample's, scored as ``train``."""
    from miles.backends.training_utils.loss.hub import response_layout

    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(tp=SimpleNamespace(group=None)))
    monkeypatch.setattr(
        response_layout,
        "get_local_response_loss_masks",
        lambda total_lengths, response_lengths, loss_masks, qkv_format="thd", max_seq_lens=None: loss_masks,
    )
    monkeypatch.setattr(
        response_layout, "compute_ess_ratio_contribution", lambda *, ppo_kl, **kwargs: ppo_kl.new_tensor(1.0)
    )
    monkeypatch.setattr(
        response_layout, "get_sum_of_sample_mean", lambda *a, **k: (lambda tensor: tensor.float().mean())
    )
    monkeypatch.setattr(
        response_layout,
        "get_log_probs_and_entropy",
        lambda *a, **k: {"log_probs": [train.clone().requires_grad_()]},
    )


@pytest.mark.parametrize("tis_clip", [math.inf, 1.2])
def test_the_policy_loss_feeds_the_trainer_over_rollout_weight(monkeypatch, tis_clip):
    """Fresh tokens score exactly what the rollout sampled (w = 1); the stale token drifted by log 0.3."""
    train = torch.tensor([-1.0, -2.0, -0.5])
    rollout = torch.tensor([-1.0, -2.0, -0.8])
    batch = {
        "advantages": [torch.tensor([1.0, -0.5, 0.25])],
        "rollout_log_probs": [rollout],
        "unconcat_tokens": [torch.tensor([7, 8, 9, 10])],
        "response_lengths": [3],
        "total_lengths": [4],
        "loss_masks": [torch.ones(3)],
        "rollout_token_fresh": [torch.tensor([1.0, 1.0, 0.0])],
        "rollout_token_stale": [torch.tensor([0.0, 0.0, 1.0])],
    }
    _patch_loss_rows(monkeypatch, train)
    is_weight_stats.IS_WEIGHT_STATS.reset()

    loss_utils.policy_loss_function(
        _loss_args(tis_clip=tis_clip),
        batch,
        logits=torch.zeros((1, 4, 8)),
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )
    metrics = _reduce(is_weight_stats.IS_WEIGHT_STATS)

    assert metrics["is/fresh_count"] == 2
    assert metrics["is/fresh_max"] == 1.0
    assert metrics["is/stale_count"] == 1
    assert metrics["is/stale_max"] == pytest.approx(math.exp(0.3), rel=1e-6)
    assert metrics["is/stale_clipfrac"] == (0.0 if tis_clip == math.inf else 1.0)
    assert metrics["is/fresh_clipfrac"] == 0.0


@pytest.mark.parametrize("old_shift", [0.0, 0.25])
def test_the_recomputed_old_policy_is_compared_with_the_training_forward(monkeypatch, old_shift):
    """With a forward-only old policy, the loss reports |old - train|; 0 exactly when the two forwards agree."""
    train = torch.tensor([-1.0, -2.0, -0.5])
    batch = {
        "advantages": [torch.tensor([1.0, -0.5, 0.25])],
        "log_probs": [train + old_shift],
        "rollout_log_probs": [torch.tensor([-1.0, -2.0, -0.8])],
        "unconcat_tokens": [torch.tensor([7, 8, 9, 10])],
        "response_lengths": [3],
        "total_lengths": [4],
        "loss_masks": [torch.ones(3)],
    }
    _patch_loss_rows(monkeypatch, train)
    is_weight_stats.IS_WEIGHT_STATS.reset()

    _, metrics = loss_utils.policy_loss_function(
        _loss_args(skip_actor_forward_only=False, use_tis=True, fully_async=True),
        batch,
        logits=torch.zeros((1, 4, 8)),
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )

    assert float(metrics["old_policy_logprob_abs_diff"]) == pytest.approx(old_shift)
