"""Tests for the ``--loss-aggregation`` pg_loss modes."""

from types import SimpleNamespace

import pytest
import torch


def _parallel_state(miles, *, cp_size: int, cp_rank: int = 0):
    return miles.ParallelState(
        intra_dp=miles.GroupInfo(rank=0, size=1, group=None),
        intra_dp_cp=miles.GroupInfo(rank=0, size=cp_size, group=None),
        cp=miles.GroupInfo(rank=cp_rank, size=cp_size, group=None),
        tp=miles.GroupInfo(rank=0, size=1, group=None),
        pp=miles.GroupInfo(rank=0, size=1, group=None),
        ep=miles.GroupInfo(rank=0, size=1, group=None),
        etp=miles.GroupInfo(rank=0, size=1, group=None),
        indep_dp=miles.GroupInfo(rank=0, size=1, group=None),
        cp_comm_type=None,
    )


def _legacy_sum_of_sample_mean(response_lengths, loss_masks):
    def reducer(x: torch.Tensor) -> torch.Tensor:
        return sum(
            (x_i * m_i).sum() / torch.clamp_min(m_i.sum(), 1)
            for x_i, m_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=True)
        )

    return reducer


RESPONSE_LENGTHS = [3, 3, 4, 4]
TOTAL_LENGTHS = [3, 3, 4, 4]
LOSS_MASKS = [
    torch.tensor([1.0, 1.0, 0.0]),
    torch.tensor([1.0, 1.0, 1.0]),
    torch.tensor([1.0, 0.0, 0.0, 0.0]),
    torch.tensor([1.0, 1.0, 1.0, 1.0]),
]
PROMPT_GROUP_INDICES = [0, 0, 1, 1]
X = torch.arange(1.0, 15.0)


def test_default_is_byte_identical_to_legacy_reducer(miles, monkeypatch):
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: _parallel_state(miles, cp_size=1))

    new = miles.get_sum_of_sample_mean(TOTAL_LENGTHS, RESPONSE_LENGTHS, LOSS_MASKS)
    legacy = _legacy_sum_of_sample_mean(RESPONSE_LENGTHS, LOSS_MASKS)

    torch.testing.assert_close(new(X), legacy(X), rtol=0, atol=0)


def test_prompt_mean_denominator_is_group_token_total(miles, monkeypatch):
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: _parallel_state(miles, cp_size=1))

    sample_denoms = [torch.tensor(5.0)] * 4
    reducer = miles.get_sum_of_sample_mean(TOTAL_LENGTHS, RESPONSE_LENGTHS, LOSS_MASKS, sample_denoms=sample_denoms)
    expected = (3 + 15) / 5 + (7 + 50) / 5
    torch.testing.assert_close(reducer(X), torch.tensor(expected))


def test_constant_divides_summed_token_loss_by_L(miles, monkeypatch):
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: _parallel_state(miles, cp_size=1))

    reducer = miles.get_sum_of_sample_mean(TOTAL_LENGTHS, RESPONSE_LENGTHS, LOSS_MASKS, constant_divisor=10.0)
    torch.testing.assert_close(reducer(X), torch.tensor(7.5))


def test_token_mean_is_unnormalized_global_token_sum(miles, monkeypatch):
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: _parallel_state(miles, cp_size=1))

    reducer = miles.get_sum_of_sample_mean(TOTAL_LENGTHS, RESPONSE_LENGTHS, LOSS_MASKS, calculate_per_token_loss=True)
    torch.testing.assert_close(reducer(X), torch.tensor(75.0))


def test_active_token_count_does_not_count_fully_masked_samples(miles):
    count = miles.loss.get_active_token_count([torch.zeros(2, dtype=torch.int32), torch.ones(2, dtype=torch.int32)])

    assert count.dtype == torch.int64
    torch.testing.assert_close(count, torch.tensor(2))


def test_token_counts_are_computed_at_optimizer_step_scope(miles):
    counts = miles.loss.get_token_counts_by_step(
        [torch.zeros(2), torch.ones(2), torch.ones(1), torch.ones(3)],
        num_steps=2,
    )

    torch.testing.assert_close(counts, torch.tensor([2.0, 4.0]))


def test_fsdp_scaling_converts_rank_local_numerators_to_global_token_mean(miles):
    global_num_tokens = torch.tensor(5.0)
    scaled_local_losses = [
        miles.loss.scale_data_parallel_token_mean_loss(
            loss,
            global_num_tokens=global_num_tokens,
            data_parallel_size=2,
        )
        for loss in (torch.tensor(6.0), torch.tensor(14.0))
    ]

    # FSDP averages gradients across DP ranks after each local loss is scaled.
    torch.testing.assert_close(sum(scaled_local_losses) / 2, torch.tensor(4.0))


def test_legacy_positional_per_token_loss_call_still_works(miles, monkeypatch):
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: _parallel_state(miles, cp_size=1))

    reducer = miles.get_sum_of_sample_mean(TOTAL_LENGTHS, RESPONSE_LENGTHS, LOSS_MASKS, True)
    torch.testing.assert_close(reducer(X), torch.tensor(75.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"calculate_per_token_loss": True, "sample_denoms": [torch.tensor(1.0)] * 4},
        {"calculate_per_token_loss": True, "constant_divisor": 4.0},
        {"sample_denoms": [torch.tensor(1.0)] * 4, "constant_divisor": 4.0},
    ],
)
def test_reducer_rejects_multiple_normalization_modes(miles, kwargs):
    with pytest.raises(ValueError, match="mutually exclusive"):
        miles.get_sum_of_sample_mean(TOTAL_LENGTHS, RESPONSE_LENGTHS, LOSS_MASKS, **kwargs)


@pytest.mark.parametrize("constant_divisor", [None, 20.0])
def test_cp_zigzag_rank_sum_matches_single_rank(miles, monkeypatch, constant_divisor):
    total_length, response_length = 10, 8
    loss_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    x_full = torch.arange(1.0, 9.0)
    sample_denoms = None if constant_divisor is not None else [torch.tensor(20.0)]

    def build():
        return miles.get_sum_of_sample_mean(
            [total_length],
            [response_length],
            [loss_mask],
            constant_divisor=constant_divisor,
            sample_denoms=sample_denoms,
        )

    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: _parallel_state(miles, cp_size=1))
    ref = build()(x_full)

    total = torch.zeros(())
    for rank in range(2):
        monkeypatch.setattr(
            miles.cp_utils,
            "get_parallel_state",
            lambda r=rank: _parallel_state(miles, cp_size=2, cp_rank=r),
        )
        x_local = miles.cp_utils._slice_loss_mask_for_local_cp(total_length, response_length, x_full, "thd", None)
        total = total + build()(x_local)

    torch.testing.assert_close(total, ref)


def _args(**overrides):
    base = dict(
        loss_aggregation="sample_mean",
        loss_aggregation_divisor=None,
        calculate_per_token_loss=False,
        loss_type="policy_loss",
        n_samples_per_prompt=2,
        qkv_format="thd",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _default_reducer(miles, *, calculate_per_token_loss=False):
    return miles.get_sum_of_sample_mean(
        TOTAL_LENGTHS,
        RESPONSE_LENGTHS,
        LOSS_MASKS,
        calculate_per_token_loss=calculate_per_token_loss,
    )


def _select(miles, args, batch, *, default_reducer=None, pg_loss_masks=LOSS_MASKS):
    batch = dict(batch)
    batch.setdefault("loss_masks", LOSS_MASKS)
    if default_reducer is None:
        default_reducer = _default_reducer(miles)
    return miles.get_pg_loss_reducer(
        args,
        batch,
        total_lengths=TOTAL_LENGTHS,
        response_lengths=RESPONSE_LENGTHS,
        pg_loss_masks=pg_loss_masks,
        max_seq_lens=None,
        default_reducer=default_reducer,
    )


@pytest.mark.parametrize(
    ("args", "batch", "expected"),
    [
        (_args(loss_aggregation="sample_mean"), {}, 26.0),
        (_args(loss_aggregation="token_mean", calculate_per_token_loss=True), {}, 75.0),
        (_args(loss_aggregation="constant", loss_aggregation_divisor=10.0), {}, 7.5),
        (
            _args(loss_aggregation="prompt_mean"),
            {"prompt_group_indices": PROMPT_GROUP_INDICES, "prompt_mask_sums": [torch.tensor(5.0)] * 4},
            30.0,
        ),
    ],
)
def test_pg_loss_reducer_modes_compute_expected_loss(miles, monkeypatch, args, batch, expected):
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: _parallel_state(miles, cp_size=1))

    reducer = _select(miles, args, batch)
    torch.testing.assert_close(reducer(X), torch.tensor(expected))


def test_prompt_mean_without_prompt_mask_sums_fails(miles):
    with pytest.raises(ValueError, match="prompt_mask_sums"):
        _select(miles, _args(loss_aggregation="prompt_mean"), {}, default_reducer=lambda x: x)


def test_token_mean_log_aggregation_keeps_metric_sample_mean(miles, monkeypatch):
    state = _parallel_state(miles, cp_size=1)
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(miles.loss, "get_parallel_state", lambda: state)
    monkeypatch.setattr(miles.log_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(miles.log_utils.MultiPGUtil, "all_reduce", lambda *args, **kwargs: None)

    args = _args(loss_aggregation="token_mean", calculate_per_token_loss=True)
    args.global_batch_size = 2
    args.use_dynamic_global_batch_size = False
    args.recompute_loss_function = False
    args.true_on_policy_mode = False

    batch = {
        "loss_masks": [torch.ones(2), torch.ones(3)],
        "total_lengths": [2, 3],
        "response_lengths": [2, 3],
    }

    def fake_loss_function(args, batch, logits, sum_of_sample_mean):
        per_token_metric = torch.arange(1.0, 6.0, device=logits.device)
        return logits.sum() * 0, {
            "pg_loss": per_token_metric.sum(),
            "ppo_kl": sum_of_sample_mean(per_token_metric),
        }

    monkeypatch.setattr(miles.loss, "get_loss_function", lambda args: fake_loss_function)

    _, _, log_dict = miles.loss.loss_function(args, batch, 1, torch.ones((), requires_grad=True))
    loss_dict = miles.log_utils.aggregate_train_losses([log_dict])

    torch.testing.assert_close(torch.tensor(loss_dict["pg_loss"]), torch.tensor(3.0))
    torch.testing.assert_close(torch.tensor(loss_dict["ppo_kl"]), torch.tensor(2.75))


def test_build_train_log_dict_carries_per_metric_normalizers(miles):
    log_dict = miles.loss._build_train_log_dict(
        {
            "loss": torch.tensor(12.0),
            "pg_loss": torch.tensor(15.0),
            "ppo_kl": torch.tensor(5.5),
        },
        num_samples=2,
        num_tokens=torch.tensor(5.0),
        device=torch.device("cpu"),
        calculate_per_token_loss=True,
        metrics_token_reduced=False,
    )

    assert log_dict["keys"] == ["loss", "pg_loss", "ppo_kl"]
    torch.testing.assert_close(log_dict["values"], torch.tensor([12.0, 15.0, 5.5]))
    torch.testing.assert_close(log_dict["normalizers"], torch.tensor([5.0, 5.0, 2.0]))


def test_non_policy_per_token_loss_uses_token_reducer(miles, monkeypatch):
    state = _parallel_state(miles, cp_size=1)
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(miles.loss, "get_parallel_state", lambda: state)

    args = _args(loss_aggregation="token_mean", calculate_per_token_loss=True)
    args.loss_type = "sft_loss"
    args.global_batch_size = 2
    args.use_dynamic_global_batch_size = False
    args.recompute_loss_function = False
    args.true_on_policy_mode = False

    batch = {
        "loss_masks": [torch.ones(2), torch.ones(3)],
        "total_lengths": [2, 3],
        "response_lengths": [2, 3],
    }

    def fake_loss_function(args, batch, logits, sum_of_sample_mean):
        token_values = torch.arange(1.0, 6.0, device=logits.device)
        loss = sum_of_sample_mean(token_values)
        return loss, {"loss": loss.detach()}

    monkeypatch.setattr(miles.loss, "get_loss_function", lambda args: fake_loss_function)

    loss, normalizer, log_dict = miles.loss.loss_function(args, batch, 1, torch.ones((), requires_grad=True))

    torch.testing.assert_close(loss, torch.tensor(15.0))
    torch.testing.assert_close(normalizer, torch.tensor(5.0))
    torch.testing.assert_close(log_dict["values"], torch.tensor([15.0]))
    torch.testing.assert_close(log_dict["normalizers"], torch.tensor([5.0]))


def test_value_loss_per_token_metrics_stay_token_normalized(miles, monkeypatch):
    state = _parallel_state(miles, cp_size=1)
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(miles.loss, "get_parallel_state", lambda: state)
    monkeypatch.setattr(miles.log_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(miles.log_utils.MultiPGUtil, "all_reduce", lambda *args, **kwargs: None)

    args = _args(loss_aggregation="token_mean", calculate_per_token_loss=True)
    args.loss_type = "value_loss"
    args.global_batch_size = 2
    args.use_dynamic_global_batch_size = False
    args.recompute_loss_function = False
    args.true_on_policy_mode = False

    batch = {
        "loss_masks": [torch.ones(2), torch.ones(3)],
        "total_lengths": [2, 3],
        "response_lengths": [2, 3],
    }

    def fake_value_loss_function(args, batch, logits, sum_of_sample_mean):
        # 3 of 5 tokens clipped: the logged fraction must be 0.6, not 3 / num_samples.
        clipfrac = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0], device=logits.device)
        loss = sum_of_sample_mean(clipfrac)
        return loss, {
            "value_loss": loss.detach(),
            "value_clipfrac": sum_of_sample_mean(clipfrac).detach(),
        }

    monkeypatch.setattr(miles.loss, "get_loss_function", lambda args: fake_value_loss_function)

    _, _, log_dict = miles.loss.loss_function(args, batch, 1, torch.ones((), requires_grad=True))
    loss_dict = miles.log_utils.aggregate_train_losses([log_dict])

    torch.testing.assert_close(torch.tensor(loss_dict["value_clipfrac"]), torch.tensor(0.6))


def test_aggregate_train_losses_rejects_mixed_normalizer_contracts(miles, monkeypatch):
    state = _parallel_state(miles, cp_size=1)
    monkeypatch.setattr(miles.log_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(miles.log_utils.MultiPGUtil, "all_reduce", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="mix"):
        miles.log_utils.aggregate_train_losses(
            [
                {
                    "keys": ["pg_loss"],
                    "values": torch.tensor([10.0]),
                    "normalizers": torch.tensor([5.0]),
                },
                {
                    "keys": ["pg_loss"],
                    "values": torch.tensor([2.0, 6.0]),
                },
            ]
        )


def test_aggregate_train_losses_rejects_key_order_mismatch(miles, monkeypatch):
    state = _parallel_state(miles, cp_size=1)
    monkeypatch.setattr(miles.log_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(miles.log_utils.MultiPGUtil, "all_reduce", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="same keys"):
        miles.log_utils.aggregate_train_losses(
            [
                {
                    "keys": ["pg_loss", "ppo_kl"],
                    "values": torch.tensor([10.0, 5.0]),
                    "normalizers": torch.tensor([5.0, 2.0]),
                },
                {
                    "keys": ["ppo_kl", "pg_loss"],
                    "values": torch.tensor([5.0, 10.0]),
                    "normalizers": torch.tensor([2.0, 5.0]),
                },
            ]
        )


def test_aggregate_train_losses_rejects_bad_legacy_value_count(miles, monkeypatch):
    state = _parallel_state(miles, cp_size=1)
    monkeypatch.setattr(miles.log_utils, "get_parallel_state", lambda: state)

    with pytest.raises(ValueError, match="Expected 2 values"):
        miles.log_utils.aggregate_train_losses(
            [
                {
                    "keys": ["pg_loss"],
                    "values": torch.tensor([10.0]),
                },
            ]
        )


def test_aggregate_train_losses_rejects_bad_normalizer_count(miles, monkeypatch):
    state = _parallel_state(miles, cp_size=1)
    monkeypatch.setattr(miles.log_utils, "get_parallel_state", lambda: state)

    with pytest.raises(ValueError, match="Expected 2 normalizers"):
        miles.log_utils.aggregate_train_losses(
            [
                {
                    "keys": ["pg_loss", "ppo_kl"],
                    "values": torch.tensor([10.0, 5.0]),
                    "normalizers": torch.tensor([5.0]),
                },
            ]
        )


def test_aggregate_train_losses_rejects_zero_token_normalizer(miles, monkeypatch):
    state = _parallel_state(miles, cp_size=1)
    monkeypatch.setattr(miles.log_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(miles.log_utils.MultiPGUtil, "all_reduce", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="non-positive normalizer"):
        miles.log_utils.aggregate_train_losses(
            [
                {
                    "keys": ["pg_loss"],
                    "values": torch.tensor([0.0]),
                    "normalizers": torch.tensor([0.0]),
                }
            ]
        )
