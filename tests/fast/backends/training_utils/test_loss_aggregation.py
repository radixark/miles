"""Tests for the ``--loss-aggregation`` pg_loss modes."""

import argparse
import importlib
import sys
import types
from enum import Enum
from types import SimpleNamespace

import pytest
import torch

_MISSING = object()
_MILES_MODULES = [
    "miles.backends.sglang_utils.arguments",
    "miles.backends.training_utils.cp_utils",
    "miles.backends.training_utils.log_utils",
    "miles.backends.training_utils.loss",
    "miles.backends.training_utils.loss_hub.losses",
    "miles.backends.training_utils.parallel",
    "miles.ray.rollout.train_data_conversion",
    "miles.utils.arguments",
    "miles.utils.types",
]


def _install_miles_import_stubs(monkeypatch):
    for name in [
        "sglang",
        "sglang.srt",
        "sglang.srt.entrypoints",
        "sglang.srt.entrypoints.openai",
    ]:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    protocol = types.ModuleType("sglang.srt.entrypoints.openai.protocol")
    protocol.Tool = object
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.openai.protocol", protocol)

    server_args = types.ModuleType("sglang.srt.server_args")

    class _ServerArgs:
        @staticmethod
        def add_cli_args(parser):
            return None

    server_args.ServerArgs = _ServerArgs
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)

    chat_template_utils = types.ModuleType("miles.utils.chat_template_utils")
    chat_template_utils.__path__ = []
    chat_template_utils.resolve_fixed_chat_template = lambda *args, **kwargs: (None, {})
    monkeypatch.setitem(sys.modules, "miles.utils.chat_template_utils", chat_template_utils)

    tito_tokenizer = types.ModuleType("miles.utils.chat_template_utils.tito_tokenizer")

    class _TITOTokenizerType(Enum):
        DEFAULT = "default"

    tito_tokenizer.TITOTokenizerType = _TITOTokenizerType
    monkeypatch.setitem(sys.modules, "miles.utils.chat_template_utils.tito_tokenizer", tito_tokenizer)

    sglang_router = types.ModuleType("sglang_router")
    sglang_router.__path__ = []
    monkeypatch.setitem(sys.modules, "sglang_router", sglang_router)

    launch_router = types.ModuleType("sglang_router.launch_router")

    class _RouterArgs:
        @staticmethod
        def add_cli_args(parser, *args, **kwargs):
            return None

    launch_router.RouterArgs = _RouterArgs
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", launch_router)


@pytest.fixture
def miles(monkeypatch):
    previous_modules = {name: sys.modules.get(name, _MISSING) for name in _MILES_MODULES}
    for name in _MILES_MODULES:
        sys.modules.pop(name, None)

    _install_miles_import_stubs(monkeypatch)

    cp_utils = importlib.import_module("miles.backends.training_utils.cp_utils")
    log_utils = importlib.import_module("miles.backends.training_utils.log_utils")
    loss = importlib.import_module("miles.backends.training_utils.loss")
    losses = importlib.import_module("miles.backends.training_utils.loss_hub.losses")
    parallel = importlib.import_module("miles.backends.training_utils.parallel")
    train_data_conversion = importlib.import_module("miles.ray.rollout.train_data_conversion")
    arguments = importlib.import_module("miles.utils.arguments")
    types_module = importlib.import_module("miles.utils.types")

    try:
        yield SimpleNamespace(
            arguments=arguments,
            convert_samples_to_train_data=train_data_conversion.convert_samples_to_train_data,
            cp_utils=cp_utils,
            log_utils=log_utils,
            loss=loss,
            get_pg_loss_reducer=losses.get_pg_loss_reducer,
            get_sum_of_sample_mean=cp_utils.get_sum_of_sample_mean,
            GroupInfo=parallel.GroupInfo,
            ParallelState=parallel.ParallelState,
            Sample=types_module.Sample,
        )
    finally:
        for name in reversed(_MILES_MODULES):
            previous = previous_modules[name]
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _parallel_state(miles, *, cp_size: int, cp_rank: int = 0):
    return miles.ParallelState(
        intra_dp=miles.GroupInfo(rank=0, size=1, group=None),
        intra_dp_cp=miles.GroupInfo(rank=0, size=cp_size, group=None),
        cp=miles.GroupInfo(rank=cp_rank, size=cp_size, group=None),
        tp=miles.GroupInfo(rank=0, size=1, group=None),
        pp=miles.GroupInfo(rank=0, size=1, group=None),
        ep=miles.GroupInfo(rank=0, size=1, group=None),
        etp=miles.GroupInfo(rank=0, size=1, group=None),
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


def test_constant_does_not_compute_sample_denoms(miles, monkeypatch):
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: _parallel_state(miles, cp_size=1))

    original_sum = torch.Tensor.sum

    def fail_sum(self, *args, **kwargs):
        raise AssertionError("constant aggregation should not compute sample denominators")

    monkeypatch.setattr(torch.Tensor, "sum", fail_sum)
    reducer = miles.get_sum_of_sample_mean([3], [3], [torch.ones(3)], constant_divisor=10.0)
    monkeypatch.setattr(torch.Tensor, "sum", original_sum)

    torch.testing.assert_close(reducer(torch.arange(1.0, 4.0)), torch.tensor(0.6))


def test_constant_and_per_token_loss_are_mutually_exclusive(miles, monkeypatch):
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: _parallel_state(miles, cp_size=1))

    with pytest.raises(ValueError, match="mutually exclusive"):
        miles.get_sum_of_sample_mean(
            TOTAL_LENGTHS, RESPONSE_LENGTHS, LOSS_MASKS, calculate_per_token_loss=True, constant_divisor=10.0
        )


def test_token_mean_is_unnormalized_global_token_sum(miles, monkeypatch):
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: _parallel_state(miles, cp_size=1))

    reducer = miles.get_sum_of_sample_mean(TOTAL_LENGTHS, RESPONSE_LENGTHS, LOSS_MASKS, calculate_per_token_loss=True)
    torch.testing.assert_close(reducer(X), torch.tensor(75.0))


def test_legacy_positional_per_token_loss_call_still_works(miles, monkeypatch):
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: _parallel_state(miles, cp_size=1))

    reducer = miles.get_sum_of_sample_mean(TOTAL_LENGTHS, RESPONSE_LENGTHS, LOSS_MASKS, True)
    torch.testing.assert_close(reducer(X), torch.tensor(75.0))


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


def test_missing_loss_aggregation_reuses_legacy_sample_mean(miles, monkeypatch):
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: _parallel_state(miles, cp_size=1))

    args = SimpleNamespace(calculate_per_token_loss=False, qkv_format="thd", n_samples_per_prompt=2)
    reducer = _select(miles, args, {})
    torch.testing.assert_close(reducer(X), torch.tensor(26.0))


def test_missing_loss_aggregation_reuses_legacy_token_mean(miles, monkeypatch):
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: _parallel_state(miles, cp_size=1))

    args = SimpleNamespace(calculate_per_token_loss=True, qkv_format="thd", n_samples_per_prompt=2)
    reducer = _select(miles, args, {})
    torch.testing.assert_close(reducer(X), torch.tensor(75.0))


def test_prompt_mean_recomputes_denoms_from_current_pg_loss_masks(miles, monkeypatch):
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: _parallel_state(miles, cp_size=1))

    pg_loss_masks = [
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
    ]

    reducer = _select(
        miles,
        _args(loss_aggregation="prompt_mean"),
        {"prompt_group_indices": PROMPT_GROUP_INDICES, "prompt_mask_sums": [torch.tensor(99.0)] * 4},
        pg_loss_masks=pg_loss_masks,
    )

    torch.testing.assert_close(reducer(X), torch.tensor(22.0))


def test_prompt_mean_handles_zero_mask_completion_in_group(miles, monkeypatch):
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: _parallel_state(miles, cp_size=1))

    args = _args(loss_aggregation="prompt_mean")
    pg_loss_masks = [torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])]
    pg_loss = torch.tensor([100.0, 200.0, 3.0, 7.0])

    reducer = miles.get_pg_loss_reducer(
        args,
        {
            "loss_masks": [torch.ones(2), torch.ones(2)],
            "prompt_group_indices": [0, 0],
            "prompt_mask_sums": [torch.tensor(99.0), torch.tensor(99.0)],
        },
        total_lengths=[2, 2],
        response_lengths=[2, 2],
        pg_loss_masks=pg_loss_masks,
        max_seq_lens=None,
        default_reducer=miles.get_sum_of_sample_mean([2, 2], [2, 2], pg_loss_masks),
    )

    torch.testing.assert_close(reducer(pg_loss), torch.tensor(10.0))


def test_prompt_mean_without_prompt_mask_sums_fails(miles):
    with pytest.raises(ValueError, match="prompt_mask_sums"):
        _select(miles, _args(loss_aggregation="prompt_mean"), {}, default_reducer=lambda x: x)


def test_prompt_mean_modified_masks_without_prompt_group_indices_fails(miles):
    pg_loss_masks = [loss_mask.clone() for loss_mask in LOSS_MASKS]
    with pytest.raises(ValueError, match="prompt_group_indices"):
        _select(
            miles,
            _args(loss_aggregation="prompt_mean"),
            {"prompt_mask_sums": [torch.tensor(5.0)] * 4},
            default_reducer=lambda x: x,
            pg_loss_masks=pg_loss_masks,
        )


def test_prompt_mean_modified_masks_rejects_partial_prompt_groups(miles, monkeypatch):
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: _parallel_state(miles, cp_size=1))

    pg_loss_masks = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0])]
    with pytest.raises(ValueError, match="complete prompt groups"):
        miles.get_pg_loss_reducer(
            _args(loss_aggregation="prompt_mean"),
            {
                "loss_masks": [torch.ones(2), torch.ones(2)],
                "prompt_group_indices": [0, 1],
                "prompt_mask_sums": [torch.tensor(99.0), torch.tensor(99.0)],
            },
            total_lengths=[2, 2],
            response_lengths=[2, 2],
            pg_loss_masks=pg_loss_masks,
            max_seq_lens=None,
            default_reducer=miles.get_sum_of_sample_mean([2, 2], [2, 2], pg_loss_masks),
        )


def test_token_mean_log_aggregation_keeps_metric_sample_mean(miles, monkeypatch):
    state = _parallel_state(miles, cp_size=1)
    monkeypatch.setattr(miles.cp_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(miles.loss, "get_parallel_state", lambda: state)
    monkeypatch.setattr(miles.log_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(miles.log_utils.dist, "all_reduce", lambda *args, **kwargs: None)

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


def _validate_args(**overrides):
    base = dict(
        loss_aggregation="sample_mean",
        loss_aggregation_divisor=None,
        calculate_per_token_loss=False,
        global_batch_size=4,
        n_samples_per_prompt=2,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("divisor", [None, 0.0, -1.0, float("nan")])
def test_validate_constant_rejects_nonpositive_divisor(miles, divisor):
    args = _validate_args(loss_aggregation="constant", loss_aggregation_divisor=divisor)
    with pytest.raises(ValueError, match="loss-aggregation-divisor"):
        miles.arguments._validate_loss_aggregation_args(args)


def test_validate_token_mean_aliases_calculate_per_token_loss(miles):
    args = _validate_args(loss_aggregation="token_mean")
    miles.arguments._validate_loss_aggregation_args(args)
    assert args.calculate_per_token_loss is True


def test_validate_calculate_per_token_loss_alone_reconciles_to_token_mean(miles):
    args = _validate_args(calculate_per_token_loss=True)
    miles.arguments._validate_loss_aggregation_args(args)
    assert args.loss_aggregation == "token_mean"
    assert args.calculate_per_token_loss is True


def test_validate_default_leaves_per_token_loss_off(miles):
    args = _validate_args()
    miles.arguments._validate_loss_aggregation_args(args)
    assert args.loss_aggregation == "sample_mean"
    assert args.calculate_per_token_loss is False


def test_validate_constant_rejects_calculate_per_token_loss(miles):
    args = _validate_args(loss_aggregation="constant", loss_aggregation_divisor=10.0, calculate_per_token_loss=True)
    with pytest.raises(ValueError, match="incompatible with --calculate-per-token-loss"):
        miles.arguments._validate_loss_aggregation_args(args)


def test_validate_constant_with_per_token_loss_off_passes(miles):
    args = _validate_args(loss_aggregation="constant", loss_aggregation_divisor=10.0, calculate_per_token_loss=False)
    miles.arguments._validate_loss_aggregation_args(args)
    assert args.calculate_per_token_loss is False


def test_validate_prompt_mean_rejects_calculate_per_token_loss(miles):
    args = _validate_args(loss_aggregation="prompt_mean", calculate_per_token_loss=True)
    with pytest.raises(ValueError, match="incompatible with --calculate-per-token-loss"):
        miles.arguments._validate_loss_aggregation_args(args)


def test_validate_prompt_mean_rejects_non_multiple_global_batch_size(miles):
    args = _validate_args(loss_aggregation="prompt_mean", global_batch_size=3, n_samples_per_prompt=2)
    with pytest.raises(ValueError, match="multiple of n_samples_per_prompt"):
        miles.arguments._validate_loss_aggregation_args(args)


def test_miles_validate_args_checks_prompt_mean_after_deriving_global_batch_size(miles):
    parser = argparse.ArgumentParser()
    miles.arguments.get_miles_extra_args_provider()(parser)
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "3",
            "--n-samples-per-prompt",
            "2",
            "--num-steps-per-rollout",
            "2",
            "--loss-aggregation",
            "prompt_mean",
        ]
    )

    with pytest.raises(ValueError, match="multiple of n_samples_per_prompt"):
        miles.arguments.miles_validate_args(args)


def test_validate_prompt_mean_accepts_multiple_global_batch_size(miles):
    args = _validate_args(loss_aggregation="prompt_mean", global_batch_size=6, n_samples_per_prompt=2)
    miles.arguments._validate_loss_aggregation_args(args)
    assert args.loss_aggregation == "prompt_mean"


def test_validate_non_prompt_mean_allows_non_multiple_global_batch_size(miles):
    args = _validate_args(loss_aggregation="sample_mean", global_batch_size=3, n_samples_per_prompt=2)
    miles.arguments._validate_loss_aggregation_args(args)
    assert args.loss_aggregation == "sample_mean"


def _make_sample(miles, group_index, index, response_length, loss_mask):
    sample = miles.Sample(group_index=group_index, index=index, response_length=response_length, reward=1.0)
    sample.tokens = [0] * response_length
    sample.loss_mask = loss_mask
    sample.status = miles.Sample.Status.COMPLETED
    return sample


def _convert_args(**overrides):
    base = dict(
        advantage_estimator="grpo",
        rewards_normalization=False,
        reward_key=None,
        use_dynamic_global_batch_size=False,
        loss_aggregation="prompt_mean",
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        grpo_std_normalization=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _convert(miles, samples, args):
    return miles.convert_samples_to_train_data(
        args,
        samples,
        metadata={},
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )


def test_convert_samples_computes_step_level_prompt_group_denoms(miles):
    samples = [
        _make_sample(miles, 0, 0, 3, [1, 1, 0]),
        _make_sample(miles, 0, 1, 3, [1, 1, 1]),
        _make_sample(miles, 1, 2, 4, [1, 0, 0, 0]),
        _make_sample(miles, 1, 3, 4, [1, 1, 1, 1]),
    ]

    train_data = _convert(miles, samples, _convert_args())

    assert train_data["prompt_group_indices"] == [0, 0, 1, 1]
    assert train_data["prompt_mask_sums"] == [5, 5, 5, 5]


def test_convert_samples_prompt_mean_rejects_none_group_index(miles):
    samples = [_make_sample(miles, None, 0, 2, [1, 1]), _make_sample(miles, None, 1, 2, [1, 0])]
    with pytest.raises(ValueError, match="group_index"):
        _convert(miles, samples, _convert_args(rollout_batch_size=1))


def test_convert_samples_omits_prompt_group_fields_for_default_mode(miles):
    samples = [_make_sample(miles, 0, 0, 2, [1, 1]), _make_sample(miles, 0, 1, 2, [1, 0])]

    train_data = _convert(miles, samples, _convert_args(loss_aggregation="sample_mean", rollout_batch_size=1))

    assert "prompt_group_indices" not in train_data
    assert "prompt_mask_sums" not in train_data
