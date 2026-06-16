"""Tests for the ``--loss-aggregation`` pg_loss modes."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch


def _install_sglang_protocol_stub():
    for name in [
        "sglang",
        "sglang.srt",
        "sglang.srt.entrypoints",
        "sglang.srt.entrypoints.openai",
    ]:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules.setdefault(name, module)
    protocol = types.ModuleType("sglang.srt.entrypoints.openai.protocol")
    protocol.Tool = object
    sys.modules.setdefault("sglang.srt.entrypoints.openai.protocol", protocol)
    server_args = types.ModuleType("sglang.srt.server_args")

    class _ServerArgs:
        @staticmethod
        def add_cli_args(parser):
            return None

    server_args.ServerArgs = _ServerArgs
    sys.modules.setdefault("sglang.srt.server_args", server_args)


_install_sglang_protocol_stub()
chat_template_utils = types.ModuleType("miles.utils.chat_template_utils")
chat_template_utils.__path__ = []
sys.modules.setdefault("miles.utils.chat_template_utils", chat_template_utils)
tito_tokenizer = types.ModuleType("miles.utils.chat_template_utils.tito_tokenizer")
tito_tokenizer.TITOTokenizerType = SimpleNamespace
sys.modules.setdefault("miles.utils.chat_template_utils.tito_tokenizer", tito_tokenizer)
sglang_router = types.ModuleType("sglang_router")
sglang_router.__path__ = []
sys.modules.setdefault("sglang_router", sglang_router)
launch_router = types.ModuleType("sglang_router.launch_router")
launch_router.RouterArgs = object
sys.modules.setdefault("sglang_router.launch_router", launch_router)

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss_hub.losses import get_pg_loss_reducer
from miles.backends.training_utils.parallel import GroupInfo, ParallelState
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.utils import arguments
from miles.utils.types import Sample


def _parallel_state(*, cp_size: int, cp_rank: int = 0) -> ParallelState:
    return ParallelState(
        intra_dp=GroupInfo(rank=0, size=1, group=None),
        intra_dp_cp=GroupInfo(rank=0, size=cp_size, group=None),
        cp=GroupInfo(rank=cp_rank, size=cp_size, group=None),
        tp=GroupInfo(rank=0, size=1, group=None),
        pp=GroupInfo(rank=0, size=1, group=None),
        ep=GroupInfo(rank=0, size=1, group=None),
        etp=GroupInfo(rank=0, size=1, group=None),
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
    torch.tensor([1.0, 1.0, 0.0]),  # group 0, mask sum 2
    torch.tensor([1.0, 1.0, 1.0]),  # group 0, mask sum 3
    torch.tensor([1.0, 0.0, 0.0, 0.0]),  # group 1, mask sum 1
    torch.tensor([1.0, 1.0, 1.0, 1.0]),  # group 1, mask sum 4
]
X = torch.arange(1.0, 15.0)


def test_default_is_byte_identical_to_legacy_reducer(monkeypatch):
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: _parallel_state(cp_size=1))

    new = get_sum_of_sample_mean(TOTAL_LENGTHS, RESPONSE_LENGTHS, LOSS_MASKS)  # sample_denoms=None
    legacy = _legacy_sum_of_sample_mean(RESPONSE_LENGTHS, LOSS_MASKS)

    torch.testing.assert_close(new(X), legacy(X), rtol=0, atol=0)


def test_prompt_mean_denominator_is_group_token_total(monkeypatch):
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: _parallel_state(cp_size=1))

    sample_denoms = [torch.tensor(5.0)] * 4
    reducer = get_sum_of_sample_mean(TOTAL_LENGTHS, RESPONSE_LENGTHS, LOSS_MASKS, sample_denoms)
    expected = (3 + 15) / 5 + (7 + 50) / 5
    torch.testing.assert_close(reducer(X), torch.tensor(expected))


def test_constant_divides_summed_token_loss_by_L(monkeypatch):
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: _parallel_state(cp_size=1))

    L = 10.0
    reducer = get_sum_of_sample_mean(TOTAL_LENGTHS, RESPONSE_LENGTHS, LOSS_MASKS, constant_divisor=L)
    torch.testing.assert_close(reducer(X), torch.tensor(75.0 / L))


def test_constant_distinct_from_sample_and_token_mean(monkeypatch):
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: _parallel_state(cp_size=1))

    sample_mean = get_sum_of_sample_mean(TOTAL_LENGTHS, RESPONSE_LENGTHS, LOSS_MASKS)
    token_sum = get_sum_of_sample_mean(TOTAL_LENGTHS, RESPONSE_LENGTHS, LOSS_MASKS, calculate_per_token_loss=True)
    constant = get_sum_of_sample_mean(TOTAL_LENGTHS, RESPONSE_LENGTHS, LOSS_MASKS, constant_divisor=10.0)

    assert sample_mean(X).item() != pytest.approx(constant(X).item())
    assert token_sum(X).item() != pytest.approx(constant(X).item())


def test_constant_and_per_token_loss_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: _parallel_state(cp_size=1))

    with pytest.raises(ValueError, match="mutually exclusive"):
        get_sum_of_sample_mean(
            TOTAL_LENGTHS, RESPONSE_LENGTHS, LOSS_MASKS, calculate_per_token_loss=True, constant_divisor=10.0
        )


def test_token_mean_is_unnormalized_global_token_sum(monkeypatch):
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: _parallel_state(cp_size=1))

    reducer = get_sum_of_sample_mean(TOTAL_LENGTHS, RESPONSE_LENGTHS, LOSS_MASKS, calculate_per_token_loss=True)
    torch.testing.assert_close(reducer(X), torch.tensor(75.0))


@pytest.mark.parametrize("constant_divisor", [None, 20.0])
def test_cp_zigzag_rank_sum_matches_single_rank(monkeypatch, constant_divisor):
    total_length, response_length = 10, 8
    loss_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])  # mask sum 7
    x_full = torch.arange(1.0, 9.0)
    sample_denoms = None if constant_divisor is not None else [torch.tensor(20.0)]

    def build():
        return get_sum_of_sample_mean(
            [total_length],
            [response_length],
            [loss_mask],
            sample_denoms,
            constant_divisor=constant_divisor,
        )

    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: _parallel_state(cp_size=1))
    ref = build()(x_full)

    total = torch.zeros(())
    for rank in range(2):
        monkeypatch.setattr(cp_utils, "get_parallel_state", lambda r=rank: _parallel_state(cp_size=2, cp_rank=r))
        x_local = cp_utils._slice_loss_mask_for_local_cp(total_length, response_length, x_full, "thd", None)
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


_DEFAULT_REDUCER = object()


def _select(args, batch):
    return get_pg_loss_reducer(
        args,
        batch,
        total_lengths=TOTAL_LENGTHS,
        response_lengths=RESPONSE_LENGTHS,
        pg_loss_masks=LOSS_MASKS,
        max_seq_lens=None,
        default_reducer=_DEFAULT_REDUCER,
    )


def test_sample_and_token_mean_reuse_the_metrics_reducer_for_pg_loss():
    assert _select(_args(loss_aggregation="sample_mean"), {}) is _DEFAULT_REDUCER
    assert _select(_args(loss_aggregation="token_mean", calculate_per_token_loss=True), {}) is _DEFAULT_REDUCER


def test_constant_and_prompt_mean_build_a_pg_loss_only_reducer(monkeypatch):
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: _parallel_state(cp_size=1))

    constant_reducer = _select(_args(loss_aggregation="constant", loss_aggregation_divisor=10.0), {})
    assert constant_reducer is not _DEFAULT_REDUCER
    torch.testing.assert_close(constant_reducer(X), torch.tensor(75.0 / 10.0))

    denoms = [torch.tensor(5.0)] * 4
    prompt_reducer = _select(_args(loss_aggregation="prompt_mean"), {"prompt_mask_sums": denoms})
    assert prompt_reducer is not _DEFAULT_REDUCER
    torch.testing.assert_close(prompt_reducer(X), torch.tensor(2 * ((3 + 15) / 5 + (7 + 50) / 5)))


def test_prompt_mean_without_prompt_mask_sums_fails():
    with pytest.raises(ValueError, match="prompt_mask_sums"):
        _select(_args(loss_aggregation="prompt_mean"), {})


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
def test_validate_constant_rejects_nonpositive_divisor(divisor):
    args = _validate_args(loss_aggregation="constant", loss_aggregation_divisor=divisor)
    with pytest.raises(ValueError, match="loss-aggregation-divisor"):
        arguments._validate_loss_aggregation_args(args)


def test_validate_token_mean_aliases_calculate_per_token_loss():
    args = _validate_args(loss_aggregation="token_mean")
    arguments._validate_loss_aggregation_args(args)
    assert args.calculate_per_token_loss is True


def test_validate_calculate_per_token_loss_alone_reconciles_to_token_mean():
    args = _validate_args(calculate_per_token_loss=True)
    arguments._validate_loss_aggregation_args(args)
    assert args.loss_aggregation == "token_mean"
    assert args.calculate_per_token_loss is True


def test_validate_default_leaves_per_token_loss_off():
    args = _validate_args()
    arguments._validate_loss_aggregation_args(args)
    assert args.loss_aggregation == "sample_mean"
    assert args.calculate_per_token_loss is False


def test_validate_constant_rejects_calculate_per_token_loss():
    args = _validate_args(loss_aggregation="constant", loss_aggregation_divisor=10.0, calculate_per_token_loss=True)
    with pytest.raises(ValueError, match="incompatible with --calculate-per-token-loss"):
        arguments._validate_loss_aggregation_args(args)


def test_validate_constant_with_per_token_loss_off_passes():
    args = _validate_args(loss_aggregation="constant", loss_aggregation_divisor=10.0, calculate_per_token_loss=False)
    arguments._validate_loss_aggregation_args(args)
    assert args.calculate_per_token_loss is False


def test_validate_prompt_mean_rejects_calculate_per_token_loss():
    args = _validate_args(loss_aggregation="prompt_mean", calculate_per_token_loss=True)
    with pytest.raises(ValueError, match="incompatible with --calculate-per-token-loss"):
        arguments._validate_loss_aggregation_args(args)


def test_validate_prompt_mean_rejects_non_multiple_global_batch_size():
    args = _validate_args(loss_aggregation="prompt_mean", global_batch_size=3, n_samples_per_prompt=2)
    with pytest.raises(ValueError, match="multiple of n_samples_per_prompt"):
        arguments._validate_loss_aggregation_args(args)


def test_validate_prompt_mean_accepts_multiple_global_batch_size():
    args = _validate_args(loss_aggregation="prompt_mean", global_batch_size=6, n_samples_per_prompt=2)
    arguments._validate_loss_aggregation_args(args)
    assert args.loss_aggregation == "prompt_mean"


def test_validate_non_prompt_mean_allows_non_multiple_global_batch_size():
    args = _validate_args(loss_aggregation="sample_mean", global_batch_size=3, n_samples_per_prompt=2)
    arguments._validate_loss_aggregation_args(args)
    assert args.loss_aggregation == "sample_mean"


def _make_sample(group_index, index, response_length, loss_mask):
    s = Sample(group_index=group_index, index=index, response_length=response_length, reward=1.0)
    s.tokens = [0] * response_length
    s.loss_mask = loss_mask
    s.status = Sample.Status.COMPLETED
    return s


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


def _convert(samples, args):
    return convert_samples_to_train_data(
        args,
        samples,
        metadata={},
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )


def test_convert_samples_computes_step_level_prompt_group_denoms():
    samples = [
        _make_sample(0, 0, 3, [1, 1, 0]),  # group 0, sum 2
        _make_sample(0, 1, 3, [1, 1, 1]),  # group 0, sum 3
        _make_sample(1, 2, 4, [1, 0, 0, 0]),  # group 1, sum 1
        _make_sample(1, 3, 4, [1, 1, 1, 1]),  # group 1, sum 4
    ]
    train_data = _convert(samples, _convert_args())
    assert train_data["prompt_mask_sums"] == [5, 5, 5, 5]


def test_convert_samples_prompt_mean_rejects_none_group_index():
    samples = [_make_sample(None, 0, 2, [1, 1]), _make_sample(None, 1, 2, [1, 0])]
    with pytest.raises(ValueError, match="group_index"):
        _convert(samples, _convert_args(rollout_batch_size=1))


def test_convert_samples_omits_denoms_for_default_mode():
    samples = [_make_sample(0, 0, 2, [1, 1]), _make_sample(0, 1, 2, [1, 0])]
    train_data = _convert(samples, _convert_args(loss_aggregation="sample_mean", rollout_batch_size=1))
    assert "prompt_mask_sums" not in train_data
