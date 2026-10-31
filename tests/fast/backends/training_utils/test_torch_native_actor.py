from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cpu_ci
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.data.rollout import DataIterator
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.backends.training_utils.torch_native import actor as base_module
from miles.backends.training_utils.torch_native.actor import TorchNativeTrainRayActor
from miles.backends.training_utils.torch_native.offload import _map_optimizer_state
from miles.backends.training_utils.torch_native.step_runner import LinearStepRunner, StepMetrics
from miles.backends.training_utils.weight_update.rollout_cell_updater import _RolloutCellUpdater

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

_MODULE = "miles.backends.training_utils.torch_native.actor"


@contextmanager
def _recording(stages, name):
    stages.append(f"enter:{name}")
    yield
    stages.append(f"exit:{name}")


@contextmanager
def _noop_timer(_name):
    yield


class _Profiler:
    def iterate_train_actor(self, it):
        return it

    def iterate_train_log_probs(self, it):
        return it

    step = MagicMock()


class _DataIterator:
    def __init__(self):
        self.resets = 0
        self.fetches = 0

    def reset(self):
        self.resets += 1
        return self


class _Provider(TorchNativeTrainRayActor):
    def __init__(self):
        self.args = SimpleNamespace(
            micro_batch_size=1,
            offload_train=False,
            debug_rollout_only=False,
            data_pad_size_multiplier=1,
            qkv_format="thd",
            ci_test=False,
            use_sampling_support_replay=False,
        )
        self.model_parts = [MagicMock()]
        self.optimizers = [object()]
        self.hf_config = object()
        self.prof = _Profiler()
        self.runner = object()
        self.align_token_side_channel = lambda t, pad: t
        self._heartbeat = MagicMock()

    def _step_runner(self):
        return self.runner


@pytest.fixture
def step(monkeypatch):
    stages: list = []
    calls = {"log_probs": [], "steps": [], "stages": stages}
    replay = SimpleNamespace(
        FALLTHROUGH="fallthrough",
        REPLAY_BACKWARD="replay_backward",
        stage=lambda name: _recording(stages, name),
        fill=MagicMock(side_effect=lambda *a, **k: stages.append("fill")),
        log_prob_stage=lambda args: "replay_forward",
        rewind=MagicMock(side_effect=lambda: stages.append("rewind")),
        reset=MagicMock(side_effect=lambda: stages.append("reset")),
    )
    calls["replay"] = replay
    monkeypatch.setattr(base_module, "routing_replay", replay)
    monkeypatch.setattr(base_module, "get_data_iterator", lambda args, parts, data: ([object()], [2]))
    monkeypatch.setattr(
        TorchNativeTrainRayActor,
        "_log_probs",
        lambda self, runner, it, n, store_prefix="": (
            calls["log_probs"].append((runner, store_prefix)) or {f"{store_prefix}log_probs": [1]}
        ),
    )
    monkeypatch.setattr(
        TorchNativeTrainRayActor,
        "_optimizer_steps",
        lambda self, runner, it, n, rollout_id: calls["steps"].append(runner),
    )
    monkeypatch.setattr(base_module, "compute_advantages_and_returns", lambda args, data: data.update(adv=True))
    monkeypatch.setattr(base_module, "log_rollout_data", lambda rid, args, data: None)
    monkeypatch.setattr(base_module, "timer", _noop_timer)
    return calls


def test_the_rollout_step_runs_ref_then_actor_then_optimizer_under_the_right_stages(step):
    actor = _Provider()
    actor.ref_runner = object()
    actor._ref_context = lambda: _recording(step["stages"], "ref")
    rollout_data = {}

    actor._train_core(rollout_id=3, rollout_data=rollout_data)

    assert step["stages"] == [
        "fill",
        "enter:fallthrough",
        "enter:ref",
        "exit:ref",
        "exit:fallthrough",
        "enter:replay_forward",
        "exit:replay_forward",
        "rewind",
        "enter:replay_backward",
        "exit:replay_backward",
        "reset",
    ]
    assert step["log_probs"] == [(actor.ref_runner, "ref_"), (actor.runner, "")]
    assert step["steps"] == [actor.runner]
    assert rollout_data == {"ref_log_probs": [1], "log_probs": [1], "adv": True}
    assert step["replay"].fill.call_args.kwargs["align"] is actor.align_token_side_channel


_VOCAB_SIZE = 11
# per DP rank, (prompt length, response loss mask) per sample; one sample per micro-batch
_LOCAL_ROLLOUTS = (
    ((3, [1, 1, 1, 1]), (2, [1, 0, 1]), (4, [1, 1, 1, 1, 1, 1])),
    ((2, [1, 1]), (5, [0, 0, 0]), (1, [1, 1, 1, 1, 1, 0, 1])),
)
_NUM_MICROBATCHES = [2, 1]


def _local_samples(rank: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(rank)
    return [
        (torch.randint(_VOCAB_SIZE, (prompt_len + len(mask),), generator=generator), torch.tensor(mask))
        for prompt_len, mask in _LOCAL_ROLLOUTS[rank]
    ]


def _per_token_mean_nll(model: torch.nn.Module, samples: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    nll, num_tokens = 0, 0
    for tokens, mask in samples:
        response = tokens[-len(mask) :]
        log_probs = model(tokens)[-len(mask) - 1 : -1].log_softmax(-1).gather(-1, response[:, None]).squeeze(-1)
        nll = nll - (log_probs * mask).sum()
        num_tokens += max(int(mask.sum()), 1)
    return nll / num_tokens


def _cpu_batch(data_iterator, keys, *_args, **_kwargs) -> dict:
    batch = data_iterator.get_next(keys)
    return {**batch, "unconcat_tokens": batch["tokens"]}


def _train_one_dp_rank(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    trivial = GroupInfo(rank=0, size=1, group=None)
    dp = GroupInfo(rank=rank, size=world_size, group=dist.group.WORLD)
    set_parallel_state(
        ParallelState(
            intra_dp=dp,
            intra_dp_cp=dp,
            cp=trivial,
            tp=trivial,
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
            is_pp_last_stage=True,
        )
    )
    torch.manual_seed(0)
    model = torch.nn.Embedding(_VOCAB_SIZE, _VOCAB_SIZE)
    step_grads = []

    def average_grads_over_dp() -> StepMetrics:
        # FSDP2's reduce-scatter averages the gradients over DP
        dist.all_reduce(model.weight.grad)
        step_grads.append(model.weight.grad / world_size)
        return StepMetrics(grad_norm=0.0)

    actor = _Provider()
    actor.args = make_args(
        loss_type="sft_loss",
        calculate_per_token_loss=True,
        use_sampling_support_replay=False,
        data_pad_size_multiplier=1,
        ci_test=False,
    )
    samples = _local_samples(rank)
    rollout_data = {
        "tokens": [tokens for tokens, _ in samples],
        "loss_masks": [mask for _, mask in samples],
        "total_lengths": [len(tokens) for tokens, _ in samples],
        "response_lengths": [len(mask) for _, mask in samples],
    }
    runner = LinearStepRunner(
        lambda batch: model(torch.cat(batch["tokens"])).unsqueeze(0),
        lambda: model.zero_grad(set_to_none=True),
        average_grads_over_dp,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base_module, "get_batch", _cpu_batch)
        patch.setattr(base_module, "aggregate_train_losses", lambda losses: {})
        patch.setattr(base_module, "log_train_step", lambda **kwargs: None)
        actor._optimizer_steps(runner, DataIterator(rollout_data, micro_batch_size=1), _NUM_MICROBATCHES, rollout_id=0)

    samples_per_rank = [_local_samples(r) for r in range(world_size)]
    expected_grads = []
    step_start = 0
    for microbatches in _NUM_MICROBATCHES:
        step_samples = [
            sample
            for rank_samples in samples_per_rank
            for sample in rank_samples[step_start : step_start + microbatches]
        ]
        expected_grads.append(torch.autograd.grad(_per_token_mean_nll(model, step_samples), model.weight)[0])
        step_start += microbatches
    torch.testing.assert_close(torch.stack(step_grads), torch.stack(expected_grads))


def test_per_token_loss_gradient_is_the_mean_over_every_token_of_the_step():
    """Torch-native per-token gradients match Megatron's: the step's DP-global token mean, not a token sum."""
    run_multiprocess(_train_one_dp_rank)


def test_moving_optimizer_state_does_not_grow_state_for_parameters_that_have_none():
    model = torch.nn.Linear(4, 4, bias=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    assert model.bias not in optimizer.state

    _map_optimizer_state([optimizer], lambda tensor: tensor.to("cpu"))

    assert model.bias not in optimizer.state
    assert optimizer.state[model.weight]["exp_avg"].device.type == "cpu"


class TestPublishedWeightVersion:
    @pytest.mark.parametrize("completed", [False, True])
    @pytest.mark.parametrize("previous_version", [0, 3])
    def test_only_completed_transfers_advance_the_published_version(
        self, monkeypatch: pytest.MonkeyPatch, completed: bool, previous_version: int
    ) -> None:
        """A skipped transfer preserves the last published ordinal and its verification target."""
        actor = _Provider()
        actor.args.debug_train_only = False
        actor.args.ci_test = True
        actor._published_weight_version = previous_version
        actor.weight_updater = MagicMock()
        actor.weight_updater.update_weights.return_value = completed
        info = SimpleNamespace(rollout_engines=[object()])
        monkeypatch.setattr(base_module, "print_memory", lambda message: None)
        monkeypatch.setattr(base_module, "clear_memory", lambda: None)

        result = TorchNativeTrainRayActor.update_weights.__wrapped__(
            actor, info, debug_weight_update_id="update-1", rollout_id=1
        )

        expected_version = previous_version + int(completed)
        assert result.weight_version == expected_version
        assert result.failed_cell_ids == ()
        assert actor._published_weight_version == expected_version
        actor.weight_updater.update_weights.assert_called_once_with(weight_version=previous_version + 1)
        if expected_version:
            actor.weight_updater.verify_engine_version.assert_called_once_with(
                info.rollout_engines, weight_version=expected_version
            )
        else:
            actor.weight_updater.verify_engine_version.assert_not_called()

    def test_failed_transfer_keeps_the_previous_published_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed transfer cannot consume an ordinal or verify unpublished weights."""
        actor = _Provider()
        actor.args.debug_train_only = False
        actor._published_weight_version = 3
        actor.weight_updater = MagicMock()
        actor.weight_updater.update_weights.side_effect = RuntimeError("transfer failed")
        monkeypatch.setattr(base_module, "print_memory", lambda message: None)

        with pytest.raises(RuntimeError, match="transfer failed"):
            TorchNativeTrainRayActor.update_weights.__wrapped__(
                actor, SimpleNamespace(rollout_engines=[]), debug_weight_update_id="update-1", rollout_id=1
            )

        assert actor._published_weight_version == 3
        actor.weight_updater.verify_engine_version.assert_not_called()


class TestFailedTargetStopsTraining:
    @pytest.mark.parametrize("ci_test", [False, True])
    @pytest.mark.parametrize("failed_cell_ids", [("cell-b",), ("cell-a", "cell-b")])
    def test_any_failed_target_raises_before_verification(
        self, monkeypatch: pytest.MonkeyPatch, ci_test: bool, failed_cell_ids: tuple[str, ...]
    ) -> None:
        """TorchNative must stop on target failures even when another target succeeds."""
        actor = _Provider()
        actor.args.debug_train_only = False
        actor.args.ci_test = ci_test
        actor._published_weight_version = 3
        actor.weight_updater = MagicMock()
        error = ConnectionError(f"Failed update targets: {failed_cell_ids}")
        cell_updater = _RolloutCellUpdater(
            args=SimpleNamespace(ft_components=[]), cell_id=failed_cell_ids[0], api_client=None
        )
        actor.weight_updater.update_weights.side_effect = lambda **kwargs: cell_updater.mark_errored(error)
        monkeypatch.setattr(base_module, "print_memory", lambda message: None)

        with pytest.raises(ConnectionError) as raised:
            TorchNativeTrainRayActor.update_weights.__wrapped__(
                actor,
                SimpleNamespace(rollout_engines=[object(), object()]),
                debug_weight_update_id="update-4",
                rollout_id=1,
            )

        assert raised.value is error
        assert actor._published_weight_version == 3
        assert all(cell_id in str(raised.value) for cell_id in failed_cell_ids)
        actor.weight_updater.verify_engine_version.assert_not_called()
