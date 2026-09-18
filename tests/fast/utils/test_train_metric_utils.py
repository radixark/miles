from __future__ import annotations

from types import SimpleNamespace

import pytest

from miles.utils import train_metric_utils
from miles.utils.misc import SingletonMeta
from miles.utils.timer import Timer
from miles.utils.train_metric_utils import log_perf_data_raw

SEQ_LENS = [1024, 2048]
FWD_TFLOPS = 60.0


@pytest.fixture
def timer():
    SingletonMeta._instances.pop(Timer, None)
    instance = Timer()
    instance.seq_lens = SEQ_LENS
    yield instance
    SingletonMeta._instances.pop(Timer, None)


@pytest.fixture
def logged(monkeypatch):
    calls = []
    monkeypatch.setattr(train_metric_utils.tracking, "log", lambda args, payload, **kw: calls.append(payload))
    return calls


def make_args(**overrides):
    defaults = dict(wandb_always_use_train_step=False, mfu_peak_tflops=None, trainer_model_id=None)
    return SimpleNamespace(**{**defaults, **overrides})


def run(timer, args, *, times: dict[str, float], peak: float | None):
    timer.timers = dict(times)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train_metric_utils, "local_peak_bf16_tflops", lambda: peak)
        log_perf_data_raw(
            rollout_id=0,
            args=args,
            is_primary_rank=True,
            compute_total_fwd_flops=lambda seq_lens: FWD_TFLOPS,
        )


def test_mfu_is_train_tflops_over_the_device_peak(timer, logged):
    run(timer, make_args(), times={"actor_train": 2.0}, peak=989.0)
    [payload] = logged
    assert payload["perf/actor_train_tflops"] == pytest.approx(3 * FWD_TFLOPS / 2.0)
    assert payload["perf/actor_train_mfu"] == pytest.approx(90.0 / 989.0)
    assert payload["perf/mfu_peak_tflops"] == 989.0


def test_denominator_is_logged_so_the_ratio_is_auditable(timer, logged):
    run(timer, make_args(), times={"actor_train": 2.0}, peak=989.0)
    [payload] = logged
    assert payload["perf/actor_train_tflops"] == pytest.approx(
        payload["perf/actor_train_mfu"] * payload["perf/mfu_peak_tflops"]
    )


def test_unknown_peak_omits_mfu_instead_of_defaulting(timer, logged):
    run(timer, make_args(), times={"actor_train": 2.0}, peak=None)
    [payload] = logged
    assert "perf/actor_train_tflops" in payload  # the throughput metric still lands
    assert "perf/actor_train_mfu" not in payload
    assert "perf/mfu_peak_tflops" not in payload


def test_no_mfu_without_a_flops_model(timer, logged):
    timer.timers = {"actor_train": 2.0}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train_metric_utils, "local_peak_bf16_tflops", lambda: 989.0)
        log_perf_data_raw(rollout_id=0, args=make_args(), is_primary_rank=True, compute_total_fwd_flops=None)
    [payload] = logged
    assert "perf/actor_train_mfu" not in payload
    assert "perf/actor_train_tflops" not in payload


def test_zero_train_time_emits_neither_throughput_nor_mfu(timer, logged):
    run(timer, make_args(), times={"actor_train": 0.0}, peak=989.0)
    [payload] = logged
    assert "perf/actor_train_tflops" not in payload
    assert "perf/actor_train_mfu" not in payload


def test_non_primary_rank_logs_nothing(timer, logged):
    timer.timers = {"actor_train": 2.0}
    log_perf_data_raw(rollout_id=0, args=make_args(), is_primary_rank=False, compute_total_fwd_flops=lambda **_: 1.0)
    assert logged == []


class TestLogPerfData:
    def test_the_perf_curves_follow_the_policy_step_axis(self, timer, monkeypatch):
        """Every perf point of a policy must land on that policy's own step axis, not on a shared one."""
        calls: list[tuple[dict, str]] = []
        monkeypatch.setattr(
            train_metric_utils.tracking, "log", lambda _args, payload, step_key: calls.append((payload, step_key))
        )
        timer.timers = {"actor_train": 2.0}

        log_perf_data_raw(
            rollout_id=3,
            args=make_args(trainer_model_id="alpha"),
            is_primary_rank=True,
            compute_total_fwd_flops=None,
        )

        [(payload, step_key)] = calls
        assert step_key == "alpha/rollout/step"
        assert payload == {"alpha/perf/actor_train_time": 2.0, "alpha/rollout/step": 3}
