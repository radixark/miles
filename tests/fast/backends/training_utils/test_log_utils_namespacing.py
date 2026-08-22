from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.backends.training_utils import log_utils
from miles.utils.metric_utils import strip_metrics_namespace


def _capture(monkeypatch) -> list[tuple[dict, str]]:
    calls: list[tuple[dict, str]] = []
    monkeypatch.setattr(log_utils.tracking, "log", lambda _args, payload, step_key: calls.append((payload, step_key)))
    return calls


@pytest.fixture()
def source_rank(monkeypatch) -> None:
    parallel_state = SimpleNamespace(effective_dp_cp=SimpleNamespace(rank=0, size=1, gloo_groups_inner_to_outer=[]))
    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(log_utils.MultiPGUtil, "gather_object", staticmethod(lambda obj, groups_inner_to_outer: [obj]))


class TestLogTrainStep:
    def test_a_policy_run_namespaces_every_train_metric_and_its_step_axis(self, monkeypatch):
        """Two policies train at their own pace, so sharing train/step would interleave their curves."""
        calls = _capture(monkeypatch)

        log_dict = log_utils.log_train_step(
            args=Namespace(trainer_model_id="alpha"),
            loss_dict={"loss": 0.5},
            grad_norm=2.0,
            rollout_id=2,
            step_id=1,
            num_steps_per_rollout=4,
            should_log=True,
        )

        [(payload, step_key)] = calls
        assert step_key == "alpha/train/step"
        assert payload == {"alpha/train/loss": 0.5, "alpha/train/grad_norm": 2.0, "alpha/train/step": 9}
        assert log_dict == payload

    def test_a_run_without_a_policy_id_keeps_the_names_it_had(self, monkeypatch):
        """Single policy runs feed dashboards written against the unprefixed names."""
        calls = _capture(monkeypatch)

        log_dict = log_utils.log_train_step(
            args=Namespace(trainer_model_id=None),
            loss_dict={"loss": 0.5},
            grad_norm=2.0,
            rollout_id=2,
            step_id=1,
            num_steps_per_rollout=4,
            should_log=True,
        )

        [(payload, step_key)] = calls
        assert step_key == "train/step"
        assert log_dict == payload == {"train/loss": 0.5, "train/grad_norm": 2.0, "train/step": 9}


class TestGatherLogData:
    @pytest.mark.parametrize("trainer_model_id", [None, "alpha"])
    def test_the_reduced_metrics_carry_the_namespace_of_their_policy(self, source_rank, monkeypatch, trainer_model_id):
        """The answer is the same object tracking receives, so a reader of either sees one naming scheme."""
        calls = _capture(monkeypatch)
        args = Namespace(trainer_model_id=trainer_model_id, wandb_always_use_train_step=False)

        reduced = log_utils.gather_log_data("rollout", args, 3, {"log_probs": -1.0})

        namespace = "" if trainer_model_id is None else f"{trainer_model_id}/"
        [(payload, step_key)] = calls
        assert step_key == f"{namespace}rollout/step"
        assert reduced == payload == {f"{namespace}rollout/log_probs": -1.0, f"{namespace}rollout/step": 3}


class TestLogCpuMemory:
    def test_the_memory_series_of_a_policy_carries_its_own_step_axis(self, monkeypatch):
        """The memory point is logged per rollout, so it needs the same policy step axis as the other perf points."""
        calls = _capture(monkeypatch)

        log_utils.log_cpu_memory(
            rollout_id=3, args=Namespace(trainer_model_id="alpha", wandb_always_use_train_step=False), label="before"
        )

        [(payload, step_key)] = calls
        assert step_key == "alpha/rollout/step"
        assert payload["alpha/rollout/step"] == 3
        assert set(payload) == {"alpha/perf/cpu_memory_before_gb", "alpha/rollout/step"}


class TestStripMetricsNamespace:
    def test_the_ci_checkers_see_the_names_they_are_written_against(self):
        """Every CI assertion indexes train/ppo_kl, whichever policy produced the step."""
        namespaced = {"alpha/train/ppo_kl": 0.1, "alpha/train/step": 4}

        assert strip_metrics_namespace(namespaced, trainer_model_id="alpha") == {
            "train/ppo_kl": 0.1,
            "train/step": 4,
        }

    def test_a_single_policy_dict_is_handed_back_untouched(self):
        """Nothing was prefixed, so stripping must not copy or rename anything."""
        log_dict = {"train/ppo_kl": 0.1}

        assert strip_metrics_namespace(log_dict, trainer_model_id=None) is log_dict
