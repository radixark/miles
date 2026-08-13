from argparse import Namespace

from miles.backends.training_utils import log_utils


def _capture(monkeypatch) -> list[tuple[dict, str]]:
    calls: list[tuple[dict, str]] = []
    monkeypatch.setattr(log_utils.tracking, "log", lambda _args, payload, step_key: calls.append((payload, step_key)))
    return calls


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

        log_utils.log_train_step(
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
        assert payload == {"train/loss": 0.5, "train/grad_norm": 2.0, "train/step": 9}


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
