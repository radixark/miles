from argparse import Namespace
from contextlib import contextmanager, nullcontext
from unittest.mock import Mock

from miles.backends.megatron_utils import actor as actor_module
from miles.backends.megatron_utils.actor import MegatronTrainRayActor


def _worker(role):
    worker = object.__new__(MegatronTrainRayActor)
    worker.args = Namespace(offload_train=True, debug_rollout_only=False)
    worker.role = role
    worker._heartbeat = Mock()
    worker.wake_up = Mock()
    worker.sleep = Mock()
    return worker


def test_critic_train_wakes_on_config_and_sleeps_on_options(monkeypatch):
    worker = _worker("critic")
    worker.train_critic = Mock(return_value={"values": ["cpu-value"]})
    monkeypatch.setattr(
        actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: ({"tokens": []}, nullcontext())
    )
    phases = []

    @contextmanager
    def capture_timer(name):
        phases.append(name)
        yield

    monkeypatch.setattr(actor_module, "timer", capture_timer)

    result = worker.train(3, object(), options={"sleep_after_train": True})

    worker.wake_up.assert_called_once_with()
    worker.train_critic.assert_called_once()
    worker.sleep.assert_called_once_with()
    assert result == {"values": ["cpu-value"]}
    assert phases == ["data_preprocess", "critic_train"]


def test_actor_receives_critic_payload_between_wake_and_sleep(monkeypatch):
    worker = _worker("actor")
    worker.train_actor = Mock(return_value=None)
    monkeypatch.setattr(
        actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: ({"tokens": []}, nullcontext())
    )
    values = {"values": ["cpu-value"]}

    result = worker.train(4, object(), external_data=values, options={"sleep_after_train": True})

    worker.wake_up.assert_called_once_with()
    worker.train_actor.assert_called_once()
    assert worker.train_actor.call_args.kwargs["external_data"] is values
    worker.sleep.assert_called_once_with()
    assert result is None


def test_train_without_options_keeps_model_resident(monkeypatch):
    worker = _worker("actor")
    worker.train_actor = Mock(return_value=None)
    monkeypatch.setattr(
        actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: ({"tokens": []}, nullcontext())
    )

    worker.train(5, object())

    worker.wake_up.assert_called_once_with()
    worker.sleep.assert_not_called()


def _lifecycle_worker(monkeypatch, asleep):
    worker = object.__new__(MegatronTrainRayActor)
    worker.args = Namespace(offload_train=True)
    worker._asleep = asleep
    saver = Mock()
    reload_groups = Mock()
    monkeypatch.setattr(actor_module, "torch_memory_saver", saver)
    monkeypatch.setattr(actor_module, "clear_memory", Mock())
    monkeypatch.setattr(actor_module, "print_memory", Mock())
    monkeypatch.setattr(actor_module, "destroy_process_groups", Mock())
    monkeypatch.setattr(actor_module, "reload_process_groups", reload_groups)
    monkeypatch.setattr(actor_module, "is_first_replica_megatron_main_rank", lambda: False)
    monkeypatch.setattr(actor_module, "is_lora_enabled", lambda _args: False)
    return worker, saver, reload_groups


def test_sleep_is_idempotent(monkeypatch):
    worker, saver, _ = _lifecycle_worker(monkeypatch, asleep=False)

    worker.sleep()
    worker.sleep()

    assert saver.pause.call_count == 1
    assert worker._asleep is True


def test_wake_up_when_resident_skips_resume_but_restores_groups(monkeypatch):
    # A retried attempt can die between wake and sleep: memory stays resident but the
    # process groups may already be gone, so wake_up must restore groups without resuming.
    worker, saver, reload_groups = _lifecycle_worker(monkeypatch, asleep=False)

    worker.wake_up()

    saver.resume.assert_not_called()
    reload_groups.assert_called_once_with()
    assert worker._asleep is False


def test_wake_up_resumes_offloaded_model_once(monkeypatch):
    worker, saver, _ = _lifecycle_worker(monkeypatch, asleep=True)

    worker.wake_up()
    worker.wake_up()

    assert saver.resume.call_count == 1
    assert worker._asleep is False
