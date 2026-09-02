import importlib
import sys
from argparse import Namespace
from contextlib import contextmanager, nullcontext
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from miles.utils.replay_base import IndexerReplayManager, RoutingReplayManager


@pytest.fixture(scope="module")
def actor_module():
    actor_module_name = "miles.backends.megatron_utils.actor"
    p2p_module_name = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p"
    actor_package = importlib.import_module("miles.backends.megatron_utils")
    p2p_package = importlib.import_module("miles.backends.megatron_utils.update_weight.update_weight_from_distributed")
    missing = object()
    saved_actor_module = sys.modules.get(actor_module_name, missing)
    saved_p2p_module = sys.modules.get(p2p_module_name, missing)
    saved_saver = sys.modules.get("torch_memory_saver", missing)
    saved_actor_package_attr = getattr(actor_package, "actor", missing)
    saved_p2p_package_attr = getattr(p2p_package, "p2p", missing)

    saver_module = ModuleType("torch_memory_saver")
    saver_module.torch_memory_saver = Mock()
    p2p_module = ModuleType(p2p_module_name)
    p2p_module.UpdateWeightP2P = Mock(
        side_effect=AssertionError("shared PPO lifecycle tests must not construct UpdateWeightP2P")
    )
    sys.modules["torch_memory_saver"] = saver_module
    sys.modules[p2p_module_name] = p2p_module
    p2p_package.p2p = p2p_module
    sys.modules.pop(actor_module_name, None)
    if saved_actor_package_attr is not missing:
        delattr(actor_package, "actor")

    try:
        yield importlib.import_module(actor_module_name)
    finally:
        sys.modules.pop(actor_module_name, None)
        if saved_actor_module is not missing:
            sys.modules[actor_module_name] = saved_actor_module
        if saved_actor_package_attr is missing:
            if hasattr(actor_package, "actor"):
                delattr(actor_package, "actor")
        else:
            actor_package.actor = saved_actor_package_attr
        sys.modules.pop(p2p_module_name, None)
        if saved_p2p_module is not missing:
            sys.modules[p2p_module_name] = saved_p2p_module
        if saved_p2p_package_attr is missing:
            if hasattr(p2p_package, "p2p"):
                delattr(p2p_package, "p2p")
        else:
            p2p_package.p2p = saved_p2p_package_attr
        if saved_saver is missing:
            sys.modules.pop("torch_memory_saver", None)
        else:
            sys.modules["torch_memory_saver"] = saved_saver


def _worker(actor_module, role, *, asleep=True):
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace(offload_train=True, debug_rollout_only=False)
    worker.role = role
    worker._asleep = asleep
    worker._heartbeat = Mock()
    worker.wake_up = Mock()
    worker.sleep = Mock()
    return worker


def test_critic_train_wakes_and_leaves_offload_to_driver(actor_module, monkeypatch):
    worker = _worker(actor_module, "critic")
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

    result = worker.train(3, object())

    worker.wake_up.assert_called_once_with()
    worker.train_critic.assert_called_once()
    worker.sleep.assert_not_called()
    assert result == {"values": ["cpu-value"]}
    assert phases == ["data_preprocess", "critic_train"]


def test_actor_receives_critic_payload_and_leaves_offload_to_driver(actor_module, monkeypatch):
    worker = _worker(actor_module, "actor")
    worker.train_actor = Mock(return_value=None)
    monkeypatch.setattr(
        actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: ({"tokens": []}, nullcontext())
    )
    values = {"values": ["cpu-value"]}

    result = worker.train(4, object(), external_data=values)

    worker.wake_up.assert_called_once_with()
    worker.train_actor.assert_called_once()
    assert worker.train_actor.call_args.kwargs["external_data"] is values
    worker.sleep.assert_not_called()
    assert result is None


def test_train_keeps_model_resident(actor_module, monkeypatch):
    worker = _worker(actor_module, "actor", asleep=False)
    worker.train_actor = Mock(return_value=None)
    monkeypatch.setattr(
        actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: ({"tokens": []}, nullcontext())
    )

    worker.train(5, object())

    worker.wake_up.assert_not_called()
    worker.sleep.assert_not_called()


def test_compute_log_prob_keeps_logits_in_model_precision(actor_module, monkeypatch):
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace()
    worker.model = [object()]
    forward_only = Mock(return_value={"log_probs": []})
    monkeypatch.setattr(actor_module, "forward_only", forward_only)
    monkeypatch.setattr(actor_module, "timer", lambda _name: nullcontext())

    result = worker.compute_log_prob([], [], rollout_id=3)

    assert result == {"log_probs": []}
    assert forward_only.call_args.kwargs["fp32_output"] is False


def test_save_model_does_not_manage_lifecycle(actor_module, monkeypatch):
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace(
        async_save=False,
        custom_megatron_post_save_hook_path=None,
        debug_rollout_only=False,
        save_hf=None,
    )
    worker.role = "actor"
    worker._heartbeat = Mock()
    worker.model = object()
    worker.optimizer = object()
    worker.opt_param_scheduler = object()
    worker.wake_up = Mock()
    worker.sleep = Mock()
    save = Mock()
    reload_groups = Mock()
    destroy_groups = Mock()
    monkeypatch.setattr(actor_module, "save", save)
    monkeypatch.setattr(actor_module, "reload_process_groups", reload_groups)
    monkeypatch.setattr(actor_module, "destroy_process_groups", destroy_groups)

    worker.save_model(6)

    save.assert_called_once_with(6, worker.model, worker.optimizer, worker.opt_param_scheduler)
    worker.wake_up.assert_not_called()
    worker.sleep.assert_not_called()
    reload_groups.assert_not_called()
    destroy_groups.assert_not_called()


@pytest.mark.parametrize("asleep", [False, True])
def test_update_weights_only_uses_temporary_process_groups_when_asleep(actor_module, monkeypatch, asleep):
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace(
        debug_rollout_only=False,
        debug_skip_weight_update=True,
        debug_train_only=False,
        offload_train=True,
        rematerialize_param_from_master_weight=False,
    )
    worker._asleep = asleep
    worker._heartbeat = Mock()
    worker.weight_updater = Mock()
    worker.weight_updater.is_rollout_engines_fresh.return_value = True
    info = Namespace(
        engine_gpu_counts=[],
        engine_gpu_offsets=[],
        has_new_engines=False,
        rollout_engine_lock=None,
        rollout_engines=[],
    )
    reload_groups = Mock()
    destroy_groups = Mock()
    monkeypatch.setattr(actor_module, "reload_process_groups", reload_groups)
    monkeypatch.setattr(actor_module, "destroy_process_groups", destroy_groups)
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 1)

    worker.update_weights(info)

    assert reload_groups.call_count == int(asleep)
    assert destroy_groups.call_count == int(asleep)


def _lifecycle_worker(actor_module, monkeypatch, asleep):
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace(
        offload_train=True,
        rematerialize_param_from_master_weight=False,
        clear_quantized_weight_workspaces_on_offload=False,
    )
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


def test_sleep_is_idempotent(actor_module, monkeypatch):
    worker, saver, _ = _lifecycle_worker(actor_module, monkeypatch, asleep=False)

    worker.sleep()
    worker.sleep()

    assert saver.pause.call_count == 1
    assert worker._asleep is True


def test_wake_up_when_resident_skips_resume_but_restores_groups(actor_module, monkeypatch):
    # A retried attempt can die between wake and sleep: memory stays resident but the
    # process groups may already be gone, so wake_up must restore groups without resuming.
    worker, saver, reload_groups = _lifecycle_worker(actor_module, monkeypatch, asleep=False)

    worker.wake_up()

    saver.resume.assert_not_called()
    reload_groups.assert_called_once_with()
    assert worker._asleep is False


def test_wake_up_resumes_offloaded_model_once(actor_module, monkeypatch):
    worker, saver, _ = _lifecycle_worker(actor_module, monkeypatch, asleep=True)

    worker.wake_up()
    worker.wake_up()

    assert saver.resume.call_count == 1
    assert worker._asleep is False


def _actor_train_args(**overrides):
    defaults = dict(
        compute_advantages_and_returns=True,
        use_rollout_logprobs=False,
        keep_old_actor=False,
        get_mismatch_metrics=False,
        skip_actor_forward_only=False,
    )
    return Namespace(**(defaults | overrides))


def _actor_reuse_worker(actor_module, **args_overrides):
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = _actor_train_args(use_critic=False, **args_overrides)
    worker.model = [object()]
    worker.optimizer = object()
    worker.opt_param_scheduler = object()
    worker.weights_backuper = Mock(backup_tags=set())
    worker._active_model_tag = "actor"
    worker._switch_model = Mock()
    worker._set_replay_stage = Mock()
    worker.compute_log_prob = Mock(return_value={"log_probs": [object()]})
    worker.rollout_data_postprocess = None
    worker.prof = Mock()
    worker._ft_test_action_executor = None
    worker.weight_updater = Mock()
    worker.weight_updater.pop_metrics.return_value = {}
    worker._heartbeat = Mock()
    return worker


def _patch_actor_reuse_dependencies(actor_module, monkeypatch, *, num_microbatches):
    @contextmanager
    def passthrough_timer(_name):
        yield

    monkeypatch.setattr(actor_module, "all_replay_managers", [])
    monkeypatch.setattr(
        actor_module,
        "get_data_iterator",
        lambda *_args: ([Namespace(micro_batch_indices=None, micro_batch_size=1)], num_microbatches),
    )
    monkeypatch.setattr(actor_module, "compute_advantages_and_returns", Mock())
    monkeypatch.setattr(actor_module, "log_train_advantage_computation_event", Mock())
    monkeypatch.setattr(actor_module, "log_rollout_data", Mock())
    monkeypatch.setattr(actor_module, "log_perf_data", Mock())
    monkeypatch.setattr(actor_module.train_dump_utils, "save_debug_train_data", Mock())
    monkeypatch.setattr(actor_module, "inverse_timer", passthrough_timer)
    monkeypatch.setattr(actor_module, "timer", passthrough_timer)
    monkeypatch.setattr(
        actor_module,
        "train",
        Mock(return_value=actor_module.TrainStepOutcome.DISCARDED_SHOULD_RETRY),
    )


@pytest.mark.parametrize(
    ("skip_actor_forward_only", "use_rollout_logprobs", "num_microbatches"),
    [
        (False, False, [1]),
        (True, False, [1]),
        (True, True, [1]),
        (True, False, [2]),
    ],
)
def test_actor_logprob_forward_is_explicit_single_step_opt_in(
    actor_module, monkeypatch, skip_actor_forward_only, use_rollout_logprobs, num_microbatches
):
    worker = _actor_reuse_worker(
        actor_module,
        skip_actor_forward_only=skip_actor_forward_only,
        use_rollout_logprobs=use_rollout_logprobs,
    )
    _patch_actor_reuse_dependencies(actor_module, monkeypatch, num_microbatches=num_microbatches)
    rollout_data = {
        "num_rollouts": [1] * len(num_microbatches),
        "total_lengths": [1] * sum(num_microbatches),
    }

    worker.train_actor(7, rollout_data, witness_info=None, attempt=0)

    assert worker.compute_log_prob.call_count == int(not skip_actor_forward_only and not use_rollout_logprobs)
    actor_module.compute_advantages_and_returns.assert_called_once_with(worker.args, rollout_data)
    train_call = actor_module.train.call_args
    assert train_call.args[6] is rollout_data["num_rollouts"]
    assert train_call.kwargs == {
        "witness_info": None,
        "attempt": 0,
        "ft_test_action_executor": None,
        "forward_only": False,
    }


def test_skip_actor_forward_only_preserves_reference_teacher_and_training_forwards(actor_module, monkeypatch):
    worker = _actor_reuse_worker(actor_module, skip_actor_forward_only=True)
    worker.weights_backuper.backup_tags = {"ref", "teacher"}
    worker.compute_log_prob.side_effect = lambda *_args, store_prefix, **_kwargs: {
        f"{store_prefix}log_probs": [object()]
    }
    _patch_actor_reuse_dependencies(actor_module, monkeypatch, num_microbatches=[1])
    rollout_data = {"num_rollouts": [1], "total_lengths": [1]}

    worker.train_actor(7, rollout_data, witness_info=None, attempt=0)

    assert [call.kwargs["store_prefix"] for call in worker.compute_log_prob.call_args_list] == ["ref_", "teacher_"]
    actor_module.train.assert_called_once()


@pytest.mark.parametrize(
    ("manager_cls", "rollout_flag", "data_key"),
    [
        (RoutingReplayManager, "use_rollout_routing_replay", "rollout_routed_experts"),
        (IndexerReplayManager, "use_rollout_indexer_replay", "rollout_indexer_topk"),
    ],
)
def test_skip_actor_forward_only_consumes_preloaded_rollout_replay_during_training(
    actor_module,
    monkeypatch,
    manager_cls,
    rollout_flag,
    data_key,
):
    manager = manager_cls()
    manager.enabled = True
    manager.enable_check_replay_result = False
    queued_top_indices = []
    replay = Mock()
    replay.record.side_effect = queued_top_indices.append
    replay.pop_backward.side_effect = lambda: queued_top_indices.pop(0)
    manager.replays = [replay]
    manager.set_current(replay)

    worker = _actor_reuse_worker(
        actor_module,
        skip_actor_forward_only=True,
        **{rollout_flag: True},
    )
    _patch_actor_reuse_dependencies(actor_module, monkeypatch, num_microbatches=[1])
    monkeypatch.setattr(actor_module, "all_replay_managers", [manager])
    worker._set_replay_stage.side_effect = lambda stage: setattr(manager, "stage", stage)

    expected_top_indices = torch.tensor([[1]], dtype=torch.int64)

    def preload_replay_data(**kwargs):
        assert kwargs["data_key"] == data_key
        assert kwargs["replay_list"] is manager.replays
        kwargs["replay_list"][0].record(kwargs["rollout_data"].pop(data_key)[0])

    fill_replay_data = Mock(side_effect=preload_replay_data)
    monkeypatch.setattr(actor_module, "fill_replay_data", fill_replay_data)

    def train_with_replay(*_args, **_kwargs):
        topk_fn = manager.get_topk_fn(
            lambda scores, topk: torch.topk(scores, topk, dim=1).indices,
            return_probs=False,
        )
        scores = torch.tensor([[0.0, 1.0]])
        torch.testing.assert_close(topk_fn(scores, 1), expected_top_indices)
        return actor_module.TrainStepOutcome.DISCARDED_SHOULD_RETRY

    train = Mock(side_effect=train_with_replay)
    monkeypatch.setattr(actor_module, "train", train)
    rollout_data = {
        "num_rollouts": [1],
        "total_lengths": [1],
        data_key: [expected_top_indices],
    }

    worker.train_actor(7, rollout_data, witness_info=None, attempt=0)

    worker.compute_log_prob.assert_not_called()
    fill_replay_data.assert_called_once()
    replay.pop_backward.assert_called_once()
    assert queued_top_indices == []


def test_skip_actor_forward_only_rejects_multiple_optimizer_steps(actor_module, monkeypatch):
    worker = _actor_reuse_worker(actor_module, skip_actor_forward_only=True)
    _patch_actor_reuse_dependencies(actor_module, monkeypatch, num_microbatches=[1, 1])
    rollout_data = {"num_rollouts": [1, 1], "total_lengths": [1, 1]}

    with pytest.raises(AssertionError, match="requires 1 optimizer step"):
        worker.train_actor(7, rollout_data, witness_info=None, attempt=0)

    worker.compute_log_prob.assert_not_called()
    actor_module.compute_advantages_and_returns.assert_not_called()
    actor_module.train.assert_not_called()


def test_skip_actor_forward_only_rejects_existing_actor_log_probs(actor_module, monkeypatch):
    worker = _actor_reuse_worker(actor_module, skip_actor_forward_only=True)
    _patch_actor_reuse_dependencies(actor_module, monkeypatch, num_microbatches=[1])
    rollout_data = {"num_rollouts": [1], "total_lengths": [1]}
    rollout_data["log_probs"] = [object()]

    with pytest.raises(AssertionError, match="without actor log probs"):
        worker.train_actor(7, rollout_data, witness_info=None, attempt=0)

    worker.compute_log_prob.assert_not_called()
    actor_module.compute_advantages_and_returns.assert_not_called()
    actor_module.train.assert_not_called()
