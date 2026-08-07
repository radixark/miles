from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.backends.experimental.fsdp_utils.adaptations import routing_replay
from miles.utils.replay_base import routing_replay_manager


@pytest.fixture(autouse=True)
def _reset_manager():
    routing_replay_manager.enabled = False
    routing_replay_manager.stage = routing_replay.FALLTHROUGH
    yield
    routing_replay_manager.enabled = False
    routing_replay_manager.stage = routing_replay.FALLTHROUGH


def test_enable_follows_use_routing_replay():
    assert routing_replay.enable(Namespace(use_routing_replay=True, ci_test=False)) is True
    assert routing_replay.enable(Namespace(use_routing_replay=False, ci_test=False)) is False


def test_enable_defaults_false_when_arg_absent():
    assert routing_replay.enable(Namespace(ci_test=False)) is False


def test_enable_turns_on_the_replay_check_only_under_ci_test():
    routing_replay.enable(Namespace(use_routing_replay=True, ci_test=True))
    assert routing_replay_manager.enable_check_replay_result is True

    routing_replay.enable(Namespace(use_routing_replay=True, ci_test=False))
    assert routing_replay_manager.enable_check_replay_result is False


def test_log_prob_stage_is_fallthrough_when_disabled():
    routing_replay_manager.enabled = False
    assert routing_replay.log_prob_stage(Namespace(use_rollout_routing_replay=True)) == routing_replay.FALLTHROUGH


def test_log_prob_stage_replays_when_routing_came_from_the_rollout():
    routing_replay_manager.enabled = True
    assert routing_replay.log_prob_stage(Namespace(use_rollout_routing_replay=True)) == routing_replay.REPLAY_FORWARD


def test_log_prob_stage_records_for_the_non_rollout_variant():
    # --use-routing-replay alone has nothing to replay yet: the queues are only filled by fill(),
    # which is gated on use_rollout_routing_replay. Replaying here would pop an empty queue.
    routing_replay_manager.enabled = True
    assert routing_replay.log_prob_stage(Namespace(use_rollout_routing_replay=False)) == routing_replay.RECORD


def test_fill_is_skipped_for_the_non_rollout_variant():
    # Passing None for every downstream argument: reaching fill_replay_data would raise.
    routing_replay.fill(Namespace(use_rollout_routing_replay=False), None, None, None, None)


def test_stage_sets_and_restores():
    routing_replay_manager.stage = routing_replay.REPLAY_BACKWARD
    with routing_replay.stage(routing_replay.REPLAY_FORWARD):
        assert routing_replay_manager.stage == routing_replay.REPLAY_FORWARD
    assert routing_replay_manager.stage == routing_replay.REPLAY_BACKWARD


def test_stage_restores_on_exception():
    routing_replay_manager.stage = routing_replay.REPLAY_BACKWARD
    with pytest.raises(RuntimeError):
        with routing_replay.stage(routing_replay.REPLAY_FORWARD):
            raise RuntimeError("boom")
    assert routing_replay_manager.stage == routing_replay.REPLAY_BACKWARD


def test_stage_nests():
    routing_replay_manager.stage = routing_replay.FALLTHROUGH
    with routing_replay.stage(routing_replay.REPLAY_BACKWARD):
        with routing_replay.stage(routing_replay.REPLAY_FORWARD):
            assert routing_replay_manager.stage == routing_replay.REPLAY_FORWARD
        assert routing_replay_manager.stage == routing_replay.REPLAY_BACKWARD
    assert routing_replay_manager.stage == routing_replay.FALLTHROUGH


def test_ref_model_creation_does_not_install_routing_replay():
    import inspect

    from miles.backends.experimental.fsdp_utils.actor import FSDPTrainRayActor

    source = inspect.getsource(FSDPTrainRayActor._create_ref_model)
    assert "routing_replay.install" not in source
