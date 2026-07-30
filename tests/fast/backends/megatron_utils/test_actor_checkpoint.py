from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.megatron_utils.actor import MegatronTrainRayActor


@pytest.mark.parametrize(
    ("rank", "expected_events"),
    [
        (0, ["finalize", "save", "finalize", "hook"]),
        (1, ["finalize", "save", "finalize"]),
    ],
)
def test_async_post_save_hook_finalizes_on_every_rank(rank: int, expected_events: list[str]) -> None:
    events: list[str] = []
    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor._heartbeat = MagicMock()
    actor.args = Namespace(
        async_save=True,
        custom_megatron_post_save_hook_path="test_checkpoint.post_save_hook",
        debug_rollout_only=False,
        offload_train=False,
        save="/checkpoints",
        save_hf=None,
    )
    actor.model = MagicMock()
    actor.optimizer = MagicMock()
    actor.opt_param_scheduler = MagicMock()
    actor.role = "actor"

    def post_save_hook(*_args: object) -> None:
        events.append("hook")

    with (
        patch("miles.backends.megatron_utils.actor.is_multi_lora_enabled", return_value=False),
        patch("miles.backends.megatron_utils.actor.save", side_effect=lambda *_args: events.append("save")),
        patch("miles.backends.megatron_utils.actor.dist.get_rank", return_value=rank),
        patch(
            "megatron.training.async_utils.maybe_finalize_async_save",
            side_effect=lambda **_kwargs: events.append("finalize"),
        ),
        patch("miles.utils.misc.load_function", return_value=post_save_hook),
    ):
        actor.save_model(rollout_id=7)

    assert events == expected_events


def test_finalize_async_save_reports_queue_state() -> None:
    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = Namespace(async_save=True, offload_train=False)

    with (
        patch("megatron.training.async_utils.maybe_finalize_async_save") as finalize,
        patch("megatron.training.async_utils.is_empty_async_queue", return_value=False),
    ):
        completed = actor.finalize_async_save(blocking=False)

    assert completed is False
    finalize.assert_called_once_with(blocking=False)
