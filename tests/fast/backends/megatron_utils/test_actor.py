import importlib
import sys
from argparse import Namespace
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

_ACTOR_MODULE_NAME = "miles.backends.megatron_utils.actor"


@pytest.fixture(scope="module")
def actor_module() -> Iterator[ModuleType]:
    """Import the Megatron actor with its unavailable native memory dependency stubbed."""
    package = importlib.import_module("miles.backends.megatron_utils")
    missing = object()
    saved_module = sys.modules.get(_ACTOR_MODULE_NAME, missing)
    saved_saver = sys.modules.get("torch_memory_saver", missing)
    saved_package_attr = getattr(package, "actor", missing)

    saver_module = ModuleType("torch_memory_saver")
    saver_module.torch_memory_saver = Mock()
    sys.modules["torch_memory_saver"] = saver_module
    sys.modules.pop(_ACTOR_MODULE_NAME, None)
    if saved_package_attr is not missing:
        delattr(package, "actor")

    try:
        yield importlib.import_module(_ACTOR_MODULE_NAME)
    finally:
        sys.modules.pop(_ACTOR_MODULE_NAME, None)
        if saved_module is not missing:
            sys.modules[_ACTOR_MODULE_NAME] = saved_module
        if saved_package_attr is missing:
            if hasattr(package, "actor"):
                delattr(package, "actor")
        else:
            package.actor = saved_package_attr
        if saved_saver is missing:
            sys.modules.pop("torch_memory_saver", None)
        else:
            sys.modules["torch_memory_saver"] = saved_saver


class TestCriticValuesValueSpec:
    def test_critic_values_are_shipped_as_a_typed_ragged_field(self, actor_module: ModuleType) -> None:
        """Variable-length critic sequences require the typed ragged object-store codec."""
        assert actor_module.CRITIC_VALUES_VALUE_SPEC["values"].codec == "typed_ragged"


class TestSendCheckpoint:
    def test_healing_before_the_first_train_step_is_refused_without_sending(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Healing before the first train step is refused without transferring a checkpoint."""
        train_actor = object.__new__(actor_module.MegatronTrainRayActor)
        train_actor.args = Namespace(keep_old_actor=False)
        train_actor._last_rollout_id = None
        train_actor.model = object()
        train_actor.optimizer = object()
        train_actor.opt_param_scheduler = object()
        checkpoint_transfer_attempted = False

        def record_checkpoint_transfer(**_kwargs: object) -> None:
            nonlocal checkpoint_transfer_attempted
            checkpoint_transfer_attempted = True

        monkeypatch.setattr(actor_module, "get_parallel_state", lambda: SimpleNamespace(indep_dp=object()))
        monkeypatch.setattr(actor_module, "_send_ckpt", record_checkpoint_transfer)

        with pytest.raises(AssertionError, match="healing before the first train step is unsupported"):
            train_actor.send_ckpt(dst_rank=1)

        assert not checkpoint_transfer_attempted
