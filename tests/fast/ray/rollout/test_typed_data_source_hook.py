import pytest
from tests.fast.fixtures.hook_consumer_fixtures import _hook
from tests.fast.ray.rollout.conftest import make_rollout_config

from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.utils import object_store
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.function_registry import function_registry


class TestTypedDataSourceHook:
    async def test_executor_initialization_passes_plugin_fields_to_the_actual_source_constructor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real executor initialization gives custom data sources flat resolved plugin settings."""
        hook = _hook()
        args = make_rollout_config(
            debug_train_only=True, data_source_path=hook, object_store_backend="ray", use_wandb=False, use_mlflow=False
        )
        monkeypatch.setattr(object_store, "_INSTANCE", None)
        executor = RolloutExecutor(
            args=args, router_providers=[], session_server_provider=None, inference_controller_provider=object()
        )

        class HookReached(Exception):
            pass

        def source(view: ImmutableNamespace) -> None:
            assert view.hook_marker == 41
            assert view.debug_train_only is True
            assert "hook_marker" not in dict(args)
            raise HookReached

        with function_registry.temporary(hook.path, source), pytest.raises(HookReached):
            await executor.init()
