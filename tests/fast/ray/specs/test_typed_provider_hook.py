from tests.fast.fixtures.hook_consumer_fixtures import _hook

from miles.ray.specs.inference import compute_engine_provider
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.function_registry import function_registry


class TestTypedProviderHook:
    def test_engine_provider_factory_receives_flat_plugin_fields_and_backend_capability(self) -> None:
        """Engine provider selection retains custom settings and the exact backend capability object."""
        hook = _hook()
        args = ImmutableNamespace(custom_inference_engine_provider_path=hook, rollout_num_gpus=6)
        capability = object()
        provider = object()

        def factory(view: ImmutableNamespace, *, capability: object) -> object:
            assert view.hook_marker == 41
            assert view.rollout_num_gpus == 6
            assert capability is expected_capability
            return provider

        expected_capability = capability
        with function_registry.temporary(hook.path, factory):
            assert compute_engine_provider(args, capability=capability) is provider
        assert "hook_marker" not in dict(args)
