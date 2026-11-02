from types import SimpleNamespace

from tests.fast.fixtures.qwen_provider_fixtures import qwen_provider_sdk

from miles.backends.training_utils.model_companion import ModelCompanion
from miles_plugins.models.qwen3_8_next.model_provider import get_qwen3_8_next_model_provider

__all__ = ["qwen_provider_sdk"]


class TestQwenProviderConfig:
    def test_the_base_provider_is_selected_without_mutating_the_custom_provider_request(
        self, qwen_provider_sdk: SimpleNamespace
    ) -> None:
        """A frozen custom-provider config survives base-model construction without recursive dispatch."""
        args = qwen_provider_sdk.args
        original = args.model_dump()

        model = get_qwen3_8_next_model_provider(pre_process=False, post_process=True, vp_stage=3, args=args)

        assert args.model_dump() == original
        [constructed] = qwen_provider_sdk.constructed
        assert constructed["pre_process"] is False
        assert constructed["post_process"] is True
        assert constructed["vp_stage"] == 3
        assert constructed["vocab_size"] == 8
        assert any(isinstance(module, ModelCompanion) for module in model.modules())
