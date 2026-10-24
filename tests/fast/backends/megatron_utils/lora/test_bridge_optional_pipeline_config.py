from argparse import Namespace
from typing import Any

from tests.fast.backends.megatron_utils.lora.bridge_config_fixtures import bridge_args, bridge_provider

from miles.backends.megatron_utils.lora.bridge import _setup_lora_model_via_bridge

__all__ = ["bridge_args", "bridge_provider"]


class TestBridgeOptionalPipelineConfig:
    def test_absent_pipeline_overrides_preserve_both_sdk_stage_defaults(
        self, bridge_args: Namespace, bridge_provider: Any
    ) -> None:
        """Absent layer-count overrides preserve the SDK provider's existing stage partition."""
        bridge_args.backend.decoder_first_pipeline_num_layers = None
        bridge_args.backend.decoder_last_pipeline_num_layers = None
        bridge_provider.num_layers_in_first_pipeline_stage = 7
        bridge_provider.num_layers_in_last_pipeline_stage = 8

        models = _setup_lora_model_via_bridge(bridge_args)

        assert models == [bridge_provider]
        assert bridge_provider.finalized
        assert bridge_provider.num_layers_in_first_pipeline_stage == 7
        assert bridge_provider.num_layers_in_last_pipeline_stage == 8
