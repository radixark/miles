from types import SimpleNamespace

import pytest
from tests.fast.backends.megatron_utils.test_dsv4_bridge_runtime import _runtime_args

from miles.backends.megatron_utils.model_provider import _apply_bridge_runtime_config


class TestBridgeOptionalConfig:
    @pytest.mark.parametrize("explicit", [False, True])
    def test_only_explicit_pipeline_and_moe_overrides_replace_provider_defaults(self, explicit: bool) -> None:
        """Absent overrides preserve HF provider settings while explicit zero MoE coefficients remain meaningful."""
        provider = SimpleNamespace(
            num_layers_in_first_pipeline_stage=7,
            num_layers_in_last_pipeline_stage=8,
            moe_router_bias_update_rate=0.5,
            moe_aux_loss_coeff=0.6,
        )
        args = _runtime_args(
            decoder_first_pipeline_num_layers=2 if explicit else None,
            decoder_last_pipeline_num_layers=3 if explicit else None,
            moe_router_bias_update_rate=0.0 if explicit else None,
            moe_aux_loss_coeff=0.0 if explicit else None,
        )

        _apply_bridge_runtime_config(provider, args)

        assert provider.num_layers_in_first_pipeline_stage == (2 if explicit else 7)
        assert provider.num_layers_in_last_pipeline_stage == (3 if explicit else 8)
        assert provider.moe_router_bias_update_rate == (0.0 if explicit else 0.5)
        assert provider.moe_aux_loss_coeff == (0.0 if explicit else 0.6)
