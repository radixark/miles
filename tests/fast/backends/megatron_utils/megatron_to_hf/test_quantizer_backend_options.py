from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8 import quantize_params_fp8
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_mxfp8 import quantize_params_mxfp8
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_nvfp4 import quantize_params_nvfp4


class TestIndexerQuantizationOwnership:
    @pytest.mark.parametrize("interleaved", [False, True])
    def test_fp8_indexer_precision_follows_backend_rope_layout(self, interleaved: bool) -> None:
        """Standalone indexer keys are quantized only when the backend RoPE layout leaves them unfused."""
        args = SimpleNamespace(backend=MegatronArgsNamespace(indexer_rope_interleave=interleaved))
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        values = [("model.layers.0.self_attn.indexer.wk.weight", weight)]

        result = quantize_params_fp8(
            args=args,
            megatron_name="module.module.decoder.layers.0.self_attention.core_attention.indexer.linear_wk.weight",
            converted_named_params=values,
            quantization_config={"quant_method": "fp8", "activation_scheme": "dynamic"},
        )

        if interleaved:
            assert result is values
        else:
            assert [name for name, _ in result] == [
                "model.layers.0.self_attn.indexer.wk.weight",
                "model.layers.0.self_attn.indexer.wk.weight_scale",
            ]
            assert result[0][1].dtype == torch.float8_e4m3fn
            torch.testing.assert_close(result[1][1], torch.tensor([4 / 448]))
            torch.testing.assert_close(result[0][1].float() * result[1][1], weight, rtol=0.0625, atol=0.0)

    @pytest.mark.parametrize("interleaved", [False, True])
    def test_mxfp8_indexer_precision_follows_backend_rope_layout(
        self, transformer_engine_quantizer: Mock, interleaved: bool
    ) -> None:
        """MXFP8 export respects backend indexer layout before invoking the external quantizer."""
        args = SimpleNamespace(
            backend=MegatronArgsNamespace(indexer_rope_interleave=interleaved),
            extra_high_precision_layers_megatron=None,
        )
        quantized = SimpleNamespace(
            _rowwise_data=torch.zeros(32, 32, dtype=torch.uint8),
            _rowwise_scale_inv=torch.ones(32, 1, dtype=torch.uint8),
        )
        sdk = transformer_engine_quantizer
        sdk.return_value.quantize.return_value = quantized
        values = [("model.layers.0.self_attn.indexer.wk.weight", torch.ones(1, 32))]

        result = quantize_params_mxfp8(
            args=args,
            megatron_name="module.module.decoder.layers.0.self_attention.core_attention.indexer.linear_wk.weight",
            converted_named_params=values,
            quantization_config={"quant_method": "mxfp8"},
        )

        if interleaved:
            assert result is values
            sdk.assert_not_called()
        else:
            sdk.assert_called_once()
            sdk.return_value.quantize.assert_called_once()
            assert [name for name, _ in result] == [
                "model.layers.0.self_attn.indexer.wk.weight",
                "model.layers.0.self_attn.indexer.wk.weight_scale_inv",
            ]
            assert result[0][1].dtype == torch.float8_e4m3fn
            assert result[0][1].shape == (1, 32)
            assert result[1][1].shape == (1, 1)


class TestNvfp4BackendGuards:
    @pytest.mark.parametrize("flag", ["fp4_param", "fp4_param_gather"])
    def test_backend_fp4_parameter_modes_are_rejected_before_quantization(self, flag: str) -> None:
        """Backend FP4 parameter storage is refused before NVFP4 checkpoint export touches weights."""
        args = SimpleNamespace(
            backend=MegatronArgsNamespace(**{flag: True}), extra_high_precision_layers_megatron=None
        )

        with pytest.raises(NotImplementedError, match="fp4-param-gather is unsupported"):
            quantize_params_nvfp4(
                args=args,
                megatron_name="unused",
                converted_named_params=[],
                quantization_config={"quant_method": "nvfp4"},
            )
