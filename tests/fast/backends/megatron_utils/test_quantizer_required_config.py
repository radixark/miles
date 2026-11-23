from collections.abc import Callable

import pytest
import torch

from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_mxfp8 import quantize_params_mxfp8
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_nvfp4 import quantize_params_nvfp4


class TestQuantizerRequiredConfig:
    @pytest.mark.parametrize(
        "quantizer,config",
        [(quantize_params_mxfp8, {"quant_method": "mxfp8"}), (quantize_params_nvfp4, {"quant_method": "nvfp4"})],
    )
    def test_missing_config_is_rejected_even_when_the_weight_would_otherwise_bypass_quantization(
        self, quantizer: Callable, config: dict[str, str]
    ) -> None:
        """A non-decoder weight does not conceal a missing trainer configuration."""
        weights = [("model.embed_tokens.weight", torch.ones(2, 2))]

        with pytest.raises(AttributeError):
            quantizer(None, "embedding.word_embeddings.weight", weights, config)

        assert torch.equal(weights[0][1], torch.ones(2, 2))
