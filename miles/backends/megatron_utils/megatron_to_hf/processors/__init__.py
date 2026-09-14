from .padding_remover import remove_padding
from .quantizer_compressed_tensors import quantize_params_compressed_tensors
from .quantizer_fp8 import quantize_params_fp8
from .quantizer_mxfp4 import is_routed_expert_param, quantize_params_mxfp4
from .quantizer_mxfp8 import quantize_params_mxfp8
from .quantizer_nvfp4 import quantize_params_nvfp4

__all__ = [
    "remove_padding",
    "quantize_param",
    "quantize_params_fp8",
    "quantize_params_mxfp4",
    "quantize_params_mxfp8",
    "quantize_params_nvfp4",
    "quantize_params_compressed_tensors",
]


def quantize_params(args, megatron_name, converted_named_params, quantization_config):
    if quantization_config is None:
        return converted_named_params
    elif quantization_config["quant_method"] == "fp8":
        if getattr(args, "rollout_fp4_experts", False) and is_routed_expert_param(megatron_name):
            return quantize_params_mxfp4(converted_named_params)
        return quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config["quant_method"] == "mxfp8":
        return quantize_params_mxfp8(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config.get("quant_algo") == "NVFP4" or quantization_config["quant_method"] == "nvfp4":
        return quantize_params_nvfp4(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config["quant_method"] == "compressed-tensors":
        # only int4 at the moment.
        return quantize_params_compressed_tensors(converted_named_params, quantization_config)
