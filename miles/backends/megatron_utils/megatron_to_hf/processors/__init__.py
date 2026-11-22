from .padding_remover import remove_padding
from .quantizer_compressed_tensors import quantize_params_compressed_tensors
from .quantizer_fp8 import is_kept_in_high_precision, quantize_params_fp8
from .quantizer_mxfp4 import is_routed_expert_param, quantize_params_mxfp4
from .quantizer_mxfp8 import quantize_params_mxfp8, quantize_routed_expert_params_mxfp8
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


def quantize_params(args, megatron_name, converted_named_params, quantization_config, packed_weight_basenames=None):
    if quantization_config is None:
        return converted_named_params
    elif quantization_config["quant_method"] == "fp8":
        routed_expert_format = getattr(
            args, "rollout_routed_expert_format", None
        )  # config-access-exempt: offline checkpoint converters pass namespaces without rollout fields
        if (
            routed_expert_format is not None
            and is_routed_expert_param(megatron_name)
            and not any(is_kept_in_high_precision(name, quantization_config) for name, _ in converted_named_params)
        ):
            if routed_expert_format == "mxfp4":
                return quantize_params_mxfp4(converted_named_params)
            return quantize_routed_expert_params_mxfp8(converted_named_params)
        return quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config["quant_method"] == "mxfp8":
        return quantize_params_mxfp8(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config.get("quant_algo") == "NVFP4" or quantization_config["quant_method"] == "nvfp4":
        return quantize_params_nvfp4(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config["quant_method"] == "compressed-tensors":
        assert (
            packed_weight_basenames is not None
        ), "compressed-tensors quantization needs the checkpoint's packed weight names"
        return quantize_params_compressed_tensors(converted_named_params, quantization_config, packed_weight_basenames)
