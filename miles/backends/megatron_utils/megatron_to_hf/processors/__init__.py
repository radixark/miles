from .padding_remover import remove_padding

__all__ = [
    "remove_padding",
    "quantize_params",
]


def quantize_params(args, megatron_name, converted_named_params, quantization_config):
    if quantization_config is None:
        return converted_named_params
    elif quantization_config["quant_method"] == "fp8":
        from .quantizer_fp8 import quantize_params_fp8

        return quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config["quant_method"] == "mxfp8":
        from .quantizer_mxfp8 import quantize_params_mxfp8

        return quantize_params_mxfp8(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config["quant_method"] == "compressed-tensors":
        # only int4 at the moment.
        from .quantizer_compressed_tensors import quantize_params_compressed_tensors

        return quantize_params_compressed_tensors(converted_named_params, quantization_config)
