"""Megatron's command-line defaults for the precision args the bridge provider builders copy."""

BRIDGE_PRECISION_ARG_DEFAULTS = {
    "fp8": None,
    "fp8_recipe": "delayed",
    "fp8_param_gather": False,
    "fp8_quantizer_factory": None,
    "fp8_margin": 0,
    "fp8_interval": 1,
    "fp8_amax_history_len": 1,
    "fp8_amax_compute_algo": "most_recent",
    "fp8_wgrad": True,
    "fp8_output_proj": False,
    "fp4": None,
    "fp4_recipe": "nvfp4",
    "fp4_param_gather": False,
    "fp4_quantizer_factory": None,
    "first_last_layers_bf16": False,
    "num_layers_at_start_in_bf16": 1,
    "num_layers_at_end_in_bf16": 1,
    "te_precision_config_file": None,
}
