import json
import os
from abc import ABC, abstractmethod


class HfWeightIteratorBase(ABC):
    @staticmethod
    def create(args, model, **kwargs):
        from .hf_weight_iterator_bridge import HfWeightIteratorBridge
        from .hf_weight_iterator_direct import HfWeightIteratorDirect

        c = {
            "raw": HfWeightIteratorDirect,
            "bridge": HfWeightIteratorBridge,
        }[args.megatron_to_hf_mode]

        return c(args, model, **kwargs)

    def __init__(self, args, model, model_name, quantization_config, **kwargs):
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = _with_checkpoint_quantized_param_basenames(quantization_config, args.hf_checkpoint)

    @abstractmethod
    def get_hf_weight_chunks(self, megatron_local_weights, weight_type="base"):
        """
        Mental model of the API:
        megatron_model.to_hf_magically().named_parameters()
        """
        raise NotImplementedError


def _with_checkpoint_quantized_param_basenames(quantization_config, hf_checkpoint):
    if quantization_config is None or quantization_config.get("quant_method") != "compressed-tensors":
        return quantization_config

    index_path = os.path.join(hf_checkpoint, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return quantization_config
    with open(index_path) as f:
        names = json.load(f)["weight_map"]
    return {
        **quantization_config,
        "_miles_quantized_basenames": {
            name.removesuffix(".weight_packed") for name in names if name.endswith(".weight_packed")
        },
    }
