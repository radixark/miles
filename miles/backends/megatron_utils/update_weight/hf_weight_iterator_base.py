from abc import ABC, abstractmethod


class HfWeightIteratorBase(ABC):
    @staticmethod
    def create(args, model, *, is_lora=False, **kwargs):
        from .hf_weight_iterator_bridge import HfWeightIteratorBridge
        from .hf_weight_iterator_direct import HfWeightIteratorDirect

        c = {
            "raw": HfWeightIteratorDirect,
            "bridge": HfWeightIteratorBridge,
        }[args.megatron_to_hf_mode]

        return c(args, model, is_lora=is_lora, **kwargs)

    def __init__(self, args, model, model_name, quantization_config, *, is_lora=False):
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.is_lora = is_lora

    @abstractmethod
    def get_hf_weight_chunks(self, megatron_local_weights):
        """
        Mental model of the API:
        megatron_model.to_hf_magically().named_parameters()
        """
        raise NotImplementedError
