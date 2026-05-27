from abc import ABC, abstractmethod


class HfWeightIteratorBase(ABC):
    @staticmethod
    def create(args, model, **kwargs):
        mode = args.megatron_to_hf_mode
        if mode == "raw":
            from .hf_weight_iterator_direct import HfWeightIteratorDirect

            return HfWeightIteratorDirect(args, model, **kwargs)
        if mode == "bridge":
            from .hf_weight_iterator_bridge import HfWeightIteratorBridge

            return HfWeightIteratorBridge(args, model, **kwargs)
        raise KeyError(mode)

    def __init__(self, args, model, model_name, quantization_config, **kwargs):
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config

    @abstractmethod
    def get_hf_weight_chunks(self, megatron_local_weights, weight_type="base"):
        """
        Mental model of the API:
        megatron_model.to_hf_magically().named_parameters()
        """
        raise NotImplementedError
