from abc import ABC


class HfWeightIteratorBase(ABC):
    def __init__(self, args, model, model_name, quantization_config):
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config

    def get_hf_weight_chunks(self, megatron_local_weights):
        """
        Mental model of the API:
        megatron_model.to_hf_magically().named_parameters()
        """
        raise NotImplementedError
