from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorDirect(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
