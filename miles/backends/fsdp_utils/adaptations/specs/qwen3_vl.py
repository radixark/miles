"""Qwen3-VL runs on the stock HF path with its packed boundaries, kept out of the vision tower."""

from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter
from miles.backends.fsdp_utils.adaptations.packing import keep_packing_kwargs_out_of_vision


class Qwen3VLAdapter(ArchAdapter):
    model_types = frozenset({"qwen3_vl"})
    verified = True
    passes_attention_packing_kwargs = True

    def patch_model(self, model, args):
        keep_packing_kwargs_out_of_vision(model)
