"""Qwen3.5/3.6 and Qwen3-Next (GatedDeltaNet hybrids): per-document resets under packing come from HF's
own padding-free kwargs, which the vision tower must not see; Qwen3.5-MoE adds the routing-replay hook."""

from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter
from miles.backends.fsdp_utils.adaptations.packing import hf_packing_kwargs
from miles.backends.fsdp_utils.adaptations.routing_replay import RoutingReplayAdapter
from miles.backends.fsdp_utils.models.replay_routers import install_qwen3_router_replay


class Qwen35Adapter(ArchAdapter):
    model_types = frozenset({"qwen3_5", "qwen3_5_text", "qwen3_next"})

    def packing_kwargs(self, *, cu_seqlens, cu_seqlens_host, max_seqlen):
        # GatedDeltaNet reads seq_idx (causal conv) and cu_seq_lens_q (FLA chunk rule) from **kwargs.
        return hf_packing_kwargs(cu_seqlens=cu_seqlens, cu_seqlens_host=cu_seqlens_host, max_seqlen=max_seqlen)

    def patch_model(self, model, args):
        from miles.backends.fsdp_utils.models.qwen3_5 import keep_packing_kwargs_out_of_vision

        keep_packing_kwargs_out_of_vision(model)


class Qwen35MoeAdapter(Qwen35Adapter):
    model_types = frozenset({"qwen3_5_moe", "qwen3_5_moe_text"})
    routing_replay = RoutingReplayAdapter(
        name="qwen3_5_moe", module_cls_name="Qwen3_5MoeTopKRouter", install=install_qwen3_router_replay
    )
