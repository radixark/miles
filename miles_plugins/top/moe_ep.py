"""Keep Megatron's MoE forward stages and replace the final weighted reduction."""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import torch
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher
from sglang.srt.true_on_policy.qwen_moe_router import SHARED_GATE_HIDDEN_SIZES

from miles_plugins.top.binding import CaptureTap
from miles_plugins.top.kernels.activation import SiluAndMulFunction
from miles_plugins.top.kernels.delegated_linear import delegated_column_linear, delegated_row_linear
from miles_plugins.top.kernels.slot_combine import combine_with_receipt
from miles_plugins.top.moe_deepep import TopDeepEPTokenDispatcher


@dataclass(frozen=True)
class QualifiedExperts:
    """An expert configuration a TOP spec binds; the op registry owns its topology."""

    name: str
    family: str  # HF model_type whose TOP spec builds the layer
    shape: tuple | None  # (hidden, expert FFN, experts, top-k); None: any size of the family
    gated_shared_expert: bool
    shared_expert: int | None  # gated shared expert FFN size; None: any size


QUALIFIED_EXPERTS = (
    QualifiedExperts(
        name="Qwen3-30B-A3B",
        family="qwen3_moe",
        shape=(2048, 768, 128, 8),
        gated_shared_expert=False,
        shared_expert=None,
    ),
    # Every Qwen3.5/3.6 MoE size (35B-A3B to 397B-A17B) runs the same row-wise program; only the counts differ.
    QualifiedExperts(
        name="Qwen3.5/3.6 MoE",
        family="qwen3_5_moe_text",
        shape=None,
        gated_shared_expert=True,
        shared_expert=None,
    ),
    QualifiedExperts(
        name="Qwen3.8-Flash-Next",
        family="qwen4_exp_text",
        shape=(2560, 640, 512, 10),
        gated_shared_expert=True,
        shared_expert=640,
    ),
)


def validate_moe_config(config, family):
    """Admit SiLU experts qualified for ``family`` on the flex DeepEP dispatcher at expert TP=1."""
    if config.activation_func is not torch.nn.functional.silu:
        raise NotImplementedError("[top] EP experts require SiLU")
    shape = (
        config.hidden_size,
        config.moe_ffn_hidden_size,
        config.num_moe_experts,
        config.moe_router_topk,
    )
    qualified = [q for q in QUALIFIED_EXPERTS if q.family == family]
    match = next((q for q in qualified if q.shape in (shape, None)), None)
    if match is None:
        names = ", ".join(q.name for q in qualified)
        raise NotImplementedError(f"[top] expected the {names} expert configuration")
    if match.shape is None and config.hidden_size not in SHARED_GATE_HIDDEN_SIZES:
        raise NotImplementedError(
            f"[top] the TOP shared-expert gate is qualified for hidden sizes {SHARED_GATE_HIDDEN_SIZES}"
        )
    # The op registry owns the EP, CP and PP degrees; the experts must still split evenly.
    if config.num_moe_experts % config.expert_model_parallel_size:
        raise NotImplementedError("[top] EP must divide the checkpoint expert count")
    required = {
        "params_dtype": torch.bfloat16,
        "moe_token_dispatcher_type": "flex",
        "moe_flex_dispatcher_backend": "deepep",
        "moe_router_score_function": "softmax",
        "moe_router_pre_softmax": False,
        "moe_router_enable_expert_bias": False,
        "moe_router_fusion": False,
        "moe_permute_fusion": False,
        "moe_apply_probs_on_input": False,
        "add_bias_linear": False,
        "gated_linear_unit": True,
    }
    unsupported = [
        "moe_expert_capacity_factor",
        "moe_latent_size",
        "moe_router_topk_scaling_factor",
        "moe_input_jitter_eps",
    ]
    if not match.gated_shared_expert:
        unsupported.append("moe_shared_expert_intermediate_size")
    else:
        if not getattr(
            config, "moe_shared_expert_intermediate_size", None
        ):  # config-access-exempt: TransformerConfig versions and model families differ in optional moe_shared_expert_intermediate_size
            raise NotImplementedError(f"[top] {match.name} requires its gated shared expert")
        required.update(moe_shared_expert_gate=True, moe_shared_expert_overlap=False)
        if match.shared_expert is not None:
            required.update(moe_shared_expert_intermediate_size=match.shared_expert)
    for field, value in required.items():
        if (
            getattr(config, field, value) != value
        ):  # config-access-exempt: TransformerConfig field selected at runtime from a table of checked names
            raise NotImplementedError(f"[top] Qwen MoE requires {field}={value!r}")
    for field in unsupported:
        if (
            getattr(config, field, None) is not None
        ):  # config-access-exempt: TransformerConfig field selected at runtime from a table of checked names
            raise NotImplementedError(f"[top] Qwen MoE does not support {field}")
    for field in ("moe_router_num_groups", "moe_router_group_topk"):
        if getattr(config, field, None) not in (
            None,
            1,
        ):  # config-access-exempt: TransformerConfig field selected at runtime from a table of checked names
            raise NotImplementedError(f"[top] Qwen MoE does not support grouped routing ({field})")
    # The TOP router routes exactly as serving does and adds no loss term, so these would be silently ignored.
    aux = getattr(
        config, "moe_aux_loss_coeff", 0.0
    )  # config-access-exempt: TransformerConfig versions and model families differ in optional moe_aux_loss_coeff
    if any(aux if isinstance(aux, (list, tuple)) else (aux,)):
        raise NotImplementedError("[top] Qwen MoE has no router aux loss; set --moe-aux-loss-coeff 0")
    if getattr(
        config, "moe_z_loss_coeff", None
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional moe_z_loss_coeff
        raise NotImplementedError("[top] Qwen MoE has no router z-loss; drop --moe-z-loss-coeff")
    if getattr(
        config, "moe_router_force_load_balancing", False
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional moe_router_force_load_balancing
        raise NotImplementedError("[top] Qwen MoE routes as serving does; drop --moe-router-force-load-balancing")
    if (
        getattr(config, "moe_router_dtype", None) is not None
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional moe_router_dtype
        raise NotImplementedError(
            "[top] Qwen MoE routes at serving's precision (BF16 gate logits, FP32 softmax); drop --moe-router-dtype"
        )


# Dispatch each micro-batch's MoE in this many token chunks (1 = whole micro-batch). One hot EP rank can receive
# every slot of its group, tokens x EP x top-k rows (GLM repetition loops near the response cap do this), and the
# path holds several row-sized tensors; chunking bounds them. Every kernel on the path is row-wise batch-invariant,
# so a row's bytes do not depend on its chunk. A fixed count keeps the DeepEP collectives matched across the group.
_TOKEN_CHUNKS = int(os.environ.get("TOP_MOE_CHUNKS", "1"))


class TopEPMoELayer(MoELayer):
    def forward(
        self, hidden_states, intermediate_tensors=None, padding_mask=None, input_ids=None, packed_seq_params=None
    ):
        if _TOKEN_CHUNKS <= 1 or intermediate_tensors is not None:
            return super().forward(hidden_states, intermediate_tensors, padding_mask, input_ids, packed_seq_params)
        n = hidden_states.shape[0]
        bounds = [n * i // _TOKEN_CHUNKS for i in range(_TOKEN_CHUNKS + 1)]
        outputs, bias = [], None
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
            out, bias = super().forward(
                hidden_states[lo:hi],
                None,
                None if padding_mask is None else padding_mask[:, lo:hi],
                # padding_mask and input_ids are [batch, sequence] beside [sequence, batch, hidden] states
                # (DS4 hash routing reads input_ids).
                None if input_ids is None else input_ids[:, lo:hi],
                packed_seq_params,
            )
            outputs.append(out)
        return torch.cat(outputs, 0), bias

    def __init__(self, config, *args, **kwargs):
        # The spec entry point validates model-wide configuration once.
        super().__init__(config, *args, **kwargs)
        if _TOKEN_CHUNKS > 1 and self.router.routing_replay is not None:
            raise NotImplementedError(
                "[top] TOP_MOE_CHUNKS > 1 routes each chunk separately, but rollout routing replay holds one "
                "micro-batch's routes per call; unset TOP_MOE_CHUNKS or drop --use-rollout-routing-replay"
            )
        if type(self.token_dispatcher) is not MoEFlexTokenDispatcher:
            raise NotImplementedError("[top] Qwen MoE requires the stock flex dispatcher")
        # Reuse the constructed dispatcher's groups and buffer state.
        self.token_dispatcher.__class__ = TopDeepEPTokenDispatcher
        self.token_dispatcher._top_install(self.router)
        # Capture point of the per-layer VJP (top_vjp): the routing weights the combine applies, and their gradient.
        self.top_routing_weights = CaptureTap()

    def postprocess(self, output, shared_expert_output):
        if (shared_expert_output is None) != (self.config.moe_shared_expert_intermediate_size is None):
            raise RuntimeError("[top] shared expert output does not match the configured shared expert")
        dispatcher = self.token_dispatcher
        rows = dispatcher.combine_postprocess(output)
        receipt = dispatcher.top_receipt
        dispatcher.top_receipt = None
        combined = combine_with_receipt(
            rows,
            receipt,
            scale=self.config.moe_router_topk_scaling_factor,
            router_topk_weights=self.top_routing_weights(self.router.last_topk_weights.float()),
        )
        combined = combined.view(dispatcher.hidden_shape)
        return combined if shared_expert_output is None else combined + shared_expert_output


def bind_moe_layer(spec, stamp, *, layer_cls=None, router_cls=None):
    """Bind a stock MoELayer spec to the EP layer (``layer_cls``, default TopEPMoELayer), TOP router (Qwen's unless
    ``router_cls``) and grouped experts."""
    from miles_plugins.top.grouped_experts import build_top_grouped_mlp
    from miles_plugins.top.router import TopTopKRouter

    router_cls = TopTopKRouter if router_cls is None else router_cls
    mlp = spec.submodules.mlp
    if not isinstance(mlp, functools.partial) or mlp.func is not MoELayer:
        raise NotImplementedError("[top] expected the stock MoELayer builder")
    moe = mlp.keywords["submodules"]
    moe.router = router_cls
    moe.experts = functools.partial(
        build_top_grouped_mlp(), **getattr(moe.experts, "keywords", {})
    )  # config-access-exempt: the experts builder is a functools.partial only on some layer specs
    layer_cls = TopEPMoELayer if layer_cls is None else layer_cls
    spec.submodules.mlp = functools.partial(layer_cls, **mlp.keywords)
    # MoE keeps pre-MLP norm standalone; the dense FC1 checkpoint alias does not apply.
    spec.submodules.sharded_state_dict_keys_map.pop("pre_mlp_layernorm.", None)
    stamp.append(("moe", "stock operations", f"{router_cls.__name__} / shared expert / DeepGEMM / slot combine"))


class TopGatedSharedExpert(SharedExpertMLP):
    def forward(self, hidden_states):
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        gate_up = delegated_column_linear(x, self.linear_fc1.weight)
        # Serving's shared expert is a dense MLP on the default (fast) SiLU build.
        y = delegated_row_linear(SiluAndMulFunction.apply(gate_up, False), self.linear_fc2.weight)
        # Qwen's separate shared gate rounds sigmoid to BF16 before multiplying.
        gate = delegated_column_linear(x, self.gate_weight).sigmoid()
        return (y * gate).reshape(shape)
