"""Trainer half of the TOP op registry: each op's trainer cells and each family's block builder.

SGLang's ``sglang.srt.true_on_policy.registry`` owns the op and family list; this module keys its
tables by those names, so a layout is admitted iff every op of the family supports it on both sides.
"""

from __future__ import annotations

from collections.abc import Sequence

import msgspec
from sglang.srt.true_on_policy.registry import (
    EVERY,
    NOT_QUALIFIED,
    OPS_BY_NAME,
    Cell,
    Family,
    admit,
    at_least,
    only,
    refused,
)


class TrainerEnvelope(msgspec.Struct, frozen=True, kw_only=True):
    ep: Cell
    # Degree of all-gather CP; the CP form itself is a core rule.
    cp: Cell
    pp: Cell


class TrainerLayout(msgspec.Struct, frozen=True, kw_only=True):
    tp: int
    etp: int
    ep: int
    cp: int
    pp: int
    virtual_pipeline: bool
    sequence_parallel: bool
    allgather_cp: bool

    @classmethod
    def from_config(cls, config, *, allgather_cp: bool) -> TrainerLayout:
        """The layout Megatron builds; ``allgather_cp`` is a Miles argument the config does not carry."""
        tp = config.tensor_model_parallel_size
        return cls(
            tp=tp,
            etp=config.expert_tensor_parallel_size or tp,
            ep=config.expert_model_parallel_size,
            cp=config.context_parallel_size,
            pp=config.pipeline_model_parallel_size,
            virtual_pipeline=config.virtual_pipeline_model_parallel_size is not None,
            sequence_parallel=config.sequence_parallel,
            allgather_cp=allgather_cp,
        )

    @classmethod
    def from_args(cls, args) -> TrainerLayout:
        """The same layout from arguments that Megatron's validate_args has resolved, before any process starts."""
        return cls(
            tp=args.tensor_model_parallel_size,
            etp=args.expert_tensor_parallel_size,
            ep=args.expert_model_parallel_size,
            cp=args.context_parallel_size,
            pp=args.pipeline_model_parallel_size,
            virtual_pipeline=args.virtual_pipeline_model_parallel_size is not None,
            sequence_parallel=args.sequence_parallel,
            allgather_cp=args.allgather_cp,
        )


def _trainer(**cells) -> tuple[TrainerEnvelope, ...]:
    return (TrainerEnvelope(**{"ep": EVERY, "cp": EVERY, "pp": EVERY, **cells}),)


# Keyed by SGLang's Op.name.
TRAINER_CELLS: dict[str, tuple[TrainerEnvelope, ...]] = {
    "bi_linear": _trainer(),
    "rms_norm": _trainer(),
    "packed_residual": _trainer(),
    "rope": _trainer(),
    "swiglu": _trainer(),
    "fa4_attention": _trainer(cp=refused("the Qwen3 attention has no all-gather CP path")),
    "lm_head": _trainer(),
    "qwen_router": _trainer(),
    "ep_slot_transport": _trainer(),
    # Refuses every layout: no e2e test runs Qwen3 MoE; the op stays as code for a family that adds a test.
    "qwen3_moe_experts": _trainer(ep=only(why="no e2e gate runs Qwen3 MoE"), pp=refused(NOT_QUALIFIED)),
    # Each stage runs its layers' chunk prefill per sample; the last stage defers the head (the dense image gate's
    # trainer PP2 phase).
    "gdn": _trainer(),
    "gdn_full_attention": _trainer(),
    # The KDA core prefills whole samples on one rank: no state hand-off across context-parallel ranks or stages yet.
    "kda": _trainer(
        cp=refused("the KDA core prefills whole samples on one rank (no CP state hand-off)"),
        pp=refused(NOT_QUALIFIED),
    ),
    # GLM-5.3-Flash: the stream mean, the clamped BF16 experts and the clamped dense/shared MLP.
    "hc_mean_head": _trainer(pp=refused(NOT_QUALIFIED)),
    "glm_clamped_experts": _trainer(
        ep=at_least(2, why="the experts run behind the DeepEP dispatcher, which needs EP>=2"),
        pp=refused(NOT_QUALIFIED),
    ),
    "clamped_swiglu_bf16": _trainer(),
    "decode_kv_split": _trainer(),
    "shared_expert_gate": _trainer(),
    # Across pipeline stages: the MoE image gate's PP2 x EP2 phase.
    "qwen2_moe_experts": _trainer(
        ep=at_least(2, why="the experts run behind the DeepEP dispatcher, which needs EP>=2"),
    ),
    "dsa": _trainer(),
    "glm_router": _trainer(),
    "glm_experts": _trainer(
        ep=at_least(2, why="the experts run behind the DeepEP dispatcher, which needs EP>=2"),
        pp=refused(NOT_QUALIFIED),
    ),
    "fp8_linear": _trainer(),
    "glm_fp8_experts": _trainer(ep=at_least(2, why="the experts run behind the DeepEP dispatcher, which needs EP>=2")),
    "clamped_swiglu": _trainer(),
    "dsv4_router": _trainer(),
    "dsv4_experts": _trainer(ep=at_least(2, why="the experts run behind the DeepEP dispatcher, which needs EP>=2")),
    "mhc": _trainer(),
    "dsv4_attention": _trainer(),
    "fp8_b32_linear": _trainer(),
    "clamped_swiglu_b32": _trainer(),
    "mhc_prev": _trainer(),
    "engram": _trainer(cp=refused("the V4.1 engram hashes whole packed requests on one rank (no CP gather)")),
    "dsv41_router": _trainer(),
    "dsv41_attention": _trainer(
        cp=refused("the V4.1 attention runtime prefills whole packed requests on one rank (no all-gather CP path)"),
        # Stage boundaries carry the cross-layer state (dsv41_pipeline), each kind as qualified there.
        pp=only(1, 2, 8, why=NOT_QUALIFIED),
    ),
    # Qwen3.8-Flash-Next scores whole samples on one rank: the gated residual's streams and the PLE's n-gram ids
    # (published where the embedding is) have no CP or stage hand-off, and QSA selects over a sample's own keys.
    "hyper_connection": _trainer(
        cp=refused("the gated residual has no context-parallel path"),
        pp=refused("no gate runs the widened residual streams across pipeline stages"),
    ),
    "ple": _trainer(
        cp=refused("the PLE convolution reads a whole sample's earlier rows on one rank"),
        pp=refused("the PLE layer hashes input ids, which only the first stage holds"),
    ),
    "qsa": _trainer(
        cp=refused("QSA selects and attends over a whole sample's keys on one rank (no all-gather CP path)"),
        pp=refused(NOT_QUALIFIED),
    ),
    # The frozen tower (miles_plugins.top.vision) runs whole on every rank that embeds (the first pipeline stage);
    # under all-gather CP each rank writes the rows of the placeholders it holds (gdn_forward).
    "qwen_vision_tower": _trainer(),
    # The frozen V4.1 tower (miles_plugins.top.dsv41_vision) encodes every image of a sample on the rank that embeds
    # it; V4.1 runs no CP (dsv41_attention) and only the first pipeline stage embeds.
    "dsv41_vision_tower": _trainer(),
    "glm5_next_dsa": _trainer(
        cp=refused("the pooled indexer compresses and selects over a whole sample's keys on one rank (no CP gather)"),
        pp=refused(NOT_QUALIFIED),
    ),
}


class TopBuild(msgspec.Struct, frozen=True, kw_only=True):
    # "module:function" building the block: (args, config, vp_stage, stamp, *, contract).
    build: str
    # "module:function" returning the serving program's ServerArgs keywords (attention backend, KV dtype, MoE backends).
    server_kwargs: str | None = None


_QWEN3 = TopBuild(build="miles_plugins.top.spec:qwen3_block")
_GDN = TopBuild(build="miles_plugins.top.gdn_spec:gdn_block")
_GLM = TopBuild(
    build="miles_plugins.top.glm_spec:glm_block", server_kwargs="miles_plugins.top.glm_spec:glm_server_kwargs"
)
_DSV4 = TopBuild(
    build="miles_plugins.top.dsv4_spec:dsv4_block", server_kwargs="miles_plugins.top.dsv4_spec:dsv4_server_kwargs"
)
_DSV41 = TopBuild(
    build="miles_plugins.top.dsv41_spec:dsv41_block", server_kwargs="miles_plugins.top.dsv41_spec:dsv41_server_kwargs"
)
_QWEN38 = TopBuild(build="miles_plugins.top.qwen38_spec:qwen38_block")
_GLM53 = TopBuild(
    build="miles_plugins.top.glm53_spec:glm53_block", server_kwargs="miles_plugins.top.glm53_spec:glm53_server_kwargs"
)
# Keyed by SGLang's Family.name.
BUILDERS: dict[str, TopBuild] = {
    "Qwen3 dense": _QWEN3,
    "Qwen3 MoE": _QWEN3,
    "Qwen GDN dense": _GDN,
    "Qwen GDN MoE": _GDN,
    "Qwen3.8-Flash-Next": _QWEN38,
    "GLM": _GLM,
    "DeepSeek-V4": _DSV4,
    "DeepSeek-V4.1": _DSV41,
    "GLM-5.3-Flash": _GLM53,
}

_SERVING_ONLY = (TrainerEnvelope(ep=EVERY, cp=EVERY, pp=EVERY),)


def refuse_core_trainer_layout(layout: TrainerLayout) -> None:
    """Refusals that hold for every family, so no op restates them."""
    if layout.tp > 1 or layout.etp > 1:
        raise NotImplementedError(
            "[top] trainer TP and expert TP must be 1: row linears run the full-K BI GEMM, with no TP-invariant reduce"
        )
    if layout.sequence_parallel:
        raise NotImplementedError("[top] sequence parallelism has no TOP program; drop --sequence-parallel")
    if layout.virtual_pipeline:
        raise NotImplementedError("[top] virtual pipeline stages have no TOP program")
    if layout.cp > 1 and not layout.allgather_cp:
        raise NotImplementedError(
            "[top] context parallelism runs in its all-gather form only (--allgather-cp); Ulysses and ring CP are refused"
        )


def _trainer_envelopes(name: str) -> Sequence[TrainerEnvelope]:
    if not OPS_BY_NAME[name].in_trainer:
        return _SERVING_ONLY
    try:
        return TRAINER_CELLS[name]
    except KeyError as exc:
        raise NotImplementedError(f"[top] op {name} declares no trainer cells") from exc


def admit_trainer(model_type: str, layout: TrainerLayout, *, fp8: bool, vision: bool = False) -> Family:
    """``vision``: the run trains samples with images or video, so the family's vision ops join its text ops."""
    refuse_core_trainer_layout(layout)
    return admit(model_type, layout, fp8=fp8, side="trainer", envelopes_of=_trainer_envelopes, vision=vision)
