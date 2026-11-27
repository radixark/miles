"""References of TOP's per-layer VJP verdict (top_vjp): one adapter per family, keyed by its op-registry family name.

Each adapter loads the family's reference (HF transformers, or the model vendor's own code where transformers lacks
the model or models it poorly: DeepSeek-V4 runs DeepSeek's inference Block, top_reference_dsv4; "HF" names
either; DeepSeek-V4.1 its own, top_reference_dsv41), runs one decoder layer or one module inside it, and replays TOP's discrete selections (routing, DSA index)
and its deployed KV rounding. torch_gdn_path keeps a reference's GDN on HF's PyTorch implementation;
VisibleWithSurrogate bridges TOP's visible module output to the reference module's surrogate gradient.
"""

from __future__ import annotations

import contextlib
import inspect
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import NamedTuple

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from miles.backends.megatron_utils.megatron_to_hf.deepseekv3 import convert_deepseekv3_to_hf
from miles.backends.megatron_utils.megatron_to_hf.glm5_next import convert_glm5_next_to_hf
from miles.backends.megatron_utils.megatron_to_hf.qwen2 import convert_qwen2_to_hf
from miles.utils.debug_utils import top_reference_dsv4, top_reference_dsv41, top_reference_glm53, top_reference_qwen38
from miles.utils.debug_utils.top_contract import ContractModule, submodule
from miles.utils.debug_utils.vendor_reference import read_config_json
from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.top.kernels.surrogate import fp8_group_fake_quant


def _load_causal_lm(model_path: str, dtype: torch.dtype) -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, attn_implementation="eager")


def _named_param_grads(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.grad for name, param in layer.named_parameters() if param.grad is not None}


class Carried(NamedTuple):
    """What one packed sequence of a layer reads and writes outside its hidden streams (DeepSeek-V4.1's mHC pre, its
    engram rows, a KV source's compressed latent), from TOP's capture."""

    # {name: TOP's value} the reference differentiates, and {name: TOP's gradient of it}.
    inputs: dict[str, torch.Tensor]
    input_grads: dict[str, torch.Tensor]
    # {name: TOP's cotangent} of each extra output run_layer returns under that name.
    cotangents: dict[str, torch.Tensor]
    # {name: TOP's value} the reference reads but does not differentiate.
    replayed: dict[str, torch.Tensor]


class FamilyAdapter(NamedTuple):
    # (HF model) -> its decoder layers, indexed like TOP's.
    layers: Callable[[torch.nn.Module], Sequence[torch.nn.Module]]
    # (HF model, layer, x [1, n, H], positions [1, n], additive causal mask [1, 1, n, n]) -> layer output [1, n, H].
    run_layer: Callable[..., torch.Tensor]
    # ({TOP parameter name: grad}, text config) -> {reference parameter name: grad}.
    to_hf_param_grads: Callable[[dict[str, torch.Tensor], object], dict[str, torch.Tensor]]
    # (checkpoint path, dtype) -> the reference model, on CPU: an HF model, or any module whose layers run_layer can
    # call (a vendor's own reference code, vendor_reference).
    load: Callable[[str, torch.dtype], torch.nn.Module] = _load_causal_lm
    # (checkpoint path) -> its config; read_config_json for a model type transformers does not know.
    config: Callable[[str], object] = load_hf_config
    # Keep the model on CPU and move each replayed layer to the GPU (a model too large for one GPU); run_layer must
    # then read nothing else from the model on the GPU.
    layer_by_layer: bool = False
    # (HF layer) -> {HF parameter name inside the layer: grad}; a module replay bypasses (GLM's indexer) has none.
    # DeepSeek-V4 names each mHC scale's three scalars apart (attn_hc.alpha_pre, ...).
    layer_grads: Callable[[torch.nn.Module], dict[str, torch.Tensor]] = _named_param_grads
    # Layers carry mHC streams [T, 1, n*H] (HF: [1, n, n_streams, H]) instead of a packed [branch, residual] pair.
    mhc_streams: bool = False
    # (HF layer, routing [n, E] | None, index selection | None, first packed row) -> a context manager under which the
    # layer runs at TOP's selections for that sequence; None: replayed_selections (routing and DSA index replay).
    replay: Callable[..., contextlib.AbstractContextManager] | None = None
    # (HF layer) -> a context manager under which the layer rounds its KV/latent cache as serving's FP8 cache stores
    # it, straight-through; None: the family has no FP8 KV reference (an FP8 kv_cache_dtype is refused).
    kv_quantization: Callable[[torch.nn.Module], contextlib.AbstractContextManager] | None = None
    # (HF layer) -> a context manager yielding {HF parameter name inside the layer: sum_t |g_t|}, filled on exit, for
    # the parameters whose gradients are token sums that can cancel; None records none.
    token_terms: Callable[[torch.nn.Module], contextlib.AbstractContextManager] | None = None
    # (HF layer) -> {TOP module (top_capture.CONTRACT_MODULES or MHC_MODULES): its declared surrogate}, which the
    # contract check differentiates at TOP's captured module inputs.
    contract: Callable[[torch.nn.Module], dict[str, ContractModule]] | None = None
    # (HF parameter name) -> TOP's name of the same tensor ("decoder.layers.N...."), for the near-cancelling parameters
    # token_terms records whose TOP tensor is the HF one; the layout check judges those against sum_t |g_t|.
    top_name: Callable[[str], str | None] | None = None
    # (reference model, microbatch capture, layer, first row, end row) -> Carried for that packed sequence; run_layer
    # then takes ``carried=`` (inputs as leaves on the replay's device) and returns (output, {name: extra output}).
    # None: the layers carry nothing outside their streams.
    carried: Callable[..., Carried] | None = None


def _qwen3_run_layer(model, layer, x, positions, causal):
    return layer(
        x, attention_mask=causal, position_ids=positions, position_embeddings=model.model.rotary_emb(x, positions)
    )


def _output(out) -> torch.Tensor:
    """An HF module's output tensor: attention returns (output, weights)."""
    return out[0] if isinstance(out, tuple) else out


def _run_module(model, module, x, positions, causal):
    return _output(module(x))


def _qwen3_attention(model, module, x, positions, causal):
    return _output(
        module(
            x, attention_mask=causal, position_embeddings=model.model.rotary_emb(x, positions), position_ids=positions
        )
    )


def _qwen3_contract(layer):
    return {"self_attention": submodule("self_attn", _qwen3_attention), "mlp": submodule("mlp", _run_module)}


_QWEN3_TOKEN_NORMS = ("q_norm", "k_norm")


@contextlib.contextmanager
def _qwen3_token_terms(layer):
    """Yields {parameter name: sum over tokens of |g_t|} for the query and key norm weights, filled when the context
    exits (after the backward run inside it); |g_t| is the norm of token t's share of the gradient, over its heads."""
    leaves = {f"self_attn.{norm}.weight": [] for norm in _QWEN3_TOKEN_NORMS}
    handles = [
        getattr(layer.self_attn, norm).register_forward_hook(
            _per_token_gain(leaves[f"self_attn.{norm}.weight"])
        )  # config-access-exempt: attribute selected at runtime from a table of names
        for norm in _QWEN3_TOKEN_NORMS
    ]
    sums: dict[str, float] = {}
    try:
        yield sums
    finally:
        for handle in handles:
            handle.remove()
    for name, recorded in leaves.items():
        grads = [leaf.grad for leaf in recorded if leaf.grad is not None]
        if grads:
            sums[name] = sum(float(g.reshape(-1, g.shape[-1]).double().norm(dim=-1).sum()) for g in grads)


def _per_token_gain(leaves: list):
    """A forward hook of a norm whose output [..., tokens, heads, dim] is its weight times the normalized input: adds
    a leaf per token at the weight's own value (appended to ``leaves``), whose gradient is that token's share of the
    weight's. The output's value and the weight's own gradient stay as they are."""
    nested = []

    def hook(module, args, out):
        if nested:  # the unit-weight call below
            return None
        weight = module.weight.detach()
        nested.append(True)
        try:
            with torch.no_grad():
                normalized = torch.func.functional_call(module, {"weight": torch.ones_like(weight)}, args)
        finally:
            nested.pop()
        leaf = weight.expand(*out.shape[:-2], 1, -1).clone().requires_grad_()
        leaves.append(leaf)
        return out + normalized * (leaf - weight)

    return hook


def _qwen3_to_hf_param_grads(param_grads, config):
    layout = SimpleNamespace(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_query_groups=config.num_key_value_heads,
        kv_channels=config.head_dim,
    )
    out = {}
    for name, grad in param_grads.items():
        local = re.search(r"decoder\.layers\.\d+\..+$", name).group(0)
        out.update(convert_qwen2_to_hf(layout, f"module.module.{local}", grad))
    return out


_HF_EXPERT_RE = re.compile(r"(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def _converted_to_hf_param_grads(param_grads, convert, layout, num_experts):
    """Checkpoint names from a weight-sync converter, with the experts stacked as HF's fused expert parameters."""
    out, experts = {}, defaultdict(dict)
    for name, grad in param_grads.items():
        local = re.search(r"decoder\.layers\.\d+\..+$", name).group(0)
        for hf_name, hf_grad in convert(layout, f"module.module.{local}", grad):
            if (match := _HF_EXPERT_RE.fullmatch(hf_name)) is None:
                out[hf_name] = hf_grad
            else:
                prefix, expert, proj = match.groups()
                experts[prefix][(int(expert), proj)] = hf_grad
    for prefix, grads in experts.items():
        ids = range(num_experts)
        missing = [e for e in ids if (e, "down_proj") not in grads]
        if missing:
            raise ValueError(f"[top] {prefix}: no gradient for experts {missing}")
        out[f"{prefix}.gate_up_proj"] = torch.stack(
            [torch.cat((grads[e, "gate_proj"], grads[e, "up_proj"])) for e in ids]
        )
        out[f"{prefix}.down_proj"] = torch.stack([grads[e, "down_proj"] for e in ids])
    return out


class _PerExpertGLMExperts(torch.nn.Module):
    """HF's eager GLM experts with one parameter per expert (views of the stacked weights), the same arithmetic.

    Indexing a stacked parameter per expert back-propagates a full-size gradient buffer for every expert hit (24 GiB
    for GLM-5's FP32 gate/up projections), which no GPU holds; per-expert parameters each accumulate their own.
    """

    def __init__(self, experts):
        super().__init__()
        self.num_experts, self.act_fn = experts.num_experts, experts.act_fn
        self.gate_up_proj = torch.nn.ParameterList(experts.gate_up_proj.data.unbind(0))
        self.down_proj = torch.nn.ParameterList(experts.down_proj.data.unbind(0))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0])
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            gate_up = torch.nn.functional.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])
            current = torch.nn.functional.linear(self._gated(gate_up), self.down_proj[expert_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))
        return final_hidden_states

    def _gated(self, gate_up):
        gate, up = gate_up.chunk(2, -1)
        return self.act_fn(gate) * up


def _load_glm(model_path: str, dtype: torch.dtype, experts_class=_PerExpertGLMExperts) -> torch.nn.Module:
    config = load_hf_config(model_path)
    # A block-FP8 checkpoint's reference is the BF16 copy tools/convert_fp8_to_bf16_reference.py writes,
    # which keeps the source's quantization_config; its weights are plain BF16.
    if hasattr(
        config, "quantization_config"
    ):  # config-access-exempt: only quantized HF checkpoints carry quantization_config
        del config.quantization_config
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, dtype=dtype, attn_implementation="eager", experts_implementation="eager"
    )
    for layer in model.model.layers:
        if hasattr(
            layer.mlp, "experts"
        ):  # config-access-exempt: HF reference modules differ by model family in experts
            layer.mlp.experts = experts_class(layer.mlp.experts)
    return model


def _glm_layer_grads(layer):
    """Per-expert gradients stacked back under HF's names (``mlp.experts.gate_up_proj``, ``mlp.experts.down_proj``)."""
    out, experts = {}, defaultdict(dict)
    for name, grad in _named_param_grads(layer).items():
        if (match := re.fullmatch(r"(mlp\.experts\.(?:gate_up_proj|down_proj))\.(\d+)", name)) is None:
            out[name] = grad
        else:
            experts[match.group(1)][int(match.group(2))] = grad
    for name, grads in experts.items():
        params = getattr(
            layer.mlp.experts, name.rsplit(".", 1)[-1]
        )  # config-access-exempt: attribute selected at runtime from a table of names
        # Stacked on CPU: the stacked FP32 gradients would need another 36 GiB next to the per-expert ones.
        out[name] = torch.stack(
            [
                grads[e].cpu() if e in grads else torch.zeros(param.shape, dtype=param.dtype)
                for e, param in enumerate(params)
            ]
        )
    return out


def _glm_run_layer(model, layer, x, positions, causal):
    # The rotary module moves its frequencies to x's device, so the model itself may stay on CPU.
    return layer(
        x, attention_mask=causal, position_ids=positions, position_embeddings=model.model.rotary_emb(x, positions)
    )[0]


def _glm_contract(layer):
    return {"self_attention": submodule("self_attn", _qwen3_attention), "mlp": submodule("mlp", _run_module)}


def _glm_to_hf_param_grads(param_grads, config):
    """The weight-sync converter's names, experts stacked; the indexer is dropped: TOP freezes it (its selection is
    discrete), and index replay bypasses HF's."""
    layout = SimpleNamespace(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_query_groups=config.num_key_value_heads,
        kv_channels=None,
        indexer_rope_interleave=False,
        num_layers=config.num_hidden_layers,
    )
    out = _converted_to_hf_param_grads(param_grads, convert_deepseekv3_to_hf, layout, config.n_routed_experts)
    return {name: grad for name, grad in out.items() if ".self_attn.indexer." not in name}


def _glm53_to_hf_param_grads(param_grads, config):
    """The weight-sync converter's names, experts stacked, as the reference layer's (top_reference_glm53)."""
    layout = SimpleNamespace(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_query_groups=config.num_key_value_heads,
        kv_channels=None,
        indexer_rope_interleave=config.indexer_rope_interleave,
        num_layers=config.num_hidden_layers,
    )
    converted = _converted_to_hf_param_grads(param_grads, convert_glm5_next_to_hf, layout, config.n_routed_experts)
    return top_reference_glm53.to_reference_names(converted)


@contextlib.contextmanager
def _glm53_replayed_selections(layer, routing: torch.Tensor | None, indices: torch.Tensor | None, lo: int):
    """TOP's expert routing and, on a DSA layer, its index selection for rows [lo, hi) of the packed stream."""
    top_reference_glm53.require_selection(layer, indices)
    keys = None if indices is None else top_reference_glm53.key_indices(indices, lo)
    with _replayed_routing(layer, routing), _replayed_indices(layer, keys):
        yield


@contextlib.contextmanager
def _glm_kv_quantization(layer):
    """The MLA latent as serving's DSA FP8 KV cache stores it: the 512 normalized nope dims (kv_a_layernorm's output)
    in E4M3 with an FP32 amax scale per 128 (quant_k_cache), straight-through; the RoPE dims stay unrounded."""
    handle = layer.self_attn.kv_a_layernorm.register_forward_hook(
        lambda module, args, out: fp8_group_fake_quant(out, 128, scale="amax")
    )
    try:
        yield
    finally:
        handle.remove()


def _load_qwen_gdn(model_path: str, dtype: torch.dtype) -> torch.nn.Module:
    """Qwen3.5 checkpoints are image-text models; the text decoder is ``model.model.language_model``."""
    return AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=dtype, attn_implementation="eager", experts_implementation="eager"
    )


def _qwen_gdn_run_layer(model, layer, x, positions, causal):
    text = model.model.language_model
    # Text-only rows: the three M-RoPE axes carry the same position.
    position_embeddings = text.rotary_emb(x, positions[None].expand(3, -1, -1))
    linear = hasattr(
        layer, "linear_attn"
    )  # config-access-exempt: HF reference modules differ by model family in linear_attn
    return layer(
        x,
        position_embeddings=position_embeddings,
        attention_mask=None if linear else causal,
        position_ids=positions,
    )


def _qwen_gdn_attention(model, module, x, positions, causal):
    rotary = model.model.language_model.rotary_emb(x, positions[None].expand(3, -1, -1))
    return _output(module(x, attention_mask=causal, position_embeddings=rotary, position_ids=positions))


def _qwen_gdn_linear_attention(model, module, x, positions, causal):
    return _output(module(x, attention_mask=None))


def _qwen_gdn_contract(layer):
    """TOP's ``self_attention`` is HF's GDN (``linear_attn``) on a linear-attention layer, else HF's ``self_attn``."""
    attention = (
        submodule("linear_attn", _qwen_gdn_linear_attention)
        if hasattr(
            layer, "linear_attn"
        )  # config-access-exempt: HF reference modules differ by model family in linear_attn
        else submodule("self_attn", _qwen_gdn_attention)
    )
    return {"self_attention": attention, "mlp": submodule("mlp", _run_module)}


# TOP's names inside a Qwen GDN decoder layer -> HF's: GDN and full attention keep HF names under self_attention.
_QWEN_GDN_NAMES = {
    "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
    "mlp.router.weight": "mlp.gate.weight",
    "mlp.shared_experts.linear_fc2.weight": "mlp.shared_expert.down_proj.weight",
    "mlp.shared_experts.gate_weight": "mlp.shared_expert_gate.weight",
    "mlp.linear_fc2.weight": "mlp.down_proj.weight",
}
_TOP_EXPERT_RE = re.compile(r"mlp\.experts\.(linear_fc[12])\.weight(\d+)$")


def _qwen_gdn_to_hf_param_grads(param_grads, config):
    """HF Qwen3.5 layer names; SwiGLU fc1 splits into gate and up, and the experts stack as HF's fused experts."""
    out, experts = {}, defaultdict(dict)
    for name, grad in param_grads.items():
        number, local = re.search(r"decoder\.layers\.(\d+)\.(.+)$", name).groups()
        prefix = f"model.layers.{number}"
        if (match := _TOP_EXPERT_RE.fullmatch(local)) is not None:
            experts[prefix][(int(match.group(2)), match.group(1))] = grad
        elif local.startswith("self_attention."):
            out[f"{prefix}.{local.removeprefix('self_attention.')}"] = grad
        elif local in ("mlp.linear_fc1.weight", "mlp.shared_experts.linear_fc1.weight"):
            mlp = "mlp" if local == "mlp.linear_fc1.weight" else "mlp.shared_expert"
            gate, up = grad.chunk(2, dim=0)
            out[f"{prefix}.{mlp}.gate_proj.weight"], out[f"{prefix}.{mlp}.up_proj.weight"] = gate, up
        else:
            out[f"{prefix}.{_QWEN_GDN_NAMES.get(local, local)}"] = grad
    _stack_top_experts(out, experts, config.num_experts)
    return out


def _stack_top_experts(out, experts, num_experts):
    """TOP's per-expert fc1/fc2 ({layer prefix: {(global id, linear_fc1|2): grad}}) as HF's fused expert parameters."""
    for prefix, grads in experts.items():
        ids = range(num_experts)
        missing = [e for e in ids if (e, "linear_fc2") not in grads]
        if missing:
            raise ValueError(f"[top] {prefix}: no gradient for experts {missing}")
        out[f"{prefix}.mlp.experts.gate_up_proj"] = torch.stack([grads[e, "linear_fc1"] for e in ids])
        out[f"{prefix}.mlp.experts.down_proj"] = torch.stack([grads[e, "linear_fc2"] for e in ids])


def _qwen38_to_hf_param_grads(param_grads, config):
    out, experts = top_reference_qwen38.converted_param_grads(param_grads, config)
    _stack_top_experts(out, experts, config.num_experts)
    return out


@contextlib.contextmanager
def _qwen38_replayed_selections(layer, routing: torch.Tensor | None, indices: torch.Tensor | None, lo: int):
    """TOP's expert routing and QSA index selection for rows [lo, hi) of the packed stream."""
    with _replayed_routing(layer, routing), top_reference_qwen38.replayed_qsa(layer, indices, lo):
        yield


ADAPTERS: dict[str, FamilyAdapter] = {
    "Qwen3 dense": FamilyAdapter(
        layers=lambda model: model.model.layers,
        run_layer=_qwen3_run_layer,
        to_hf_param_grads=_qwen3_to_hf_param_grads,
        token_terms=_qwen3_token_terms,
        contract=_qwen3_contract,
    ),
    "Qwen GDN MoE": FamilyAdapter(
        layers=lambda model: model.model.language_model.layers,
        run_layer=_qwen_gdn_run_layer,
        to_hf_param_grads=_qwen_gdn_to_hf_param_grads,
        load=_load_qwen_gdn,
        contract=_qwen_gdn_contract,
    ),
    # transformers' qwen4_exp. The n-gram table stays on the host, so layers move to the GPU one at a time; the PLE
    # layer reads the document's token ids (carried) and the QSA layer runs at TOP's index selection.
    "Qwen3.8-Flash-Next": FamilyAdapter(
        layers=top_reference_qwen38.layers,
        run_layer=top_reference_qwen38.run_layer,
        to_hf_param_grads=_qwen38_to_hf_param_grads,
        load=top_reference_qwen38.load,
        layer_by_layer=True,
        mhc_streams=True,
        replay=_qwen38_replayed_selections,
        contract=top_reference_qwen38.contract,
        carried=top_reference_qwen38.carried,
    ),
    "GLM": FamilyAdapter(
        layers=lambda model: model.model.layers,
        run_layer=_glm_run_layer,
        to_hf_param_grads=_glm_to_hf_param_grads,
        load=_load_glm,
        layer_by_layer=True,
        layer_grads=_glm_layer_grads,
        kv_quantization=_glm_kv_quantization,
        contract=_glm_contract,
    ),
    # transformers' glm5_next: KDA on its PyTorch chunk path, DSA at TOP's index selection, mHC with its sublayer norm.
    "GLM-5.3-Flash": FamilyAdapter(
        layers=top_reference_glm53.layers,
        run_layer=top_reference_glm53.run_layer,
        to_hf_param_grads=_glm53_to_hf_param_grads,
        load=top_reference_glm53.load,
        layer_by_layer=True,
        layer_grads=_glm_layer_grads,
        mhc_streams=True,
        replay=_glm53_replayed_selections,
        token_terms=top_reference_glm53.token_terms,
        contract=top_reference_glm53.contract,
        top_name=top_reference_glm53.top_name,
    ),
    # DeepSeek's own inference code: HF transformers models DeepSeek poorly.
    "DeepSeek-V4": FamilyAdapter(
        layers=lambda model: model.layers,
        run_layer=top_reference_dsv4.run_layer,
        to_hf_param_grads=top_reference_dsv4.to_reference_param_grads,
        load=top_reference_dsv4.load,
        config=read_config_json,
        layer_by_layer=True,
        layer_grads=top_reference_dsv4.layer_grads,
        mhc_streams=True,
        replay=top_reference_dsv4.replayed_selections,
        kv_quantization=top_reference_dsv4.kv_quantization,
        token_terms=top_reference_dsv4.token_terms,
        contract=top_reference_dsv4.contract,
        top_name=top_reference_dsv4.top_name,
    ),
    # DeepSeek's own inference code, as for DeepSeek-V4; the layers carry mHC pre, engram rows and source latents.
    # No contract modules are declared yet: the verdict reports its layer replays and judges the capture checks only.
    "DeepSeek-V4.1": FamilyAdapter(
        layers=lambda model: model.layers,
        run_layer=top_reference_dsv41.run_layer,
        to_hf_param_grads=top_reference_dsv41.to_reference_param_grads,
        load=top_reference_dsv41.load,
        config=read_config_json,
        layer_by_layer=True,
        layer_grads=top_reference_dsv41.layer_grads,
        mhc_streams=True,
        replay=top_reference_dsv41.replayed_selections,
        kv_quantization=top_reference_dsv41.kv_quantization,
        token_terms=top_reference_dsv41.token_terms,
        carried=top_reference_dsv41.carried,
    ),
}


@contextlib.contextmanager
def _replayed_routing(layer, routing: torch.Tensor | None):
    """HF's router at TOP's selection: ``routing`` [n, E] bool; the weights are HF's own softmax at those experts."""
    gate = getattr(
        getattr(layer, "mlp", None), "gate", None
    )  # config-access-exempt: HF reference modules differ by model family in gate
    if (
        routing is None or gate is None or not hasattr(gate, "top_k")
    ):  # config-access-exempt: HF reference modules differ by model family in top_k
        yield
        return
    ids = routing.float().topk(gate.top_k, dim=-1).indices.sort(dim=-1).values

    def biased_forward(hidden_states):
        """GLM: FP32 sigmoid scores at TOP's experts, renormalized, times the routed scaling (HF's own arithmetic)."""
        logits = torch.nn.functional.linear(hidden_states.reshape(-1, gate.hidden_dim).float(), gate.weight.float())
        chosen = ids.to(logits.device)
        top = logits.sigmoid().gather(1, chosen)
        if gate.norm_topk_prob:
            top = top / (top.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, top * gate.routed_scaling_factor, chosen

    def forward(hidden_states):
        hidden_states = hidden_states.reshape(-1, gate.hidden_dim)
        logits = torch.nn.functional.linear(hidden_states, gate.weight)
        probs = torch.softmax(logits, dtype=torch.float, dim=-1)
        chosen = ids.to(logits.device)
        top = probs.gather(1, chosen)
        if getattr(
            gate, "norm_topk_prob", True
        ):  # config-access-exempt: HF reference modules differ by model family in norm_topk_prob
            top = top / top.sum(dim=-1, keepdim=True)
        return logits, top.to(logits.dtype), chosen

    gate.forward = (
        biased_forward if hasattr(gate, "e_score_correction_bias") else forward
    )  # config-access-exempt: HF reference modules differ by model family in e_score_correction_bias
    try:
        yield
    finally:
        del gate.forward


def _hf_key_indices(indices: torch.Tensor, lo: int) -> torch.Tensor:
    """TOP's selection for rows [lo, hi) of the packed stream ([n, 1, topk], packed key rows, -1 unselected) as HF's
    [1, n, topk] key positions within the sequence. A -1 pad appears only in a row with fewer causal keys than the top-k,
    which then selects all of them, itself included, so the row's own position stands in for the pad."""
    keys = indices.reshape(indices.shape[0], -1).long()
    own = torch.arange(keys.shape[0])[:, None].expand_as(keys)
    keys = torch.where(keys >= 0, keys - lo, own)
    if bool(((keys < 0) | (keys > own)).any()):
        raise ValueError("[top] a captured DSA selection reaches outside its row's causal keys")
    return keys[None]


@contextlib.contextmanager
def replayed_selections(layer, routing: torch.Tensor | None, indices: torch.Tensor | None, lo: int):
    """Routing and DSA index replay for rows [lo, hi) of the packed stream (every family but DeepSeek-V4)."""
    with (
        _replayed_routing(layer, routing),
        _replayed_indices(layer, None if indices is None else _hf_key_indices(indices, lo)),
    ):
        yield


@contextlib.contextmanager
def _replayed_indices(layer, indices: torch.Tensor | None):
    """HF's DSA attention over TOP's selected keys: the layer's indexer is bypassed and the selection is passed the
    way HF passes a shared layer the previous full layer's."""
    attn = getattr(
        layer, "self_attn", None
    )  # config-access-exempt: HF reference modules differ by model family in self_attn
    if (
        indices is None or attn is None or not hasattr(attn, "indexer")
    ):  # config-access-exempt: HF reference modules differ by model family in indexer
        yield
        return
    indexer, original = attn.indexer, attn.forward
    indices = indices.to(next(attn.parameters()).device)

    def forward(*args, prev_topk_indices=None, **kwargs):
        return original(*args, prev_topk_indices=indices, **kwargs)

    attn.indexer, attn.forward = None, forward
    try:
        yield
    finally:
        attn.indexer = indexer
        del attn.forward


class VisibleWithSurrogate(torch.autograd.Function):
    """The value TOP's forward produced, with the gradient of the reference's surrogate of that module: the contract
    check judges whether TOP's backward is the declared one, apart from how good that surrogate is."""

    @staticmethod
    def forward(ctx, visible, surrogate):
        if visible.shape != surrogate.shape or visible.dtype != surrogate.dtype or visible.device != surrogate.device:
            raise ValueError(
                f"[top] visible {tuple(visible.shape)} {visible.dtype} and surrogate {tuple(surrogate.shape)} "
                f"{surrogate.dtype} differ"
            )
        return visible.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output


# HF's GDN and KDA kernel entry points (Qwen GDN, GLM-5.3-Flash); each falls back to its PyTorch implementation when
# FLA / causal-conv1d is absent.
_GDN_KERNELS = (
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "chunk_kimi_delta_attention",
    "recurrent_kimi_delta_attention",
    "causal_conv1d_fn",
    "causal_conv1d_update",
)


@contextlib.contextmanager
def torch_gdn_path(layer: torch.nn.Module):
    """HF's GDN runs its PyTorch implementations instead of FLA's Triton and causal-conv1d's CUDA kernels, which it
    calls when installed: an FP32 Triton dot may round through TF32."""
    patched = []
    # A vendor reference's module is not in sys.modules (vendor_reference loads it by path); it has no GDN.
    modules = {sys.modules.get(type(child).__module__) for child in layer.modules()} - {None}
    for module in modules:
        for name in _GDN_KERNELS:
            if (
                kernel := getattr(module, name, None)
            ) is not None:  # config-access-exempt: attribute selected at runtime from a table of names
                patched.append((module, name, kernel))
                setattr(module, name, inspect.unwrap(kernel))
    try:
        yield
    finally:
        for module, name, kernel in patched:
            setattr(module, name, kernel)
