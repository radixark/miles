"""GLM-5.3-Flash (HF ``glm5_next``) reference of TOP's per-layer VJP verdict (top_vjp): transformers' own model.

A decoder layer takes the 4 mHC streams apart ([1, n, 4, H]) and returns them so. Attention is NoPE: KDA layers run
the reference's PyTorch chunk KDA (torch_gdn_path), DSA layers its MLA at TOP's captured index selection (the indexer
is bypassed, as TOP freezes it). The routed experts clamp the SwiGLU inputs at the checkpoint's limit, one parameter
per expert. Each mHC site is the reference's hyper-connection followed by the sublayer norm TOP's mhc_pre fuses; under
token_terms its base and scale take one leaf per token, so their near-cancelling gradients are judged against
sum_t |g_t|.
"""

from __future__ import annotations

import contextlib
import functools
import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText

from miles.utils.debug_utils.top_capture import MHC_MODULES
from miles.utils.debug_utils.top_contract import ContractModule, submodule

# Each mHC site: the reference's hyper-connection, the sublayer norm TOP's mhc_pre fuses, TOP's module.
_SITES = {"attn_hc": "input_layernorm", "ffn_hc": "post_attention_layernorm"}
_TOP_SITE = dict(zip(_SITES, MHC_MODULES, strict=True))
# Checkpoint names (weight sync) -> the reference module's, inside a layer.
_RENAMES = {
    **{f"hc_{site}_{part}": f"{site}_hc.{part}" for site in ("attn", "ffn") for part in ("fn", "base", "scale")},
    **{f"self_attn.{name}": f"self_attn.forget_gate.{name}" for name in ("A_log", "dt_bias")},
    **{f"self_attn.{proj}.weight": f"self_attn.forget_gate.{proj}.weight" for proj in ("f_a_proj", "f_b_proj")},
}
_CONV = ("q_conv1d", "k_conv1d", "v_conv1d")
_HF_NAME_RE = re.compile(r"^(model\.layers\.\d+)\.(.+)$")
# The indexer's selection is discrete and TOP freezes it; the router's bias is a buffer.
_DROPPED = (".self_attn.indexer.", ".mlp.gate.e_score_correction_bias")
_TOKEN_PARAMS = ("base", "scale")
_HC_BASE_RE = re.compile(r"^model\.layers\.(\d+)\.(attn_hc|ffn_hc)\.base$")


def layers(model):
    return model.model.language_model.layers


class _PerExpertClampedExperts(torch.nn.Module):
    """The reference's eager experts with one parameter per expert (views of the stacked weights) and its clamped
    SwiGLU: indexing a stacked parameter per expert back-propagates a full-size gradient buffer for every expert hit.
    """

    def __init__(self, experts):
        super().__init__()
        self.num_experts, self.swiglu_limit = experts.num_experts, experts.swiglu_limit
        self.gate_up_proj = torch.nn.ParameterList(experts.gate_up_proj.data.unbind(0))
        self.down_proj = torch.nn.ParameterList(experts.down_proj.data.unbind(0))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert in hit:
            expert = int(expert[0])
            if expert == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert])
            gate, up = F.linear(hidden_states[token_idx], self.gate_up_proj[expert]).chunk(2, dim=-1)
            gated = F.silu(gate.clamp(max=self.swiglu_limit)) * up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            current = F.linear(gated, self.down_proj[expert]) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


def load(model_path: str, dtype: torch.dtype) -> torch.nn.Module:
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=dtype, attn_implementation="eager", experts_implementation="eager"
    )
    for layer in layers(model):
        if hasattr(layer.mlp, "experts"):  # config-access-exempt: the dense first layer's MLP has no experts
            layer.mlp.experts = _PerExpertClampedExperts(layer.mlp.experts)
    return model


def _linear(layer) -> bool:
    return layer.block_type == "linear_attention"


def run_layer(model, layer, x, positions, causal):
    """x [1, n, 4, H] -> [1, n, 4, H]. NoPE: no position embeddings; every row is a valid key."""
    mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    return layer(x, attention_mask=mask, position_ids=positions, position_embeddings=None)[0]


def _kda(model, module, x, positions, causal):
    return module(x, attention_mask=None)


def _dsa(model, module, x, positions, causal):
    # The indexer is bypassed (replayed_selections), so no padding mask is read.
    return module(x, attention_mask=None)[0]


def _mlp(model, module, x, positions, causal):
    return module(x)


def _run_hc(site: str, model, layer, x, positions, causal):
    """TOP's mHC module at the streams x [1, n, 4*H]: the sublayer input after the fused norm [1, n, H] (output.0),
    comb [1, n, 16] (output.1) and post [1, n, 4] (output.2)."""
    hc = layer.get_submodule(site)
    post, comb, collapsed = hc(x.unflatten(-1, (hc.hc_mult, -1)))
    return layer.get_submodule(_SITES[site])(collapsed), comb.flatten(-2), post


def _hc_contract(site: str) -> ContractModule:
    return ContractModule(
        params=(f"{site}.", f"{_SITES[site]}."),
        run=functools.partial(_run_hc, site),
        outputs=("output.0", "output.1", "output.2"),
        judges_input_grad=False,
    )


def contract(layer) -> dict[str, ContractModule]:
    """TOP's ``self_attention`` is the reference's KDA or DSA attention at its normed input, ``mlp`` the dense MLP or
    the MoE block with its shared expert, and each mHC module the reference's hyper-connection with the norm after it.
    """
    return {
        "self_attention": submodule("self_attn", _kda if _linear(layer) else _dsa),
        "mlp": submodule("mlp", _mlp),
        **{_TOP_SITE[site]: _hc_contract(site) for site in _SITES},
    }


def require_selection(layer, indices: torch.Tensor | None) -> None:
    """A DSA layer runs at TOP's selection only: the reference indexer's own would be another function."""
    if (indices is None) != _linear(layer):
        raise ValueError(
            "[top] a GLM-5.3-Flash DSA layer is replayed at TOP's captured index selection, and a KDA layer has none"
        )


def key_indices(indices: torch.Tensor, lo: int) -> torch.Tensor:
    """TOP's selection for rows [lo, hi) of the packed stream ([n, 1, W] packed key rows, -1 unselected) as the
    reference's [1, n, W] key positions within the sequence, -1 kept: a row that closes a pool the indexer did not pick
    attends to neither that pool nor itself, and the reference's mask reads -1 as no key."""
    keys = indices.reshape(indices.shape[0], -1).long()
    valid = keys >= 0
    local = keys - lo
    if bool((valid & ((local < 0) | (local > torch.arange(keys.shape[0])[:, None]))).any()):
        raise ValueError("[top] a captured GLM-5.3-Flash DSA selection reaches outside its row's causal keys")
    if not bool(valid.any(-1).all()):
        raise ValueError("[top] a captured GLM-5.3-Flash DSA selection leaves a row with no key")
    return torch.where(valid, local, -1)[None]


def to_reference_names(converted: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Weight sync's checkpoint names (experts already stacked) as the reference layer's: mHC and forget-gate names,
    the q/k/v convolutions as the one depthwise convolution; the indexer and router bias dropped."""
    out, conv = {}, {}
    for name, grad in converted.items():
        if any(dropped in name for dropped in _DROPPED):
            continue
        prefix, local = _HF_NAME_RE.fullmatch(name).groups()
        if (part := local.removeprefix("self_attn.").removesuffix(".weight")) in _CONV:
            conv.setdefault(prefix, {})[part] = grad
            continue
        out[f"{prefix}.{_RENAMES.get(local, local)}"] = grad
    for prefix, parts in conv.items():
        out[f"{prefix}.self_attn.conv1d.weight"] = torch.cat([parts[part] for part in _CONV])
    return out


def _per_token(param: torch.Tensor, tokens: tuple, leaves: list | None) -> torch.Tensor:
    """``param`` broadcast over ``tokens``; when ``leaves`` records, a leaf per token whose gradient is that token's
    share of the parameter's, carrying the parameter's own gradient as the sum."""
    if leaves is None:
        return param.expand(*tokens, -1)
    leaf = param.detach().expand(*tokens, -1).clone().requires_grad_()
    leaves.append(leaf)
    return leaf + (param - param.detach())


def _hc_forward(hc, hidden_streams, leaves: dict):
    """The reference hyper-connection's forward, its base and scale taken per token (``_per_token``)."""
    batch_size, seq_len = hidden_streams.shape[:2]
    n = hc.hc_mult
    flattened = F.linear(hc.input_norm(hidden_streams.view(batch_size, seq_len, -1).float()), hc.fn.float())
    pre_w, post_w, comb_w = flattened.split([n, n, n * n], dim=-1)
    base = _per_token(hc.base, (batch_size, seq_len), leaves.get("base"))
    scale = _per_token(hc.scale, (batch_size, seq_len), leaves.get("scale"))
    pre_b, post_b, comb_b = base.split([n, n, n * n], dim=-1)
    comb_w, comb_b = comb_w.unflatten(-1, (n, n)), comb_b.unflatten(-1, (n, n))
    pre = torch.sigmoid(pre_w * scale[..., 0:1] + pre_b) + hc.hc_eps
    post = 2 * torch.sigmoid(post_w * scale[..., 1:2] + post_b)
    comb = torch.softmax(comb_w * scale[..., 2:3, None] + comb_b, dim=-1) + hc.hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc.hc_eps)
    for _ in range(hc.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc.hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc.hc_eps)
    collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
    return post, comb, collapsed


@contextlib.contextmanager
def token_terms(layer):
    """Yields {parameter name: sum over tokens of |g_t|} for each mHC site's base and scale, filled when the context
    exits (after the backward run inside it)."""
    leaves = {f"{site}.{part}": [] for site in _SITES for part in _TOKEN_PARAMS}
    for site in _SITES:
        hc = layer.get_submodule(site)
        hc.forward = functools.partial(
            _hc_forward, hc, leaves={part: leaves[f"{site}.{part}"] for part in _TOKEN_PARAMS}
        )
    sums: dict[str, float] = {}
    try:
        yield sums
    finally:
        for site in _SITES:
            del layer.get_submodule(site).forward
    for name, recorded in leaves.items():
        grads = [leaf.grad for leaf in recorded if leaf.grad is not None]
        if grads:
            sums[name] = sum(float(g.reshape(-1, g.shape[-1]).double().norm(dim=-1).sum()) for g in grads)


def top_name(hf_name: str) -> str | None:
    """TOP's name of an mHC base; a scale packs TOP's three alpha scalars, so it has none."""
    if (match := _HC_BASE_RE.match(hf_name)) is None:
        return None
    return f"decoder.layers.{match.group(1)}.{_TOP_SITE[match.group(2)]}.bias"
