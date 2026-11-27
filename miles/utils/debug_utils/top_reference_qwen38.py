"""Qwen3.8-Flash-Next (HF ``qwen4_exp``) reference of TOP's per-layer VJP verdict (top_vjp): transformers' own model.

A decoder layer takes the 4 hyper-connection streams flat ([1, n, 4*H]); the verdict hands them apart ([1, n, 4, H]).
The PLE layer hashes the document's token ids, which the capture carries beside the streams (``carried``). The QSA
layer runs at TOP's captured index selection: the reference indexer's own selection is a Python loop per query, and a
different top-k would be another function. The frozen n-gram table (~92 GB) stays on the host when a layer moves to
the GPU and is no parameter, as in TOP.
"""

from __future__ import annotations

import contextlib
import re
from types import SimpleNamespace

import torch
from transformers import AutoModelForImageTextToText

from miles.backends.megatron_utils.megatron_to_hf.qwen3_8_next import convert_qwen3_8_next_to_hf
from miles.utils.debug_utils.top_contract import ContractModule, submodule

_TOP_EXPERT_RE = re.compile(r"mlp\.experts\.(linear_fc[12])\.weight(\d+)$")
_HF_LAYER_RE = re.compile(r"model\.language_model\.layers\.(\d+)\.(.+)$")
# The indexer's selection is discrete: neither side differentiates through it.
_INDEXER = "self_attn.indexer."


class _HostTable:
    """The frozen n-gram embedding table, kept on the host in the reference's dtype: not a module, so a layer moved to
    the GPU leaves it behind and it has no gradient to compare."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight.detach()

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(ids, self.weight)


def layers(model):
    return model.model.language_model.layers


def load(model_path: str, dtype: torch.dtype) -> torch.nn.Module:
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=dtype, attn_implementation="eager", experts_implementation="eager"
    )
    for layer in layers(model):
        if layer.ple is not None:
            embedding = layer.ple.ple_embedding
            table = embedding._modules.pop("ngram_embedding")
            embedding.ngram_embedding = _HostTable(table.weight.data)
    return model


def carried(model, capture: dict, number: int, lo: int, hi: int):
    """The document's token ids, which the PLE layer hashes into n-gram ids; nothing is differentiated."""
    # Imported here: top_reference imports this module.
    from miles.utils.debug_utils.top_reference import Carried

    return Carried(inputs={}, input_grads={}, cotangents={}, replayed={"input_ids": capture["input_ids"][lo:hi]})


def _position_embeddings(model, x, positions):
    # Text-only rows: the three M-RoPE axes carry the same position.
    return model.model.language_model.rotary_emb(x, positions[None].expand(3, -1, -1))


def run_layer(model, layer, x, positions, causal, *, carried):
    """x [1, n, 4, H] -> ([1, n, 4, H], no extra outputs)."""
    streams = x.shape[-2]
    flat = x.flatten(-2)
    linear = hasattr(
        layer, "linear_attn"
    )  # config-access-exempt: HF reference modules differ by model family in linear_attn
    out = layer(
        flat,
        position_embeddings=_position_embeddings(model, flat, positions),
        attention_mask=None if linear else causal,
        ple_input_ids=None if layer.ple is None else carried.replayed["input_ids"][None].to(x.device),
    )
    return out.unflatten(-1, (streams, -1)), {}


def _linear_attention(model, module, x, positions, causal):
    return module(x, attention_mask=None)


def _qsa_attention(model, module, x, positions, causal):
    return module(x, _position_embeddings(model, x, positions), attention_mask=causal)[0]


def _mlp(model, module, x, positions, causal):
    return module(x)


def _hyper_connection(layer, site: str) -> ContractModule:
    """TOP's hyper-connection at ``site`` ("attn" or "mlp"): the streams in, the mixed sublayer input (output.0) and
    the write gates (output.1) out. TOP's module also returns the streams it read (output.2): the same tensor as
    its input, whose captured gradient is that tensor's total, so it is no cotangent of its own and the captured
    input gradient is not judged. The PLE layer's attention site adds the PLE increment to the streams first, as
    TOP's module does; the PLE's own parameters stay outside the contract (their gradient through the streams'
    residual path is not in these two cotangents), so only the layer replays report them."""
    with_ple = site == "attn" and layer.ple is not None

    def run(model, layer, x, positions, causal, *, carried=None):
        streams = x
        if with_ple:
            streams = streams + layer.ple(streams, carried.replayed["input_ids"][None].to(x.device), None)
        mixed, _, write_gates = layer.get_submodule(f"{site}_hyper_connection")(streams)
        return mixed, write_gates

    return ContractModule(
        params=(f"{site}_hyper_connection.",),
        run=run,
        outputs=("output.0", "output.1"),
        judges_input_grad=False,
        takes_carried=with_ple,
    )


def contract(layer) -> dict[str, ContractModule]:
    """TOP's ``self_attention`` is the reference's GDN (``linear_attn``) on a linear-attention layer, else its QSA
    attention at TOP's selection; ``mlp`` is the MoE block with its gated shared expert; each hyper-connection is
    the reference's gated residual at its site."""
    attention = (
        submodule("linear_attn", _linear_attention)
        if hasattr(
            layer, "linear_attn"
        )  # config-access-exempt: HF reference modules differ by model family in linear_attn
        else submodule("self_attn", _qsa_attention)
    )
    return {
        "self_attention": attention,
        "mlp": submodule("mlp", _mlp),
        "self_attention_hyper_connection": _hyper_connection(layer, "attn"),
        "mlp_hyper_connection": _hyper_connection(layer, "mlp"),
    }


def _selected_keys(indices: torch.Tensor, lo: int) -> torch.Tensor:
    """TOP's selection for the document's rows ([n, W] packed key rows, ascending, -1 unused) as a [n, n] mask of the
    keys each row attends to."""
    keys = indices.reshape(indices.shape[0], -1).long()
    n = keys.shape[0]
    local = torch.where(keys >= 0, keys - lo, n)
    if bool(((local < 0) | ((keys >= 0) & (local >= n))).any()):
        raise ValueError("[top] a captured QSA selection reaches outside its document")
    selected = torch.zeros(n, n + 1, dtype=torch.bool).scatter_(1, local, True)[:, :n]
    if bool(selected.triu(1).any()) or not bool(selected.diagonal().all()):
        raise ValueError("[top] a captured QSA selection is not causal or misses its own row")
    return selected


@contextlib.contextmanager
def replayed_qsa(layer, indices: torch.Tensor | None, lo: int):
    """The reference attention over TOP's selected keys: its indexer returns the selection mask TOP's chose."""
    attn = getattr(
        layer, "self_attn", None
    )  # config-access-exempt: HF reference modules differ by model family in self_attn
    if attn is None:
        yield
        return
    if indices is None:
        raise ValueError("[top] a QSA layer is replayed at TOP's captured index selection (top_index_selection)")
    selected = _selected_keys(indices, lo)

    def forward(hidden_states, position_embeddings, attention_mask, past_key_values):
        mask = selected.to(attention_mask.device)[None, None]
        if attention_mask.is_floating_point():
            return torch.where(mask, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min)
        return mask

    attn.indexer.forward = forward
    try:
        yield
    finally:
        del attn.indexer.forward


def converted_param_grads(param_grads: dict[str, torch.Tensor], config) -> tuple[dict, dict]:
    """({reference parameter name: grad}, {layer prefix: {(global expert id, linear_fc1|2): grad}}): the reference
    layer's names through the weight-sync converter, which splits Megatron's fused QKV rows and SwiGLU fc1; the
    per-expert gradients are returned apart, to be stacked as the reference's fused expert parameters."""
    # The Qwen3.5 converter reads Megatron flags from ``args.backend``, as on a trainer config.
    layout = SimpleNamespace(
        backend=SimpleNamespace(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_query_groups=config.num_key_value_heads,
            kv_channels=config.head_dim,
        )
    )
    out, experts = {}, {}
    for name, grad in param_grads.items():
        local = re.search(r"decoder\.layers\.(\d+)\.(.+)$", name)
        if (expert := _TOP_EXPERT_RE.fullmatch(local.group(2))) is not None:
            experts.setdefault(f"model.layers.{local.group(1)}", {})[int(expert.group(2)), expert.group(1)] = grad
            continue
        for hf_name, hf_grad in convert_qwen3_8_next_to_hf(layout, f"module.module.{local.group(0)}", grad):
            number, hf_local = _HF_LAYER_RE.fullmatch(hf_name).groups()
            if hf_local.startswith(_INDEXER):
                if bool(hf_grad.count_nonzero()):
                    raise ValueError(f"[top] {name}: the QSA indexer has a gradient, but its selection is discrete")
                continue
            out[f"model.layers.{number}.{hf_local}"] = hf_grad
    return out, experts
