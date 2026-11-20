"""Inkling2 attention residuals across pipeline stages: a stage's exit layer packs the
completed blocks into the one tensor Megatron sends, and the next stage's entry layer unpacks
them, so the pipelined forward and backward match an unsplit run."""

from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"], num_gpus=1
)

# Inkling2's layers import Transformer Engine; the checks themselves run on CPU tensors.
pytest.importorskip("transformer_engine")

from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

from miles_plugins.models.inkling2.layers import (  # noqa: E402
    Inkling2RMSNorm,
    Inkling2TransformerLayer,
    pipeline_stage_edges,
)

HIDDEN, TOKENS, EVERY, WIDTH = 8, 7, 3, 3


def _pp_config(num_layers: int, pp: int, first: int | None = None, last: int | None = None) -> TransformerConfig:
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=HIDDEN,
        num_attention_heads=1,
        pipeline_model_parallel_size=pp,
        num_layers_in_first_pipeline_stage=first,
        num_layers_in_last_pipeline_stage=last,
        pipeline_dtype=torch.float32,
    )


def _stages(config: TransformerConfig) -> list[range]:
    edges = [pipeline_stage_edges(config, idx) for idx in range(config.num_layers)]
    starts = [0, *(idx for idx, (entry, _) in enumerate(edges) if entry), config.num_layers]
    for idx, (_, is_exit) in enumerate(edges):
        assert is_exit == (idx + 1 in starts[1:-1]), f"layer {idx}: exit flag disagrees with the next entry"
    return [range(a, b) for a, b in pairwise(starts)]


def test_stage_edges_follow_uneven_first_and_last_stages() -> None:
    """48 layers over PP8 with 2 first and 4 last layers: 2 / 7 x 6 / 4."""
    stages = _stages(_pp_config(48, 8, first=2, last=4))
    assert [len(stage) for stage in stages] == [2, 7, 7, 7, 7, 7, 7, 4]


def test_stage_edges_are_empty_without_pipeline_parallelism() -> None:
    config = _pp_config(6, 1)
    assert all(pipeline_stage_edges(config, idx) == (False, False) for idx in range(6))


class _Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(x))


def _layer(idx: int, num_layers: int, edges: tuple[bool, bool]) -> Inkling2TransformerLayer:
    """The real forward and mixing on stand-in attention and MLP (no TE or parallel state)."""
    torch.manual_seed(idx)
    config = SimpleNamespace(num_layers=num_layers, hidden_size=HIDDEN, params_dtype=torch.float32)
    layer = Inkling2TransformerLayer.__new__(Inkling2TransformerLayer)
    nn.Module.__init__(layer)
    layer.config = config
    layer.inkling2 = SimpleNamespace(n_layers_per_attn_res_block=EVERY, attn_res_normalize_values=True)
    layer.layer_idx, layer.layer_number, layer.is_local = idx, idx + 1, True
    layer.is_stage_entry, layer.is_stage_exit = edges
    layer.attn_norm = Inkling2RMSNorm(config, HIDDEN)
    layer.pre_mlp_layernorm = Inkling2RMSNorm(config, HIDDEN)
    if idx > 0:
        layer.attn_query = nn.Parameter(torch.randn(HIDDEN))
    layer.mlp_query = nn.Parameter(torch.randn(HIDDEN))
    layer.mlp_sconv_weight = nn.Parameter(0.1 * torch.randn(HIDDEN, 1, WIDTH))
    if idx + 1 == num_layers:
        layer.output_query = nn.Parameter(torch.randn(HIDDEN))
    attention, mlp = _Linear(), _Linear()
    layer.attention_proj, layer.mlp_proj = attention, mlp
    layer.self_attention = lambda inputs, packed_seq_params: attention(inputs[0])
    layer.mlp = lambda x: (mlp(x), None)
    return layer


def _run(layers, x, packed):
    context = None
    for layer in layers:
        x, context = layer.forward(x, context=context, packed_seq_params=packed)
    return x


@pytest.mark.parametrize("pp,first,last", [(2, None, None), (3, 1, 2), (4, None, None)])
def test_pipelined_forward_and_backward_match_one_stage(pp: int, first: int | None, last: int | None) -> None:
    num_layers = 8
    config = _pp_config(num_layers, pp, first, last)
    stages = _stages(config)
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 3, TOKENS]))
    embedding = torch.randn(TOKENS, 1, HIDDEN)

    reference = [_layer(idx, num_layers, (False, False)) for idx in range(num_layers)]
    pipelined = [_layer(idx, num_layers, pipeline_stage_edges(config, idx)) for idx in range(num_layers)]
    ref_input = embedding.clone().requires_grad_()
    ref_out = _run(reference, ref_input, packed)

    # Each stage starts from a received leaf tensor; backward hands its grad to the stage before.
    stage_inputs, stage_outputs = [], []
    x = embedding.clone().requires_grad_()
    for stage in stages:
        stage_inputs.append(x)
        out = _run([pipelined[idx] for idx in stage], x, packed)
        stage_outputs.append(out)
        if stage.stop < num_layers:
            assert out.shape[-1] > HIDDEN, "a stage before the last must send its completed blocks"
            x = out.detach().requires_grad_()
    torch.testing.assert_close(stage_outputs[-1], ref_out, rtol=0, atol=0)

    grad = torch.randn_like(ref_out)
    ref_out.backward(grad)
    for stage_input, stage_output in zip(reversed(stage_inputs), reversed(stage_outputs), strict=True):
        stage_output.backward(grad)
        grad = stage_input.grad
    torch.testing.assert_close(grad, ref_input.grad)
    for ref_layer, pp_layer in zip(reference, pipelined, strict=True):
        for (name, ref_param), (_, pp_param) in zip(
            ref_layer.named_parameters(), pp_layer.named_parameters(), strict=True
        ):
            torch.testing.assert_close(pp_param.grad, ref_param.grad, msg=f"layer {ref_layer.layer_idx} {name}")


def test_entry_layer_rejects_a_payload_with_the_wrong_block_count() -> None:
    layer = _layer(4, 8, (True, False))
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, TOKENS]))
    # Layer 4 with blocks every 3 layers expects the embedding and block 3 (2 rows), not 1.
    with pytest.raises(AssertionError, match="stage-boundary payload width"):
        layer.forward(torch.randn(TOKENS, 1, 2 * HIDDEN), packed_seq_params=packed)


def test_only_block_boundaries_build_a_new_context() -> None:
    """Full recompute keeps every layer's input context until backward, so a layer that adds no
    block must pass its input context through instead of stacking a copy of the blocks."""
    num_layers = 8
    layers = [_layer(idx, num_layers, (False, False)) for idx in range(num_layers)]
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 3, TOKENS]))
    x, context, blocks = torch.randn(TOKENS, 1, HIDDEN), None, []
    for layer in layers:
        out, new_context = layer.forward(x, context=context, packed_seq_params=packed)
        if layer.layer_idx % EVERY == 0:
            blocks.append(x)
            torch.testing.assert_close(new_context, torch.stack(blocks, dim=-2), rtol=0, atol=0)
            assert new_context is not context, f"layer {layer.layer_idx}: a boundary must add its block"
        else:
            assert new_context is context, f"layer {layer.layer_idx}: a non-boundary layer copied the context"
        x, context = out, new_context
