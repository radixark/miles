"""Check indexer arithmetic and packed region selection against independent CPU references."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")
from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from miles_plugins.models.step5.indexer import (  # noqa: E402
    CHECKPOINT_NAMES,
    FP32_PARAMS,
    INDEXER_ROPE,
    TP_SHARDED_PARAMS,
    Step5Indexer,
    read_checkpoint_layer,
)
from miles_plugins.models.step5.indexer_ops import (  # noqa: E402
    WEIGHT_SCALE,
    _triton_sum,
    compress_regions,
    norm_rope_key,
    norm_rope_query,
    rope_cos_sin,
    score_regions,
    to_e4m3,
    top_regions,
)
from miles_plugins.models.step5.region_selection import MAX_SELECTED_REGIONS, REGION_SIZE, TOP_REGIONS  # noqa: E402

_HIDDEN = 16
_EPS = 1e-5


def _grid(*shape: int, scale: float, generator: torch.Generator) -> torch.Tensor:
    """Small multiples of ``scale``: the projections of such values are exact in BF16 in any order."""
    return torch.randint(-2, 3, shape, generator=generator).float() * scale


def _checkpoint_layer(seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        CHECKPOINT_NAMES["q_weight"]: _grid(16 * 256, _HIDDEN, scale=2**-4, generator=generator).bfloat16(),
        CHECKPOINT_NAMES["k_weight"]: _grid(256, _HIDDEN, scale=2**-4, generator=generator).bfloat16(),
        CHECKPOINT_NAMES["z_weight"]: _grid(256, _HIDDEN, scale=2**-4, generator=generator).bfloat16(),
        CHECKPOINT_NAMES["w_weight"]: _grid(16, _HIDDEN, scale=2**-4, generator=generator),
        CHECKPOINT_NAMES["q_norm_weight"]: torch.randn(256, generator=generator) * 0.1,
        CHECKPOINT_NAMES["k_norm_weight"]: 1 + torch.randn(256, generator=generator) * 0.1,
        CHECKPOINT_NAMES["k_norm_bias"]: torch.randn(256, generator=generator) * 0.1,
        # one per attention head of _indexer's config
        CHECKPOINT_NAMES["ssmax_s"]: torch.full((4,), 0.085),
    }


def _indexer(tp_size: int = 1, tp_rank: int = 0, **kwargs) -> Step5Indexer:
    config = TransformerConfig(
        num_layers=1, hidden_size=_HIDDEN, num_attention_heads=4, num_query_groups=4, layernorm_epsilon=_EPS
    )
    tp_group = None if tp_size == 1 else SimpleNamespace(size=lambda: tp_size, rank=lambda: tp_rank)
    indexer = Step5Indexer(config, tp_group=tp_group, **kwargs)
    indexer.load_checkpoint_layer(_checkpoint_layer())
    return indexer


def _pack(lengths: list[int], seed: int = 1) -> tuple[torch.Tensor, PackedSeqParams]:
    generator = torch.Generator().manual_seed(seed)
    hidden = _grid(sum(lengths), _HIDDEN, scale=2**-3, generator=generator).bfloat16()
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    return hidden, PackedSeqParams(cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens, qkv_format="thd")


def _reference_top(scores: list[float], candidates: int) -> list[int]:
    """Descending score, NaN first, equal scores (-0 == +0) to the lower id; then ascending ids."""
    order = sorted(
        range(candidates), key=lambda r: (not math.isnan(scores[r]), -scores[r] if scores[r] == scores[r] else 0, r)
    )
    return sorted(order[:TOP_REGIONS])


def _reference_sequence(indexer: Step5Indexer, tokens: torch.Tensor) -> list[list[list[int]]]:
    """One sequence, token by token: every earlier region while there are at most 512, else the top 512."""
    length, groups = tokens.shape[0], indexer.num_local_groups
    cos, sin = rope_cos_sin(INDEXER_ROPE, length, tokens.device)
    complete = length // REGION_SIZE * REGION_SIZE
    key = norm_rope_key(
        (tokens[:complete].double() @ indexer.k_weight.double().T).bfloat16(),
        indexer.k_norm_weight,
        indexer.k_norm_bias,
        cos[:complete],
        sin[:complete],
        _EPS,
    )
    logits = (tokens[:complete].double() @ indexer.z_weight.double().T).bfloat16()
    region_keys = compress_regions(key.view(-1, 8, 256), logits.view(-1, 8, 256))
    rows = []
    for position in range(length):
        current = position // REGION_SIZE
        if current <= TOP_REGIONS:
            history = [list(range(current))] * groups
        else:
            q = (tokens[position : position + 1].double() @ indexer.q_weight.double().T).bfloat16()
            q = to_e4m3(
                norm_rope_query(
                    q.view(1, -1, 256),
                    indexer.q_norm_weight,
                    cos[position : position + 1],
                    sin[position : position + 1],
                    _EPS,
                )
            )
            w = (tokens[position : position + 1].double() @ indexer.w_weight.bfloat16().double().T).bfloat16().float()
            scores = score_regions(
                q.view(1, groups, 4, 256), w.view(1, groups, 4) * WEIGHT_SCALE, region_keys[:current]
            )
            history = [_reference_top(scores[0, group].tolist(), current) for group in range(groups)]
        rows.append([ids + [current] + [-1] * (MAX_SELECTED_REGIONS - len(ids) - 1) for ids in history])
    return rows


def test_rope_table_is_llama3_with_unit_factor():
    cos, sin = rope_cos_sin(INDEXER_ROPE, 5000, torch.device("cpu"))
    position = torch.arange(5000, dtype=torch.float64)[:, None]
    angle = position * INDEXER_ROPE.theta ** (-torch.arange(0, 32, 2, dtype=torch.float64) / 32)
    torch.testing.assert_close(cos.double(), angle.cos(), rtol=0, atol=2**-8)
    torch.testing.assert_close(sin.double(), angle.sin(), rtol=0, atol=2**-8)


def test_norm_sums_follow_the_warp_shuffle_tree():
    x = torch.ones(256)
    x[0] = 2.0**24
    lanes = (x[0::2] + x[1::2]).view(4, 32).clone()
    for offset in (16, 8, 4, 2, 1):
        lanes = lanes + lanes[:, torch.arange(32) ^ offset]
    warps = lanes[:, 0]
    assert _triton_sum(x) == (warps[0] + warps[2]) + (warps[1] + warps[3]) == 2.0**24 + 254
    sequential = torch.zeros((), dtype=torch.float32)
    for value in x:
        sequential = sequential + value
    assert sequential == 2.0**24  # the order is part of the result


def test_norms_and_rope_match_their_definitions():
    generator = torch.Generator().manual_seed(0)
    query = (torch.randn(64, 4, 256, generator=generator) * 0.7).bfloat16()
    key = (torch.randn(64, 256, generator=generator) * 1.3 + 0.2).bfloat16()
    q_weight, k_weight, k_bias = torch.randn(3, 256, generator=generator) * 0.1
    cos, sin = rope_cos_sin(INDEXER_ROPE, 64, torch.device("cpu"))

    def rotate(y):
        c, s = cos.double(), sin.double()
        if y.ndim == 3:
            c, s = c[:, None], s[:, None]
        first, second = y[..., :16], y[..., 16:32]
        return torch.cat([first * c - second * s, second * c + first * s, y[..., 32:]], -1).bfloat16()

    x = query.double()
    rms = x / (x.pow(2).mean(-1, keepdim=True) + _EPS).sqrt() * (1 + q_weight.bfloat16().double())
    expected_q = rotate(rms.bfloat16().double())
    x = key.double()
    layer_norm = (x - x.mean(-1, keepdim=True)) / (x.var(-1, unbiased=False, keepdim=True) + _EPS).sqrt()
    expected_k = rotate((layer_norm * k_weight.bfloat16().double() + k_bias.bfloat16().double()).bfloat16().double())

    for got, want in (
        (norm_rope_query(query, q_weight, cos, sin, _EPS), expected_q),
        (norm_rope_key(key, k_weight, k_bias, cos, sin, _EPS), expected_k),
    ):
        # FP32 statistics can round a BF16 boundary the other way: rarely, and by one ulp.
        assert got.dtype == torch.bfloat16
        assert (got == want).float().mean() > 0.99
        torch.testing.assert_close(got.double(), want.double(), rtol=2**-6, atol=2**-4)


def test_compression_is_a_per_channel_softmax_under_one_region_maximum():
    generator = torch.Generator().manual_seed(0)
    key = torch.randn(200, 8, 256, generator=generator).bfloat16()
    logits = (torch.randn(200, 8, 256, generator=generator) * 3).bfloat16()
    z, k = logits.double(), key.double()
    weight = (z - z.amax(dim=(1, 2), keepdim=True)).exp()
    expected = to_e4m3(((weight * k).sum(1) / weight.sum(1)).bfloat16())
    got = compress_regions(key, logits)
    assert got.dtype == torch.float8_e4m3fn
    # exp2 and FP32 sums can round a BF16 or E4M3 boundary the other way: rarely, by one E4M3 step.
    got, expected = got.float(), expected.float()
    assert (got == expected).float().mean() > 0.99
    assert ((got - expected).abs() <= expected.abs() / 8 + 2**-9).all()

    # One maximum spans every channel: a channel 200 below it has an empty denominator and yields 0,
    # where a per-channel maximum would give the mean of its keys.
    logits = torch.zeros(1, 8, 256, dtype=torch.bfloat16)
    logits[..., 0] = -200
    assert compress_regions(torch.ones(1, 8, 256, dtype=torch.bfloat16), logits)[0, 0].float() == 0
    # Equal logits average the keys; results past 448 saturate.
    key = torch.full((1, 8, 256), 1000.0, dtype=torch.bfloat16)
    assert compress_regions(key, torch.zeros_like(key))[0].float().unique().tolist() == [448.0]


def test_e4m3_rounds_to_nearest_even_and_saturates():
    values = torch.tensor([17.0, 19.0, 448.0, 460.0, 1000.0, -1000.0, float("inf"), float("nan")])
    assert to_e4m3(values).float()[:7].tolist() == [16.0, 20.0, 448.0, 448.0, 448.0, -448.0, 448.0]
    assert to_e4m3(values).float()[7].isnan()


def test_scores_sum_heads_pairwise_and_drop_nan_dots():
    region_keys = torch.zeros(3, 256)
    region_keys[:, :16] = 256.0
    query = torch.zeros(1, 1, 4, 256)
    query[..., :16] = 256.0
    weights = torch.tensor([[[2.0**6, 1.0 / 2**20, -(2.0**6), 1.0 / 2**20]]])
    # Terms 2**26, 1, -2**26, 1: (t0 + t2) + (t1 + t3) keeps both ones; a sequential sum loses one.
    scores = score_regions(to_e4m3(query), weights, to_e4m3(region_keys))
    assert scores.tolist() == [[[2.0, 2.0, 2.0]]]

    generator = torch.Generator().manual_seed(0)
    query = to_e4m3(torch.randn(8, 2, 4, 256, generator=generator))
    region_keys = to_e4m3(torch.randn(40, 256, generator=generator))
    weights = torch.randn(8, 2, 4, generator=generator)
    dots = query.double() @ region_keys.double().T
    expected = (dots.clamp(min=0) * weights.double()[..., None]).sum(2)
    torch.testing.assert_close(score_regions(query, weights, region_keys).double(), expected, rtol=1e-6, atol=1e-6)

    # A NaN query channel (a NaN hidden state) makes that head's dots NaN, which PTX max turns into 0.
    nan_query = query.float()
    nan_query[0, 0, 1, 0] = float("nan")
    without_head = weights.clone()
    without_head[0, 0, 1] = 0
    scores = score_regions(nan_query.to(torch.float8_e4m3fn), weights, region_keys)
    assert not scores.isnan().any()
    assert torch.equal(scores[0, 0], score_regions(query, without_head, region_keys)[0, 0])


@pytest.mark.parametrize("candidates", [513, 514, 600])
def test_top_regions_rank_like_deepselect(candidates):
    generator = torch.Generator().manual_seed(candidates)
    scores = torch.randint(-3, 4, (6, 2, 600), generator=generator).float() / 4
    scores[scores == 0] = torch.tensor([0.0, -0.0])[
        torch.randint(0, 2, (int((scores == 0).sum()),), generator=generator)
    ]
    scores[0, 0, 5] = scores[1, 1, 590] = float("nan")
    scores[2] = 0.0
    rows = torch.full((6,), candidates)
    rows[3] = 513
    got = top_regions(scores, rows)
    assert got.shape == (6, 2, TOP_REGIONS) and got.dtype == torch.int32
    for t in range(6):
        for g in range(2):
            assert got[t, g].tolist() == _reference_top(scores[t, g].tolist(), int(rows[t]))


def test_selection_matches_a_per_token_walk_of_the_pack():
    lengths = [4105, 37, 4104, 4400]
    hidden, packed = _pack(lengths)
    indexer = _indexer()
    selection = indexer.select_regions(hidden[:, None], packed)

    expected, start = [], 0
    for length in lengths:
        expected += _reference_sequence(indexer, hidden[start : start + length])
        start += length
    assert selection.region_ids.dtype == torch.int32
    assert selection.region_ids.shape == (sum(lengths), 4, MAX_SELECTED_REGIONS)
    assert selection.region_ids.tolist() == expected
    assert torch.equal(selection.cu_seqlens, packed.cu_seqlens_q)
    # 512 earlier regions are all kept (last token of the third sequence); 513 lose one (first sequence).
    assert (selection.region_ids[4105 + 37 + 4103] == torch.arange(MAX_SELECTED_REGIONS, dtype=torch.int32)).all()
    assert (selection.region_ids[4104, :, -1] == 513).all() and (selection.region_ids[4104] != -1).all()
    # The scored rows of the last sequence select by score, not simply the oldest 512 regions.
    assert (selection.region_ids[4105 + 37 + 4104 + 4104 :, :, TOP_REGIONS - 1] > TOP_REGIONS - 1).float().mean() > 0.5

    chunked = _indexer(score_chunk_elements=16 * 600 * 7).select_regions(hidden[:, None], packed)
    assert torch.equal(chunked.region_ids, selection.region_ids)


def test_the_pack_must_cover_every_token():
    hidden, packed = _pack([40, 30])

    def select(cu_seqlens: torch.Tensor):
        return _indexer().select_regions(hidden[:, None], PackedSeqParams(cu_seqlens_q=cu_seqlens, qkv_format="thd"))

    with pytest.raises(AssertionError, match="cover its 70 tokens"):
        select(packed.cu_seqlens_q[:-1])
    with pytest.raises(AssertionError, match="int64 cu_seqlens"):
        select(packed.cu_seqlens_q.long())


def test_tp_ranks_select_for_their_own_groups():
    hidden, packed = _pack([4200, 4150])
    full = _indexer().select_regions(hidden[:, None], packed).region_ids
    for tp_size in (2, 4):
        for tp_rank in range(tp_size):
            groups = slice(tp_rank * 4 // tp_size, (tp_rank + 1) * 4 // tp_size)
            local = _indexer(tp_size, tp_rank).select_regions(hidden[:, None], packed).region_ids
            assert torch.equal(local, full[:, groups])


def test_parameters_are_frozen_fp32_where_the_checkpoint_is_and_tp_split_by_group():
    indexer = _indexer(tp_size=2, tp_rank=1)
    checkpoint = _checkpoint_layer()
    assert not any(param.requires_grad for param in indexer.parameters())
    for name in FP32_PARAMS:
        param = getattr(indexer, name)
        assert param.dtype == torch.float32 and param.keep_in_fp32
    for name in TP_SHARDED_PARAMS:
        param = getattr(indexer, name)
        assert param.tensor_model_parallel and param.partition_dim == 0
        torch.testing.assert_close(param, checkpoint[CHECKPOINT_NAMES[name]].chunk(2)[1], rtol=0, atol=0)
    torch.testing.assert_close(indexer.w_weight, checkpoint[CHECKPOINT_NAMES["w_weight"]][8:], rtol=0, atol=0)


def test_checkpoint_layer_reads_one_layer_from_safetensors(tmp_path):
    tensors = _checkpoint_layer()
    named = {f"model.layers.3.self_attn.{name}": tensor for name, tensor in tensors.items()}
    save_file(named | {"model.layers.2.input_layernorm.weight": torch.ones(4)}, tmp_path / "model.safetensors")
    weight_map = {name: "model.safetensors" for name in named}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    loaded = read_checkpoint_layer(str(tmp_path), 3)

    assert loaded.keys() == tensors.keys()
    for name, tensor in tensors.items():
        assert loaded[name].dtype == tensor.dtype
        torch.testing.assert_close(loaded[name], tensor, rtol=0, atol=0)
