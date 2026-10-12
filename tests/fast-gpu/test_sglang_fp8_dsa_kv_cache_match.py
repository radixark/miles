"""--match-sglang-fp8-dsa-kv-cache-values against SGLang's own FP8 DSA KV cache.

The sampler writes each key with quantize_k_cache_separate + set_mla_kv_buffer_triton into
its paged cache, and its prefill attention reads it back with dequantize_k_cache_paged. The
trainer value must equal those cache bytes and BF16 bits exactly: a tolerance would hide
reciprocal, rounding or layout differences, which is the mismatch this feature removes.
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=180,
    suite="stage-b-2-gpu-h200",
    labels=["precision", "megatron"],
    hardware=["hopper", "blackwell"],
    num_gpus=1,
)

import socket
import sys

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state, tensor_parallel
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.experimental_attention_variant_module_specs import get_dsa_module_spec_for_backend
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from sglang.kernels.ops.attention.dsa.dequant_k_cache import dequantize_k_cache_paged
from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache, quantize_k_cache_separate
from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_triton

from miles.backends.megatron_utils.sglang_fp8_dsa_kv_cache_match import (
    match_sglang_fp8_dsa_kv_cache_value,
    register_sglang_fp8_dsa_kv_cache_match_hook,
    register_sglang_fp8_dsa_kv_cache_match_hooks,
    sglang_fp8_dsa_kv_cache_value,
)

NOPE_DIM = 512
ROPE_DIM = 64
KV_DIM = NOPE_DIM + ROPE_DIM
CACHE_ROW_BYTES = NOPE_DIM + 4 * (NOPE_DIM // 128) + 2 * ROPE_DIM
# Steps between adjacent E4M3 magnitudes in [1, 448]; midpoints exercise round-to-nearest-even.
E4M3_MAX = 448.0


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.int16 if tensor.element_size() == 2 else torch.uint8)


def _sampler_cache(kv: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache rows and prefill-read BF16 values, as SGLang's DSA FP8 KV pool produces them.

    Mirrors DSATokenToKVPool: a uint8 [slots, 1, 656] buffer with padded slot 0, written by
    _write_mla_kv_buffer at out-of-order locations, read through its float8 view.
    """
    rows = kv.reshape(-1, 1, KV_DIM)
    num_tokens = rows.shape[0]
    num_slots = 2 * num_tokens + 1
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loc = (torch.randperm(num_slots - 1, generator=generator)[:num_tokens] + 1).to(kv.device)
    kv_buffer = torch.zeros((num_slots, 1, CACHE_ROW_BYTES), dtype=torch.uint8, device=kv.device)

    # The model passes the latent and RoPE parts as strided views of one projection output.
    nope_bytes, rope_bytes = quantize_k_cache_separate(rows[..., :NOPE_DIM], rows[..., NOPE_DIM:])
    set_mla_kv_buffer_triton(kv_buffer, loc, nope_bytes, rope_bytes)

    cache_rows = kv_buffer[loc]
    prefill_read = dequantize_k_cache_paged(kv_buffer.view(torch.float8_e4m3fn), loc.to(torch.int32))
    return cache_rows, prefill_read.reshape(kv.shape)


def _trainer_cache_rows(kv: torch.Tensor) -> torch.Tensor:
    return quantize_k_cache(kv.reshape(-1, 1, 1, KV_DIM)).view(torch.uint8).view(-1, 1, CACHE_ROW_BYTES)


def _random_dynamic_range(num_tokens: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.randn((num_tokens, 1, KV_DIM), device="cuda", generator=generator)
    block_scale = torch.exp(
        torch.empty((num_tokens, 1, KV_DIM // 64, 1), device="cuda").uniform_(-7, 7, generator=generator)
    )
    return (values.view(num_tokens, 1, KV_DIM // 64, 64) * block_scale).view(num_tokens, 1, KV_DIM).bfloat16()


def _e4m3_rounding_boundaries(num_tokens: int, seed: int) -> torch.Tensor:
    """Blocks whose scaled values sit on and one BF16 step around E4M3 rounding midpoints."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    grid = torch.arange(256, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn).float()
    grid = grid[torch.isfinite(grid) & (grid > 0)].unique().sort().values
    midpoints = (grid[:-1] + grid[1:]) / 2
    picks = midpoints[torch.randint(len(midpoints), (num_tokens, 1, KV_DIM), device="cuda", generator=generator)]
    signs = torch.where(torch.rand(picks.shape, device="cuda", generator=generator) < 0.5, -1.0, 1.0)
    exponents = torch.randint(-20, 21, (num_tokens, 1, 1), device="cuda", generator=generator)
    scale = torch.exp2(exponents.float())
    values = (picks * signs * scale).bfloat16()
    step = torch.randint(-1, 2, values.shape, device="cuda", generator=generator).to(torch.int16)
    values = (values.view(torch.int16) + step).view(torch.bfloat16)
    # Pin each 128-block's amax so the block scale is exactly the one the midpoints were drawn for.
    values.view(num_tokens, 1, KV_DIM)[..., 0:NOPE_DIM:128] = (E4M3_MAX * scale).bfloat16()
    return values


def _special_values(num_tokens: int, seed: int) -> torch.Tensor:
    """Signed zeros, single nonzero entries, BF16 extremes, subnormals and exact powers of two."""
    values = _random_dynamic_range(num_tokens, seed)
    bf16 = torch.finfo(torch.bfloat16)
    blocks = values.view(num_tokens, 1, KV_DIM)
    blocks[0::7, :, 0:128] = -0.0
    blocks[1::7, :, 128:256] = 0.0
    blocks[1::7, :, 129] = 3.0
    blocks[2::7, :, 256:384] = bf16.max
    blocks[2::7, :, 257] = -bf16.max
    blocks[3::7, :, 384:512] = bf16.tiny
    blocks[3::7, :, 385] = bf16.tiny / 4
    blocks[4::7, :, 0:128] = torch.exp2(torch.arange(-60, 68, device="cuda", dtype=torch.float32)).bfloat16()
    blocks[5::7, :, NOPE_DIM:] = -0.0
    return values


_GENERATORS = {
    "dynamic_range": _random_dynamic_range,
    "rounding_boundaries": _e4m3_rounding_boundaries,
    "special_values": _special_values,
}


@pytest.mark.parametrize("values", sorted(_GENERATORS))
# 768 locations switch set_mla_kv_buffer_triton from its Triton to its TMA bulk-store write.
@pytest.mark.parametrize("num_tokens", [1, 7, 767, 768, 16384])
def test_trainer_value_matches_sampler_cache_bits(values: str, num_tokens: int) -> None:
    kv = _GENERATORS[values](num_tokens, seed=num_tokens)
    cache_rows, prefill_read = _sampler_cache(kv, seed=num_tokens)

    assert torch.equal(_trainer_cache_rows(kv), cache_rows)
    trainer_value = sglang_fp8_dsa_kv_cache_value(kv)
    assert torch.equal(_bits(trainer_value), _bits(prefill_read))
    assert torch.equal(_bits(trainer_value[..., NOPE_DIM:]), _bits(kv[..., NOPE_DIM:]))


def test_trainer_value_keeps_layout_and_matches_per_token() -> None:
    kv = _random_dynamic_range(4 * 3 * 64, seed=3).view(64, 3, 4, 1, KV_DIM)[:, :, 0].contiguous()
    trainer_value = sglang_fp8_dsa_kv_cache_value(kv)
    _, prefill_read = _sampler_cache(kv, seed=3)

    assert trainer_value.shape == kv.shape
    assert torch.equal(_bits(trainer_value), _bits(prefill_read))
    # Context and tensor parallelism hand each rank a subset of tokens; the value must not depend on it.
    halves = [sglang_fp8_dsa_kv_cache_value(part.contiguous()) for part in kv.chunk(2, dim=0)]
    assert torch.equal(_bits(torch.cat(halves)), _bits(trainer_value))


def test_all_zero_latent_block_matches_sampler() -> None:
    kv = _random_dynamic_range(8, seed=5)
    kv[:, :, 128:256] = 0
    cache_rows, prefill_read = _sampler_cache(kv, seed=5)

    assert torch.equal(_trainer_cache_rows(kv), cache_rows)
    trainer_value = sglang_fp8_dsa_kv_cache_value(kv)
    assert torch.equal(_bits(trainer_value), _bits(prefill_read))
    assert torch.isfinite(trainer_value[:, :, 128:256]).all()
    assert (trainer_value[:, :, 128:256] == 0).all()


def test_empty_token_partition() -> None:
    kv = torch.empty((0, 1, KV_DIM), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    served = match_sglang_fp8_dsa_kv_cache_value(kv)
    served.sum().backward()

    assert served.shape == kv.shape
    assert served.numel() == 0
    assert kv.grad is not None and kv.grad.numel() == 0


def test_backward_is_straight_through() -> None:
    kv = _random_dynamic_range(256, seed=7).requires_grad_(True)
    grad_output = torch.randn_like(kv)
    served = match_sglang_fp8_dsa_kv_cache_value(kv)
    served.backward(grad_output)

    assert torch.equal(_bits(served.detach()), _bits(sglang_fp8_dsa_kv_cache_value(kv.detach())))
    assert torch.equal(_bits(kv.grad), _bits(grad_output))


@pytest.mark.parametrize(
    "kv",
    [
        pytest.param(lambda: torch.zeros(4, 1, KV_DIM, device="cuda"), id="float32"),
        pytest.param(lambda: torch.zeros(4, 1, KV_DIM + 64, device="cuda", dtype=torch.bfloat16), id="width"),
        pytest.param(lambda: torch.zeros(4, 2, KV_DIM, device="cuda", dtype=torch.bfloat16), id="heads"),
        pytest.param(
            lambda: torch.zeros(4, 1, 2 * KV_DIM, device="cuda", dtype=torch.bfloat16)[..., :KV_DIM], id="strided"
        ),
    ],
)
def test_rejects_keys_outside_the_cache_format(kv) -> None:
    with pytest.raises(ValueError):
        sglang_fp8_dsa_kv_cache_value(kv())


@pytest.fixture(scope="module")
def megatron_single_rank():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    yield
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _dsa_config() -> MLATransformerConfig:
    return MLATransformerConfig(
        multi_latent_attention=True,
        experimental_attention_variant="dsa",
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=16,
        q_lora_rank=512,
        kv_lora_rank=NOPE_DIM,
        qk_head_dim=128,
        qk_pos_emb_head_dim=ROPE_DIM,
        v_head_dim=128,
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=64,
        dsa_indexer_loss_coeff=0.0,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        qk_layernorm=True,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        apply_rope_fusion=False,
        rope_type="rope",
        rotary_base=10000,
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 1, multiplier=2.0),
        gradient_accumulation_fusion=False,
        attention_backend=AttnBackend.unfused,
        dsa_kernel_backend="none",
    )


class _KeyRecorder:
    def __init__(self) -> None:
        self.keys: list[torch.Tensor] = []

    def __call__(self, module, args, kwargs):
        self.keys.append(args[1].detach().clone())


def _build_attention(config: MLATransformerConfig, seed: int):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    spec = get_dsa_module_spec_for_backend(config=config, backend=TESpecProvider())
    return build_module(spec, config=config, layer_number=1, cp_comm_type=None, pg_collection=None).cuda()


def _model_chunk_with_attention(attention: torch.nn.Module) -> torch.nn.Module:
    """Wrap one real attention in a TransformerLayer shell to exercise production discovery."""
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.layer_number = 1
    layer.self_attention = attention
    chunk = torch.nn.Module()
    chunk.add_module("layer", layer)
    return chunk


def _attention_inputs(config: MLATransformerConfig, seqlen: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden_states = torch.randn(
        (seqlen, 1, config.hidden_size), dtype=torch.bfloat16, device="cuda", generator=generator
    ).requires_grad_(True)
    output_grad = torch.randn(
        (seqlen, 1, config.hidden_size), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    mask = torch.triu(torch.full((1, 1, seqlen, seqlen), float("-inf"), device="cuda"), diagonal=1)
    return hidden_states, output_grad, mask


def _grads(module: torch.nn.Module, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
    grads = {name: param.grad.clone() for name, param in module.named_parameters() if param.grad is not None}
    grads["hidden_states"] = hidden_states.grad.clone()
    return grads


def test_hook_feeds_attention_the_sampler_cache_value(megatron_single_rank) -> None:
    config = _dsa_config()
    attention = _build_attention(config, seed=11)
    before, after = _KeyRecorder(), _KeyRecorder()
    attention.core_attention.register_forward_pre_hook(before, with_kwargs=True)
    chunk = _model_chunk_with_attention(attention)
    assert register_sglang_fp8_dsa_kv_cache_match_hooks([chunk]) == 1
    attention.core_attention.register_forward_pre_hook(after, with_kwargs=True)

    hidden_states, output_grad, mask = _attention_inputs(config, seqlen=256, seed=13)
    output, _ = attention(hidden_states, attention_mask=mask)
    output.backward(output_grad)

    assert len(before.keys) == len(after.keys) == 1
    _, prefill_read = _sampler_cache(before.keys[0], seed=17)
    assert torch.equal(_bits(after.keys[0]), _bits(prefill_read))
    assert not torch.equal(_bits(after.keys[0]), _bits(before.keys[0]))
    with pytest.raises(RuntimeError):
        register_sglang_fp8_dsa_kv_cache_match_hook(attention)


def test_gradients_match_attention_at_the_cached_value(megatron_single_rank) -> None:
    """Hooked layer grads equal: attention backward at the cached value, then identity into the key."""
    config = _dsa_config()
    hidden_states, output_grad, mask = _attention_inputs(config, seqlen=256, seed=19)

    hooked = _build_attention(config, seed=23)
    register_sglang_fp8_dsa_kv_cache_match_hook(hooked)
    hooked_output, _ = hooked(hidden_states, attention_mask=mask)
    hooked_output.backward(output_grad)
    hooked_grads = _grads(hooked, hidden_states)

    reference = _build_attention(config, seed=37)
    reference.load_state_dict(hooked.state_dict())
    reference_hidden_states = hidden_states.detach().clone().requires_grad_(True)
    captured: dict[str, torch.Tensor] = {}

    def feed_cached_value_as_leaf(module, args, kwargs):
        captured["key"] = args[1]
        captured["cached"] = sglang_fp8_dsa_kv_cache_value(args[1].detach()).requires_grad_(True)
        return (args[0], captured["cached"], *args[2:]), kwargs

    reference.core_attention.register_forward_pre_hook(feed_cached_value_as_leaf, with_kwargs=True)
    reference_output, _ = reference(reference_hidden_states, attention_mask=mask)
    reference_output.backward(output_grad)
    torch.autograd.backward(captured["key"], captured["cached"].grad)
    reference_grads = _grads(reference, reference_hidden_states)

    assert torch.equal(_bits(hooked_output.detach()), _bits(reference_output.detach()))
    assert hooked_grads.keys() == reference_grads.keys()
    for name, grad in hooked_grads.items():
        # Exact even for the input: its query and key contributions are a two-term sum.
        assert torch.equal(_bits(grad), _bits(reference_grads[name])), name


def test_activation_recompute_feeds_the_same_value(megatron_single_rank) -> None:
    config = _dsa_config()
    attention = _build_attention(config, seed=29)
    register_sglang_fp8_dsa_kv_cache_match_hook(attention)
    after = _KeyRecorder()
    attention.core_attention.register_forward_pre_hook(after, with_kwargs=True)
    hidden_states, output_grad, mask = _attention_inputs(config, seqlen=256, seed=31)

    def run_attention(hidden_states: torch.Tensor) -> torch.Tensor:
        return attention(hidden_states, attention_mask=mask)[0]

    output = tensor_parallel.checkpoint(run_attention, False, hidden_states)
    output.backward(output_grad)

    assert len(after.keys) == 2, "the checkpointed forward and its recomputation each feed attention once"
    assert torch.equal(_bits(after.keys[0]), _bits(after.keys[1]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
