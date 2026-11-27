"""The TOP trainer's frozen DeepSeek-V4.1 vision tower is serving's tower, bit for bit.

V4.1 tower widths (dim 1024, 16 heads of 64, MLP 2816, 14-pixel patches, 3x3 aligner windows) at depth 2 with
seeded weights and a narrow LLM width. Serving's ``ViT`` + ``Aligner`` + ``image_span`` under the TOP contract and
``TopV41VisionTower`` over the same tensors encode the same patches to the same image-token rows, for grids whose
sides do and do not divide by the aligner window. An FP64 torch tower on the CPU bounds the error both sides would
otherwise share.

Image tokens also change two things the language model does: every row of a vision engine routes through the
VL-biased gate, image rows by the VL bias, and Engram leaves image rows' streams as they came. The trainer's gate
returns serving's ``vision_topk`` experts and weights, and the fused Engram gate with token ids gives text rows the
text program's bytes.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEPTH, DIM, HEADS, INTER, PATCH, RATIO, LLM_DIM = 2, 1024, 16, 2816, 14, 3, 512
CONFIG = {
    "vision_config": {
        "num_hidden_layers": DEPTH,
        "hidden_size": DIM,
        "num_attention_heads": HEADS,
        "intermediate_size": INTER,
        "patch_size": PATCH,
        "rope_theta": 10000,
        "downsample_ratio": RATIO,
    }
}
# (n_vit_h, n_vit_w) patch grids: square and divisible, ragged in either side, a single aligner row.
GRIDS = ((39, 39), (40, 28), (17, 44), (3, 50))


@pytest.fixture(scope="module")
def top_runtime():
    """A published TOP scheduler runtime with SGLang's process groups and batch-invariant overrides, torn down
    after the module: both are process-wide, and tests after this module must not run under them."""
    from sglang.srt.batch_invariant_ops import disable_batch_invariant_mode, enable_batch_invariant_mode
    from sglang.srt.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.runtime_context import SpawnRanks, publish, reset_context
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.true_on_policy.contracts import TRUE_ON_POLICY_V1

    publish(
        ServerArgs(model_path="none", enable_deterministic_inference=True, true_on_policy_contract=TRUE_ON_POLICY_V1),
        role="scheduler",
        ranks=SpawnRanks(world_rank=0, dp_rank=0, gpu_id=0),
    )
    init_distributed_environment(
        backend="nccl", world_size=1, rank=0, local_rank=0, distributed_init_method="tcp://127.0.0.1:29714"
    )
    initialize_model_parallel()
    enable_batch_invariant_mode()
    yield
    disable_batch_invariant_mode()
    destroy_model_parallel()
    destroy_distributed_environment()
    reset_context()


def _checkpoint_tensors():
    """Seeded tensors under the checkpoint's raw names, BF16 as the release stores them."""
    g = torch.Generator().manual_seed(0)

    def randn(*shape, scale):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    t = {
        "vision.patch_embed.proj.weight": randn(DIM, 3 * PATCH * PATCH, scale=0.04),
        "vision.patch_embed.proj.bias": randn(DIM, scale=0.1),
        "vision.norm.weight": (1 + randn(DIM, scale=0.1).float()).to(torch.bfloat16),
        "aligner.w1.weight": randn(LLM_DIM, DIM * RATIO * RATIO, scale=0.01),
        "aligner.w1.bias": randn(LLM_DIM, scale=0.1),
        "aligner.w2.weight": randn(LLM_DIM, LLM_DIM, scale=0.04),
        "aligner.w2.bias": randn(LLM_DIM, scale=0.1),
        "image_start": randn(LLM_DIM, scale=1.0),
        "image_end": randn(LLM_DIM, scale=1.0),
        "image_newline": randn(LLM_DIM, scale=1.0),
    }
    for i in range(DEPTH):
        p = f"vision.blocks.{i}."
        t |= {
            p + "norm1.weight": (1 + randn(DIM, scale=0.1).float()).to(torch.bfloat16),
            p + "norm2.weight": (1 + randn(DIM, scale=0.1).float()).to(torch.bfloat16),
            p + "attn.wqkv.weight": randn(3 * DIM, DIM, scale=0.03),
            p + "attn.wqkv.bias": randn(3 * DIM, scale=0.1),
            p + "attn.wo.weight": randn(DIM, DIM, scale=0.03),
            p + "attn.wo.bias": randn(DIM, scale=0.1),
            p + "mlp.w1.weight": randn(2 * INTER, DIM, scale=0.03),
            p + "mlp.w2.weight": randn(DIM, INTER, scale=0.02),
        }
    return t


@pytest.fixture(scope="module")
def towers(top_runtime):
    """(serving's encode, the TOP trainer tower, the raw tensors) over one set of seeded tensors."""
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
    from sglang.srt.models.deepseek_v41_vit import Aligner, ViT, image_span

    from miles_plugins.top.dsv41_vision import TopV41VisionTower

    tensors = _checkpoint_tensors()
    args = SimpleNamespace(
        vision_n_layers=DEPTH,
        vision_dim=DIM,
        vision_n_heads=HEADS,
        vision_inter_dim=INTER,
        vision_patch_size=PATCH,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=RATIO,
        dim=LLM_DIM,
    )
    # Built as serving's loader builds it: BF16 default dtype on the device, parameters filled in place.
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            vit, aligner = ViT(args).eval(), Aligner(args).eval()
    finally:
        torch.set_default_dtype(torch.float32)
    params = {"vision." + n: p for n, p in vit.named_parameters()} | {
        "aligner." + n: p for n, p in aligner.named_parameters()
    }
    with torch.no_grad():
        for name, value in tensors.items():
            if name.startswith("image_"):
                continue
            served = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(name)
            params.pop(served).copy_(value)
    assert not params, sorted(params)
    boundary = {k: tensors[f"image_{k}"].cuda() for k in ("start", "newline", "end")}

    @torch.no_grad()
    def serve(patches, h, w):
        return image_span(aligner(vit(patches, h, w), h, w), n_vit_h=h, n_vit_w=w, ratio=RATIO, **boundary)

    trainer_tensors = {
        name: value.cuda().float() if name.endswith(("norm.weight", "norm1.weight", "norm2.weight")) else value.cuda()
        for name, value in tensors.items()
    }
    return serve, TopV41VisionTower(trainer_tensors, CONFIG), tensors


def _patches(i):
    h, w = GRIDS[i]
    g = torch.Generator().manual_seed(100 + i)
    return torch.randn(h * w, 3, PATCH, PATCH, generator=g).clamp(-1, 1).to(torch.bfloat16).cuda(), h, w


@pytest.mark.parametrize("i", range(len(GRIDS)), ids=[f"{h}x{w}" for h, w in GRIDS])
def test_rows_equal_serving(towers, i):
    serve, trainer, _ = towers
    patches, h, w = _patches(i)
    served = serve(patches, h, w)
    trained = trainer(patches, n_vit_h=h, n_vit_w=w)
    lh, lw = -(-h // RATIO), -(-w // RATIO)
    assert served.shape == trained.shape == (lh * (lw + 1) + 2, LLM_DIM)
    assert served.dtype == trained.dtype == torch.bfloat16
    assert torch.equal(served, trained)
    # An image's rows do not depend on what either side encoded before it.
    other, oh, ow = _patches((i + 1) % len(GRIDS))
    serve(other, oh, ow)
    trainer(other, n_vit_h=oh, n_vit_w=ow)
    assert torch.equal(served, serve(patches, h, w))
    assert torch.equal(trained, trainer(patches, n_vit_h=h, n_vit_w=w))


def _reference_span(tensors, patches, h, w):
    """The tower's math in FP64 on the CPU: torch GEMMs, the release's half-split rotary, torch attention."""
    from sglang.srt.models.deepseek_v41_vit import get_vision_cos_sin, image_span

    t = {k: v.double() for k, v in tensors.items()}

    def lin(name, x):
        return F.linear(x, t[name + ".weight"], t.get(name + ".bias"))

    def norm(name, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * t[name + ".weight"]

    def rot(x, cos, sin):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    x = lin("vision.patch_embed.proj", patches.cpu().double().flatten(1))
    cos, sin = (v.double() for v in get_vision_cos_sin(h, w, DIM // HEADS // 2, 10000.0))
    for i in range(DEPTH):
        p = f"vision.blocks.{i}."
        q, k, v = (
            s.reshape(-1, HEADS, DIM // HEADS) for s in lin(p + "attn.wqkv", norm(p + "norm1", x)).split(DIM, -1)
        )
        q, k = rot(q, cos, sin), rot(k, cos, sin)
        out = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)).transpose(0, 1)
        x = x + lin(p + "attn.wo", out.reshape(x.shape[0], -1))
        gate, up = lin(p + "mlp.w1", norm(p + "norm2", x)).chunk(2, -1)
        x = x + lin(p + "mlp.w2", F.silu(gate) * up)
    x = norm("vision.norm", x).view(h, w, -1).permute(2, 0, 1)
    x = F.pad(x, (0, -w % RATIO, 0, -h % RATIO))
    x = F.unfold(x.unsqueeze(0), RATIO, stride=RATIO).squeeze(0).transpose(0, 1)
    x = lin("aligner.w2", F.gelu(lin("aligner.w1", x)))
    return image_span(
        x.to(torch.bfloat16).double(),
        n_vit_h=h,
        n_vit_w=w,
        ratio=RATIO,
        start=t["image_start"],
        newline=t["image_newline"],
        end=t["image_end"],
    )


def test_rows_are_the_towers_rows_to_bf16_noise(towers):
    """Both sides share the program, so their equality cannot see an error in it: an FP64 tower must agree."""
    _, trainer, tensors = towers
    patches, h, w = _patches(1)
    rows = trainer(patches, n_vit_h=h, n_vit_w=w).double().cpu()
    expected = _reference_span(tensors, patches, h, w)
    relative = float((rows - expected).square().mean().sqrt() / expected.square().mean().sqrt())
    print(f"relative RMS to the FP64 tower: {relative:.5f}")
    # Arbitrary bound above the BF16 noise measured here; a wrong program is off by order 1.
    assert relative < 0.05, relative


# V4.1's routing width: 384 experts, 6 per token, and its image token id.
EXPERTS, TOPK, IMAGE_TOKEN_ID = 384, 6, 129264


def _routing_case(rows=257, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = (torch.randn(rows, EXPERTS, generator=g) * 2).cuda()
    bias = (torch.randn(EXPERTS, generator=g) * 0.5).cuda()
    bias_vl = (torch.randn(EXPERTS, generator=g) * 0.5).cuda()
    input_ids = torch.randint(0, 1000, (rows,), generator=g)
    input_ids[40:120] = IMAGE_TOKEN_ID
    input_ids[200:203] = IMAGE_TOKEN_ID
    return logits, bias, bias_vl, input_ids.cuda()


def test_trainer_vision_gate_is_servings_vision_topk(top_runtime):
    """The trainer's VL-biased gate returns serving's vision_topk weights and experts for every row, and image rows
    rank by the VL bias: forcing their bias changes their experts and no text row's."""
    from sglang.srt.layers.moe.topk import TopKConfig
    from sglang.srt.multimodal.dsv41.vl_routing import vision_topk

    from miles_plugins.top.dsv41_moe import vision_gate

    logits, bias, bias_vl, input_ids = _routing_case()
    scale = 1.5
    config = TopKConfig(
        top_k=TOPK,
        renormalize=True,
        correction_bias=bias,
        routed_scaling_factor=scale,
        scoring_func="sqrtsoftplus",
        sqrtsoftplus_log1p=True,
        allow_routed_experts_capture=False,
        layer_id=0,
    )
    moe = SimpleNamespace(
        topk=SimpleNamespace(topk_config=config, routing_inputs=lambda: (None, None)),
        gate=SimpleNamespace(e_score_correction_bias=bias, e_score_correction_bias_vl=bias_vl),
        config=SimpleNamespace(image_token_id=IMAGE_TOKEN_ID),
        layer_id=0,
    )
    served = vision_topk(moe, logits, input_ids)

    def trained(bias_vl):
        return vision_gate(
            logits,
            bias=bias,
            bias_vl=bias_vl,
            input_ids=input_ids,
            image_token_id=IMAGE_TOKEN_ID,
            topk=TOPK,
            routed_scaling_factor=scale,
        )

    weights, ids = trained(bias_vl)
    assert torch.equal(weights, served.topk_weights.float())
    assert torch.equal(ids, served.topk_ids.to(torch.int32))
    image = input_ids == IMAGE_TOKEN_ID
    forced = torch.full_like(bias_vl, -100.0)
    forced[:TOPK] = 100.0
    forced_weights, forced_ids = trained(forced)
    assert torch.equal(forced_ids[~image], ids[~image])
    assert torch.equal(
        forced_ids[image].sort(-1).values, torch.arange(TOPK, device="cuda").expand(int(image.sum()), -1)
    )


def test_engram_gate_keeps_image_rows_and_text_rows_are_the_text_programs(top_runtime):
    """With token ids, the fused gate leaves image rows' streams as they came and gives every other row the bytes
    the text program gives it; the trainer's gradient passes image rows straight through."""
    from sglang.srt.true_on_policy.dsv41_engram import gate

    from miles_plugins.top.dsv41_engram import top_engram_gate

    g = torch.Generator().manual_seed(1)
    rows, hc, dim = 97, 4, 512
    x = torch.randn(rows, hc, dim, generator=g).to(torch.bfloat16).cuda()
    kv = torch.randn(rows, (hc + 1) * dim, generator=g).to(torch.bfloat16).cuda()
    q, k = (torch.randn(hc, dim, generator=g).to(torch.bfloat16).cuda() for _ in range(2))
    input_ids = torch.randint(0, 1000, (rows,), generator=g).cuda()
    input_ids[10:30] = IMAGE_TOKEN_ID
    image = input_ids == IMAGE_TOKEN_ID

    text_program = gate(x, kv, q, k, 1e-6, 1e-6)
    with_ids = gate(x, kv, q, k, 1e-6, 1e-6, input_ids, IMAGE_TOKEN_ID)
    assert torch.equal(with_ids[image], x[image])
    assert torch.equal(with_ids[~image], text_program[~image])

    xg = x.clone().requires_grad_(True)
    out = top_engram_gate(xg, kv, q.float(), k.float(), 1e-6, 1e-6, input_ids, IMAGE_TOKEN_ID)
    assert torch.equal(out.detach(), with_ids)
    out.float().square().sum().backward()
    assert torch.equal(xg.grad[image], (2 * x[image].float()).to(xg.grad.dtype))


def test_engram_hash_blocks_only_image_tokens(top_runtime):
    """Hashing with the image token id leaves every text row's ids as the text program hashes them; a row whose
    n-gram reaches back over an image token changes. The hash kernel reads an id at or above its pad shift as an
    image, so the trainer must pass serving's shift: the kernel's default 0 reads every token as an image."""
    from miles_plugins.top.dsv41_engram import packed_hash_ids

    g = torch.Generator().manual_seed(2)
    layers, n, heads = 2, 4, 4
    hasher = SimpleNamespace(
        max_ngram_size=n,
        token_map=torch.arange(IMAGE_TOKEN_ID + 16, dtype=torch.int64).cuda() % 50_000,
        multipliers=torch.randint(1, 1 << 20, (layers, n), generator=g).cuda(),
        primes=torch.randint(1 << 10, 1 << 16, (layers, n - 1, heads), generator=g).cuda(),
        offsets=torch.randint(0, 1 << 12, (layers, (n - 1) * heads), generator=g).cuda(),
        pad_id=2,
    )
    lengths = [37, 20]
    text = torch.randint(3, 1000, (sum(lengths),), generator=g).cuda()

    def hashed(ids, image_token_id):
        return packed_hash_ids(hasher, ids, lengths, 0, image_token_id=image_token_id)

    assert torch.equal(hashed(text, IMAGE_TOKEN_ID), hashed(text, None))
    with_image = text.clone()
    with_image[10] = IMAGE_TOKEN_ID
    changed = (hashed(with_image, IMAGE_TOKEN_ID) != hashed(text, None)).any(-1).nonzero().flatten().tolist()
    # Each row hashes its n-grams of 2 to n tokens ending at itself, so rows 10 to 10 + n - 1 reach the image.
    assert changed == list(range(10, 10 + n)), changed
