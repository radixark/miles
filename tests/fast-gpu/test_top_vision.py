"""The TOP trainer's frozen vision tower is serving's tower, bit for bit, and neither depends on its batch.

Qwen3.5 tower widths (hidden 1152, 16 heads of 72, MLP 4304) at depth 2 with seeded weights. Serving's
``Qwen3VLMoeVisionModel`` under the TOP contract and ``TopVisionTower`` over the same parameters encode the same
patches: every batch gives the same rows on both sides, and the rows of an image are those it gets alone. Seven
images cross the count where serving switches to its vectorized position interpolation when TOP does not pin the
per-image loop. An FP64 torch tower on the CPU bounds the error both sides would otherwise share.

Image tokens also change what full attention rotates by: [3, T] mRoPE positions instead of a token's index. The
trainer's rotary forward is serving's object; its backward reference must select each rotary lane's axis as that
object does, which the rotary tests check against the served rotation and against FP64. Qwen3.8-Flash-Next's QSA
indexer rotates with its own kernels, which read each lane at its axis' position: the last test checks every lane
against the same kernel at that position on every axis, the text program.
"""

import pytest
import torch
import torch.nn.functional as F
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

VISION = dict(
    depth=2,
    hidden_size=1152,
    num_heads=16,
    intermediate_size=4304,
    hidden_act="gelu_pytorch_tanh",
    in_channels=3,
    patch_size=16,
    temporal_patch_size=2,
    spatial_merge_size=2,
    out_hidden_size=2048,
    num_position_embeddings=2304,
)
# (t, h, w) patch grids; the last is two frame groups, each its own attention sequence.
GRIDS = ((1, 26, 38), (1, 6, 36), (1, 6, 6), (1, 24, 36), (1, 38, 26), (1, 8, 8), (2, 14, 20))
PATCH_DIM = 3 * 2 * 16 * 16


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
        backend="nccl", world_size=1, rank=0, local_rank=0, distributed_init_method="tcp://127.0.0.1:29713"
    )
    initialize_model_parallel()
    enable_batch_invariant_mode()
    yield
    disable_batch_invariant_mode()
    destroy_model_parallel()
    destroy_distributed_environment()
    reset_context()


@pytest.fixture(scope="module")
def towers(top_runtime):
    """(serving tower, TOP trainer tower, FP64 reference tower) over one set of seeded parameters."""
    from sglang.srt.configs.qwen3_vl import Qwen3VLVisionConfig
    from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel

    from miles_plugins.models.qwen3_5.vision import Qwen3_5VisionConfig, Qwen3_5VisionTower
    from miles_plugins.top.vision import TopVisionTower

    torch.manual_seed(0)
    # On the CPU, where the batch-invariant GEMM overrides (CUDA only) do not reach it.
    reference = Qwen3_5VisionTower(Qwen3_5VisionConfig.from_hf(VISION)).eval().requires_grad_(False)
    for name, param in reference.named_parameters():
        if name.endswith("bias") or "norm" in name:
            torch.nn.init.normal_(param, mean=1.0 if name.endswith("weight") else 0.0, std=0.1)
    state = {name: value.to(torch.bfloat16) for name, value in reference.state_dict().items()}
    reference.double().load_state_dict({name: value.double() for name, value in state.items()})

    trainer = Qwen3_5VisionTower(Qwen3_5VisionConfig.from_hf(VISION)).cuda().to(torch.bfloat16).eval()
    trainer.load_state_dict(state)
    trainer.requires_grad_(False)
    # Built as serving's loader builds it: BF16 default dtype on the device, parameters filled in place, no cast.
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            serving = Qwen3VLMoeVisionModel(Qwen3VLVisionConfig(**VISION, deepstack_visual_indexes=[])).eval()
    finally:
        torch.set_default_dtype(torch.float32)
    params = dict(serving.named_parameters())
    assert len(params) == len(state)
    with torch.no_grad():
        for name, value in state.items():
            params[name.replace(".attn.qkv.", ".attn.qkv_proj.")].copy_(value)
    # The serving refusals read the checkpoint's config; they have their own tests.
    serving.top_admitted = True
    return serving, TopVisionTower(trainer), reference


def _patches(order):
    pixels, grids = [], []
    for i in order:
        t, h, w = GRIDS[i]
        g = torch.Generator().manual_seed(100 + i)
        pixels.append(torch.randn(t * h * w, PATCH_DIM, generator=g))
        grids.append([t, h, w])
    return torch.cat(pixels).cuda(), torch.tensor(grids, device="cuda")


def _reference_rows(reference, pixels, grids):
    """The upstream tower's parameters in FP64 on CPU: torch GEMMs, eager rotary, torch attention per sequence."""
    import sglang.srt.layers.rotary_embedding as rotary
    from sglang.srt.layers.rotary_embedding.utils import apply_rotary_pos_emb_native_eager

    import miles_plugins.models.qwen3_5.vision as upstream

    def sdpa_varlen(q, k, v, *, cu_seqlens_q, **_):
        bounds = cu_seqlens_q.tolist()
        return torch.cat(
            [
                F.scaled_dot_product_attention(
                    q[a:b].transpose(0, 1), k[a:b].transpose(0, 1), v[a:b].transpose(0, 1)
                ).transpose(0, 1)
                for a, b in zip(bounds, bounds[1:], strict=False)
            ]
        )

    patched = pytest.MonkeyPatch()
    patched.setattr(upstream, "_flash_attn_varlen_func", lambda: sdpa_varlen)
    patched.setattr(rotary, "apply_rotary_pos_emb", apply_rotary_pos_emb_native_eager)
    try:
        return reference(pixels.cpu(), grids.cpu())
    finally:
        patched.undo()


@pytest.mark.parametrize(
    "order", [(0,), (2,), (6,), (0, 1), (1, 0), (0, 0, 0), tuple(range(7)), (6, 4, 2, 0, 5, 3, 1)], ids=str
)
def test_rows_equal_serving_and_do_not_depend_on_the_batch(towers, order):
    serving, trainer, _ = towers
    pixels, grids = _patches(order)
    with torch.no_grad():
        served = serving(pixels, grid_thw=grids)
    trained = trainer(pixels, grids)
    assert served.dtype == trained.dtype == torch.bfloat16
    assert torch.equal(served, trained)
    alone = torch.cat([trainer(*_patches((i,))) for i in order])
    assert torch.equal(trained, alone)


def test_rows_are_the_towers_rows_to_bf16_noise(towers):
    """Both sides share the program, so their equality cannot see an error in it: an FP64 tower must agree."""
    _, trainer, reference = towers
    pixels, grids = _patches(tuple(range(7)))
    rows, expected = trainer(pixels, grids).double().cpu(), _reference_rows(reference, pixels, grids)
    relative = float((rows - expected).square().mean().sqrt() / expected.square().mean().sqrt())
    print(f"relative RMS to the FP64 tower: {relative:.5f}")
    # Arbitrary bound above the BF16 noise measured here; a wrong program is off by order 1.
    assert relative < 0.05, relative


def test_the_unbound_tower_cannot_run_under_top(towers):
    """Upstream's tower fuses the bias into its GEMMs, which TOP's BI GEMM refuses: it must fail, not differ."""
    _, trainer, _ = towers
    with pytest.raises(NotImplementedError, match="fused bias"):
        trainer.tower(*_patches((2,)))


# Qwen3.5's full attention: 256-wide heads, the first quarter rotated, three interleaved mRoPE axes.
HEAD_DIM, ROTARY_DIM, MROPE_SECTION = 256, 64, [11, 11, 10]


@pytest.fixture(scope="module")
def mrope(top_runtime):
    from sglang.srt.layers.rotary_embedding import get_rope

    with torch.device("cuda"):
        return get_rope(
            head_size=HEAD_DIM,
            rotary_dim=HEAD_DIM,
            max_position=4096,
            base=10000000,
            rope_scaling={"rope_type": "default", "mrope_section": MROPE_SECTION, "mrope_interleaved": True},
            partial_rotary_factor=ROTARY_DIM / HEAD_DIM,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )


def _image_positions(rows):
    """[3, rows]: text rows count on every axis; from row 5 on, a 6-column grid at one temporal position."""
    index = torch.arange(rows, device="cuda")
    image = index >= 5
    grid = index - 5
    return torch.stack(
        (
            torch.where(image, 5, index),
            torch.where(image, 5 + grid // 6, index),
            torch.where(image, 5 + grid % 6, index),
        )
    )


def test_the_rotary_reference_is_the_served_mrope_rotation(mrope):
    """The reference gives the trainer's gradient, so it must rotate every lane by the axis serving rotates it by:
    a wrong axis map would still pass every text sample, whose three axes are equal."""
    from miles_plugins.top.gdn_attention import rotary_reference

    assert mrope.rotary_dim == ROTARY_DIM and mrope.mrope_section == MROPE_SECTION
    rows, heads = 41, 3
    generator = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(rows, heads * HEAD_DIM, generator=generator, device="cuda").to(torch.bfloat16)
    positions = _image_positions(rows)
    served, _ = mrope(positions, q.clone(), q.clone())
    reference = rotary_reference(q, positions, mrope)
    # The served rotation rounds each product to BF16; the reference stays FP32.
    assert (served.float() - reference).abs().max() < 2**-5
    assert torch.equal(served[:5], mrope(torch.arange(5, device="cuda"), q[:5].clone(), q[:5].clone())[0])
    text = torch.arange(rows, device="cuda")
    assert torch.equal(rotary_reference(q, text.expand(3, -1), mrope), rotary_reference(q, text, mrope))
    assert not torch.equal(reference[5:], rotary_reference(q, text, mrope)[5:])


def test_qsa_rotates_each_lane_as_the_text_program_at_its_axis(mrope):
    """QSA's index queries and compressed keys rotate lane by lane at the position of the lane's MRoPE axis. A
    lane's bytes must be those the kernel gives when every axis holds that position (a text token), so an image
    position changes which position a lane reads and nothing else."""
    from sglang.srt.true_on_policy import qsa

    rows, heads, dim = 37, 3, 128
    generator = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(rows, heads, dim, generator=generator, device="cuda").to(torch.bfloat16)
    weight = (0.1 * torch.randn(dim, generator=generator, device="cuda")).to(torch.bfloat16)
    positions = qsa.rope_positions(_image_positions(rows))
    axes = qsa.lane_axes(mrope, "cuda")
    assert axes.tolist() == mrope.axis_map.tolist() and set(axes.tolist()) == {0, 1, 2}
    params = dict(weight=weight, eps=1e-6, rotary_dim=ROTARY_DIM, base=10000000, axes=axes)
    lanes = torch.arange(dim, device="cuda")
    half = ROTARY_DIM // 2
    # The axis of every dim: a rotated dim's lane is its index in either half; a dim past the rotary part has none.
    dim_axis = torch.where(lanes < ROTARY_DIM, axes[(lanes % half).clamp_max(half - 1)].long(), -1)

    queries = qsa.norm_rope(x, positions, **params)
    locs = torch.arange(rows // 4 * 4, device="cuda").reshape(-1, 4)
    keys = qsa.compress_norm_rope(x[:, 0], locs, positions[locs[:, 0]], **params)
    for axis in range(3):
        text = positions[:, axis : axis + 1].expand(-1, 3).contiguous()
        on_axis = dim_axis == axis
        assert torch.equal(queries[..., on_axis], qsa.norm_rope(x, text, **params)[..., on_axis])
        text_keys = qsa.compress_norm_rope(x[:, 0], locs, text[locs[:, 0]], **params)
        assert torch.equal(keys[:, on_axis], text_keys[:, on_axis])
    unrotated = dim_axis < 0
    assert torch.equal(queries[..., unrotated], qsa.norm_rope(x, positions * 0, **params)[..., unrotated])
    # The image rows' axes differ, so a single-position program would not give these rows.
    assert not torch.equal(queries[5:], qsa.norm_rope(x, positions[:, :1].expand(-1, 3).contiguous(), **params)[5:])
