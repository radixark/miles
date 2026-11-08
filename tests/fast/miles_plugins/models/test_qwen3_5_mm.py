from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig as HFVisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig
from miles_plugins.models.qwen3_5.mm_towers import check_multimodal_setup
from miles_plugins.models.qwen3_5.vision import Qwen3_5VisionConfig, Qwen3_5VisionTower, _PatchEmbed, _rot_pos_ids

_VISION_CONFIG = dict(
    depth=2,
    hidden_size=64,
    num_heads=4,
    intermediate_size=128,
    hidden_act="gelu_pytorch_tanh",
    in_channels=3,
    patch_size=4,
    temporal_patch_size=2,
    spatial_merge_size=2,
    out_hidden_size=32,
    num_position_embeddings=36,
)


def test_tower_loads_the_hf_checkpoint_layout():
    hf = Qwen3_5VisionModel(HFVisionConfig(**_VISION_CONFIG))
    tower = Qwen3_5VisionTower(Qwen3_5VisionConfig(**_VISION_CONFIG))

    assert {k: v.shape for k, v in tower.state_dict().items()} == {k: v.shape for k, v in hf.state_dict().items()}


def test_config_rejects_deepstack_mergers():
    with pytest.raises(ValueError, match="deepstack"):
        Qwen3_5VisionConfig.from_hf({**_VISION_CONFIG, "deepstack_visual_indexes": [3]})


def test_patch_embed_gemm_matches_conv3d():
    torch.manual_seed(0)
    embed = _PatchEmbed(Qwen3_5VisionConfig(**_VISION_CONFIG)).double()
    pixel_values = torch.randn(10, 3 * 2 * 4 * 4, dtype=torch.float64)

    expected = embed.proj(pixel_values.view(-1, 3, 2, 4, 4)).view(-1, 64)

    torch.testing.assert_close(embed(pixel_values), expected)


def test_rotary_positions_follow_the_spatial_merge_block_order():
    # A 2x4 grid arrives as two 2x2 merge blocks, each read row by row.
    assert _rot_pos_ids(2, 4, 2).tolist() == [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]]


def _rope_config(**overrides):
    return SimpleNamespace(
        **{"apply_rope_fusion": False, "rotary_interleaved": False, "fused_single_qkv_rope": False, **overrides}
    )


def _engines(*, num_gpus_per_engine, **base_args):
    group = ServerGroupConfig(
        worker_type=WorkerType.REGULAR,
        num_gpus_per_engine=num_gpus_per_engine,
        needs_offload=False,
        overrides={"model_path": "/ckpt"},
    )
    actor = ModelConfig(name="actor", model_path=None, server_groups=[group], update_weights=True)
    return SglangConfig(models=[actor], base_args=base_args)


def test_rope_fusion_without_the_fused_mrope_kernel_is_rejected(monkeypatch):
    """Megatron would then hand per-token frequencies to a kernel that reads a position table per sequence."""
    import megatron.core.fusions.fused_mrope as fused_mrope

    monkeypatch.setattr(fused_mrope, "is_fused_mrope_available", lambda: False)
    with pytest.raises(ValueError, match="--no-rope-fusion"):
        check_multimodal_setup(_rope_config(apply_rope_fusion=True), _engines(num_gpus_per_engine=1))

    monkeypatch.setattr(fused_mrope, "is_fused_mrope_available", lambda: True)
    check_multimodal_setup(_rope_config(apply_rope_fusion=True), _engines(num_gpus_per_engine=1))
    with pytest.raises(ValueError, match="--no-rope-fusion"):
        check_multimodal_setup(
            _rope_config(apply_rope_fusion=True, rotary_interleaved=True), _engines(num_gpus_per_engine=1)
        )


def test_tensor_parallel_engines_without_the_dp_encoder_are_warned_about(caplog):
    """An engine that shards the vision tower across TP ranks serves other rows than the trainer's whole tower; the
    launcher sizes TP from the group's GPUs, so the check reads the group, not the run-wide default."""
    with caplog.at_level("WARNING"):
        check_multimodal_setup(_rope_config(), _engines(num_gpus_per_engine=2, mm_enable_dp_encoder=None))
    assert "--sglang-mm-enable-dp-encoder" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        check_multimodal_setup(_rope_config(), _engines(num_gpus_per_engine=2, mm_enable_dp_encoder=True))
        check_multimodal_setup(_rope_config(), _engines(num_gpus_per_engine=1, mm_enable_dp_encoder=None))
    assert "dp-encoder" not in caplog.text
