import sys
from types import ModuleType, SimpleNamespace

import torch
from tests.ci.ci_register import register_cpu_ci
from transformers.models.glm_ocr.configuration_glm_ocr import GlmOcrVisionConfig

from miles_plugins.models.glm5_next import vision as vision_module
from miles_plugins.models.glm5_next.vision import Glm5NextVisionModel

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])


def test_glm5_next_visual_tower_emits_one_embedding_per_merged_patch():
    config = GlmOcrVisionConfig(
        attention_bias=True,
        attention_dropout=0.0,
        depth=1,
        hidden_act="silu",
        hidden_size=8,
        in_channels=3,
        intermediate_size=16,
        num_heads=2,
        out_hidden_size=8,
        patch_size=2,
        projection_intermediate_size=16,
        rms_norm_eps=1e-5,
        spatial_merge_size=2,
        swiglu_limit=10.0,
        temporal_patch_size=2,
        _attn_implementation="sdpa",
    )
    visual = Glm5NextVisionModel(config)
    pixel_values = torch.randn(4, config.in_channels * config.temporal_patch_size * config.patch_size**2)
    image_grid_thw = torch.tensor([[1, 2, 2]])

    output = visual(pixel_values, grid_thw=image_grid_thw).pooler_output

    assert output.shape == (1, config.out_hidden_size)
    assert output.isfinite().all()


def test_glm5_next_provider_decorates_default_model(monkeypatch):
    args = SimpleNamespace(hf_checkpoint="/checkpoint")
    model = object()
    calls = []
    megatron_module = ModuleType("megatron")
    megatron_training_module = ModuleType("megatron.training")
    megatron_training_module.get_args = lambda: args
    megatron_module.training = megatron_training_module
    model_provider_module = ModuleType("miles.backends.megatron_utils.model_provider")
    model_provider_module.build_default_gpt_model = (
        lambda *provider_args, **provider_kwargs: calls.append((provider_args, provider_kwargs)) or model
    )
    monkeypatch.setitem(sys.modules, "megatron", megatron_module)
    monkeypatch.setitem(sys.modules, "megatron.training", megatron_training_module)
    monkeypatch.setitem(
        sys.modules,
        "miles.backends.megatron_utils.model_provider",
        model_provider_module,
    )
    monkeypatch.setattr(
        vision_module,
        "wire_glm5_next_visual",
        lambda wired_model, checkpoint: calls.append((wired_model, checkpoint)),
    )
    result = vision_module.glm5_next_vlm_model_provider(
        pre_process=False,
        post_process=True,
        vp_stage=2,
    )

    assert result is model
    assert calls == [
        (
            (args, "actor"),
            {"pre_process": False, "post_process": True, "vp_stage": 2},
        ),
        (model, "/checkpoint"),
    ]
