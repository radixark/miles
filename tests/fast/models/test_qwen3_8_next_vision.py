from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import json
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel

import miles_plugins.models.qwen3_8_next.vision as vision


def test_build_qwen3_8_next_visual_strictly_loads_and_freezes(tmp_path, monkeypatch):
    config = Qwen3_5MoeVisionConfig(
        depth=1,
        hidden_size=32,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=16,
        num_position_embeddings=16,
    )
    reference = Qwen3_5MoeVisionModel(config)
    filename = "model-00001-of-00001.safetensors"
    prefixed_state = {
        f"model.visual.{name}": tensor.detach().contiguous() for name, tensor in reference.state_dict().items()
    }
    save_file(prefixed_state, tmp_path / filename)
    index = {"weight_map": {name: filename for name in prefixed_state}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    monkeypatch.setattr(vision, "load_hf_config", lambda _: SimpleNamespace(vision_config=config))

    actual = vision.build_qwen3_8_next_visual(
        str(tmp_path),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert not actual.training
    assert not any(parameter.requires_grad for parameter in actual.parameters())
    for name, expected in reference.state_dict().items():
        assert torch.equal(actual.state_dict()[name], expected)

    output = actual(torch.randn(16, 24), grid_thw=torch.tensor([[1, 4, 4]])).pooler_output
    assert output.shape == (4, 16)
    assert torch.isfinite(output).all()
