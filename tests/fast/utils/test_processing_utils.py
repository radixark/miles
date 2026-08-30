import json

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

from miles.utils.processing_utils import extract_multimodal_train_inputs, load_processor
from miles_plugins.models.glm5_next import processor as glm5_next_processor


def test_extract_multimodal_train_inputs_drops_qwen3_vl_token_metadata():
    pixel_values = object()
    image_grid_thw = object()
    processor_output = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
        "mm_token_type_ids": [[0, 1, 0]],
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

    assert extract_multimodal_train_inputs(processor_output) == {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    assert (
        extract_multimodal_train_inputs(
            {
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]],
                "mm_token_type_ids": [[0, 1, 0]],
            }
        )
        is None
    )


def test_load_processor_dispatches_local_glm5_next_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "glm5_next"}))
    expected = object()
    calls = []

    def fake_load(name_or_path, **kwargs):
        calls.append((name_or_path, kwargs))
        return expected

    monkeypatch.setattr(glm5_next_processor, "load_glm5_next_processor", fake_load)

    assert load_processor(str(tmp_path), revision="test-revision") is expected
    assert calls == [(str(tmp_path), {"revision": "test-revision"})]
