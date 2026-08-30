from scripts.run_qwen3_8_next import ScriptArgs, _train

import miles.utils.external_utils.command_utils as U


def _capture_train_args(monkeypatch, *, task: str) -> str:
    captured = {}
    monkeypatch.setattr(U, "execute_train", lambda **kwargs: captured.update(kwargs))
    _train(
        ScriptArgs(
            task=task,
            num_nodes=1,
            num_gpus_per_node=8,
            rollout_batch_size=2,
            n_samples_per_prompt=2,
            global_batch_size=4,
        )
    )
    return captured["train_args"]


def test_geo3k_recipe_enables_qwen3_8_next_vision_path(monkeypatch):
    train_args = _capture_train_args(monkeypatch, task="geo3k")

    assert "--prompt-data /root/datasets/geo3k_imgurl/train.parquet" in train_args
    assert "--input-key problem --multimodal-keys" in train_args
    assert "--label-key answer" in train_args
    assert "--rollout-batch-size 2" in train_args
    assert "--n-samples-per-prompt 2" in train_args
    assert "--global-batch-size 4" in train_args
    assert "--sglang-mm-attention-backend sdpa" in train_args
    assert (
        "--custom-model-provider-path "
        "miles_plugins.models.qwen3_8_next.model_provider.get_qwen3_8_next_vlm_model_provider" in train_args
    )


def test_text_recipe_keeps_qwen3_8_next_text_path(monkeypatch):
    train_args = _capture_train_args(monkeypatch, task="dapo-math")

    assert "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl" in train_args
    assert "--multimodal-keys" not in train_args
    assert "--sglang-mm-attention-backend" not in train_args
    assert "--offload-rollout-level kv_cache" not in train_args
    assert (
        "--custom-model-provider-path "
        "miles_plugins.models.qwen3_8_next.model_provider.get_qwen3_8_next_model_provider" in train_args
    )
