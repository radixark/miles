from pathlib import Path

import pytest
import torch
from tests.fast.backends.test_torchtitan import _checkpoint_dir, _config_args


def test_torchtitan_backend_options_reach_real_trainer_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real trainer config preserves backend optimizer, batch, precision and seed options."""
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import build_trainer_config

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 128)
    args = _config_args(
        seq_length=2048,
        global_batch_size=512,
        dp_replicate_size=2,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        expert_model_parallel_size=8,
        micro_batch_size=2,
        clip_grad=0.75,
        lr=0.002,
        adam_beta1=0.7,
        adam_beta2=0.8,
        adam_eps=0.00001,
        weight_decay=0.125,
        seed=37,
        gradient_checkpointing=True,
        save=str(tmp_path / "save"),
    )

    config = build_trainer_config(
        args,
        hf_assets_path=_checkpoint_dir(tmp_path / "hf", tie_word_embeddings=False),
        lr_total_steps=17,
        dump_subdir="actor",
    )

    assert config.training.seq_len == 2048
    assert config.training.local_batch_size == 16
    assert config.training.global_batch_size == 256
    assert config.parallelism.data_parallel_replicate_degree == 2
    assert config.parallelism.tensor_parallel_degree == 2
    assert config.parallelism.pipeline_parallel_degree == 4
    assert config.parallelism.context_parallel_degree == 1
    assert config.parallelism.expert_parallel_degree == 8
    assert config.training.steps == 17
    assert config.training.max_norm == 0.75
    assert config.optimizer.param_groups[0].optimizer_name == "AdamW"
    assert config.optimizer.param_groups[0].optimizer_kwargs == {
        "lr": 0.002,
        "betas": (0.7, 0.8),
        "eps": 0.00001,
        "weight_decay": 0.125,
    }
    assert config.activation_checkpoint is not None
    assert config.debug.seed == 37
    assert config.dump_folder == str(tmp_path / "save" / "torchtitan" / "actor")


def test_torchtitan_rejects_backend_optimizer_before_loading_assets() -> None:
    """An unsupported backend optimizer fails before checkpoint files are inspected."""
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import build_trainer_config

    with pytest.raises(ValueError, match="supports --optimizer adam"):
        build_trainer_config(
            _config_args(optimizer="sgd"), hf_assets_path="/missing", lr_total_steps=1, dump_subdir="actor"
        )
