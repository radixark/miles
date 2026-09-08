from argparse import Namespace

from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import _select_bridge_checkpoint


def test_bridge_uses_direct_hf_trainer_seed_for_export_mappings(tmp_path):
    trainer_seed = tmp_path / "trainer"
    trainer_seed.mkdir()
    (trainer_seed / "model.safetensors.index.json").write_text("{}")
    rollout_schema = tmp_path / "rollout-schema"
    rollout_schema.mkdir()

    selected = _select_bridge_checkpoint(
        Namespace(load=str(trainer_seed), ref_load=str(trainer_seed), hf_checkpoint=str(rollout_schema))
    )

    assert selected == str(trainer_seed)


def test_bridge_uses_hf_reference_after_resume(tmp_path):
    training_checkpoint = tmp_path / "torch-dist"
    training_checkpoint.mkdir()
    (training_checkpoint / "latest_checkpointed_iteration.txt").write_text("1")
    trainer_seed = tmp_path / "trainer"
    trainer_seed.mkdir()
    (trainer_seed / "model.safetensors.index.json").write_text("{}")

    selected = _select_bridge_checkpoint(
        Namespace(
            load=str(training_checkpoint),
            ref_load=str(trainer_seed),
            hf_checkpoint=str(tmp_path / "rollout-schema"),
        )
    )

    assert selected == str(trainer_seed)
