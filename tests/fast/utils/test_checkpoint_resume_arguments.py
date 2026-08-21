from argparse import Namespace

from miles.utils.arguments import _lora_checkpoint_root, _resolve_checkpoint_resume
from miles.utils.lora import start_rollout_id_from_checkpoint


def _args(tmp_path, **overrides):
    values = {
        "load": None,
        "ref_load": None,
        "hf_checkpoint": str(tmp_path / "base"),
        "start_rollout_id": None,
        "lora_adapter_path": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_fresh_bridge_run_starts_at_zero(tmp_path):
    args = _args(tmp_path)
    _resolve_checkpoint_resume(args)
    assert args.load == args.hf_checkpoint
    assert args.start_rollout_id == 0
    assert args.rollout_data_load is None


def test_native_lora_checkpoint_defers_start_to_actor_and_sets_data_root(tmp_path):
    adapter = tmp_path / "run" / "iter_0000007" / "adapter"
    adapter.mkdir(parents=True)
    args = _args(tmp_path, lora_adapter_path=str(adapter), rollout_global_dataset=True)

    _resolve_checkpoint_resume(args)

    assert args.start_rollout_id is None
    assert args.rollout_data_load == str(tmp_path / "run")
    assert args.lora_training_state_resume_enabled


def test_bridge_training_checkpoint_preserves_automatic_resume(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("7")
    args = _args(tmp_path, load=str(checkpoint))

    _resolve_checkpoint_resume(args)

    assert args.load == str(checkpoint)
    assert args.start_rollout_id is None


def test_explicit_start_rollout_id_is_not_overwritten(tmp_path):
    args = _args(tmp_path, start_rollout_id=5)
    _resolve_checkpoint_resume(args)
    assert args.start_rollout_id == 5


def test_non_checkpoint_adapter_path_is_weight_only_with_global_dataset(tmp_path):
    adapter = tmp_path / "released-adapter"
    adapter.mkdir()
    args = _args(tmp_path, lora_adapter_path=str(adapter), rollout_global_dataset=True)

    _resolve_checkpoint_resume(args)

    assert _lora_checkpoint_root(str(adapter)) is None
    assert args.rollout_data_load is None
    assert not args.lora_training_state_resume_enabled


def test_weight_only_adapter_starts_a_new_run():
    args = Namespace(lora_adapter_path="/adapter", lora_training_state_loaded=False)
    assert start_rollout_id_from_checkpoint(args, loaded_rollout_id=0) == 0


def test_native_adapter_training_state_advances_rollout():
    args = Namespace(lora_adapter_path="/adapter", lora_training_state_loaded=True)
    assert start_rollout_id_from_checkpoint(args, loaded_rollout_id=7) == 8
