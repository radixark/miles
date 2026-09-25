from argparse import Namespace

import pytest

from miles.utils.arguments import _resolve_checkpoint_resume


def _args(tmp_path, **overrides):
    values = {
        "megatron_to_hf_mode": "bridge",
        "load": None,
        "ref_load": str(tmp_path / "ref"),
        "ref_ckpt_step": None,
        "hf_checkpoint": str(tmp_path / "base"),
        "start_rollout_id": None,
        "lora_adapter_path": None,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize("mode", ["bridge", "raw"])
def test_fresh_run_starts_at_zero(tmp_path, mode):
    args = _args(tmp_path, megatron_to_hf_mode=mode)

    _resolve_checkpoint_resume(args)

    assert args.start_rollout_id == 0
    assert args.lora_resume_root is None


@pytest.mark.parametrize("mode", ["bridge", "raw"])
def test_lora_checkpoint_resumes_its_run(tmp_path, mode):
    adapter = tmp_path / "run" / "iter_0000007" / "adapter"
    args = _args(tmp_path, megatron_to_hf_mode=mode, lora_adapter_path=str(adapter))

    _resolve_checkpoint_resume(args)

    assert args.lora_resume_root == str((tmp_path / "run").resolve())
    assert args.start_rollout_id is None


@pytest.mark.parametrize("adapter", ["released-adapter", "run/iter_7/adapter"])
def test_other_adapters_are_weight_only_warm_starts(tmp_path, adapter):
    args = _args(tmp_path, lora_adapter_path=str(tmp_path / adapter))

    _resolve_checkpoint_resume(args)

    assert args.lora_resume_root is None
    assert args.start_rollout_id == 0


@pytest.mark.parametrize("adapter", [None, "run/iter_0000007/adapter"], ids=["no-adapter", "lora-resume"])
def test_megatron_checkpoint_load_is_kept(tmp_path, adapter):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("7")
    adapter_path = None if adapter is None else str(tmp_path / adapter)
    args = _args(tmp_path, megatron_to_hf_mode="raw", load=str(checkpoint), lora_adapter_path=adapter_path)

    _resolve_checkpoint_resume(args)

    assert args.load == str(checkpoint)
    assert args.start_rollout_id is None
    assert (args.lora_resume_root is None) == (adapter is None)
