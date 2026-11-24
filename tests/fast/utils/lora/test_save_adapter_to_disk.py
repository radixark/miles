import pytest
import safetensors.torch
import torch

from miles.utils.lora.utils import save_adapter_to_disk


def test_a_crash_mid_write_leaves_no_truncated_adapter(tmp_path, monkeypatch):
    # an RL --save writes its PEFT export into the committed dir, where rollout-only engines read it
    def crash(tensors, filename):
        with open(filename, "wb") as f:
            f.write(b"truncated")
        raise OSError("disk full")

    monkeypatch.setattr(safetensors.torch, "save_file", crash)
    with pytest.raises(OSError, match="disk full"):
        save_adapter_to_disk(tmp_path, {"r": 2}, {"w": torch.zeros(2)})
    assert not (tmp_path / "adapter_model.safetensors").exists()
