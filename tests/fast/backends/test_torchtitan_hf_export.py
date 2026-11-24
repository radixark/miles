import json
from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

pytest.importorskip("torchtitan")
safetensors_torch = pytest.importorskip("safetensors.torch")


def _write_source_checkpoint(root) -> None:
    root.mkdir()
    safetensors_torch.save_file(
        {"trained": torch.zeros(2, 3, dtype=torch.bfloat16), "frozen": torch.arange(4, dtype=torch.bfloat16)},
        str(root / "model-00001-of-00001.safetensors"),
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "trained": "model-00001-of-00001.safetensors",
                    "frozen": "model-00001-of-00001.safetensors",
                }
            }
        )
    )
    safetensors_torch.save_file({"mtp.proj": torch.ones(3)}, str(root / "mtp.safetensors"))
    (root / "config.json").write_text("{}")
    (root / "tiktoken").mkdir()
    (root / "tiktoken" / "tokenizer.model").write_text("vocab")


def test_the_export_is_the_source_checkpoint_with_the_trained_tensors_replaced(tmp_path, monkeypatch):
    from miles.backends.torchtitan_utils import hf_export

    source, out = tmp_path / "source", tmp_path / "out"
    _write_source_checkpoint(source)
    monkeypatch.setattr(hf_export, "hf_weights", lambda trainer, **_: iter([("trained", torch.full((2, 3), 7.0))]))
    monkeypatch.setattr(hf_export, "dist", SimpleNamespace(get_rank=lambda: 0, barrier=lambda: None))

    hf_export.save_hf_checkpoint(None, hf_checkpoint=str(source), path=str(out))

    weight_map = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(weight_map) == {"trained", "frozen"}
    tensors = {}
    for shard in set(weight_map.values()):
        tensors |= safetensors_torch.load_file(str(out / shard))
    assert tensors["trained"].dtype == torch.bfloat16 and torch.equal(
        tensors["trained"], torch.full((2, 3), 7.0).bfloat16()
    )
    assert torch.equal(tensors["frozen"], torch.arange(4, dtype=torch.bfloat16))
    assert (out / "mtp.safetensors").read_bytes() == (source / "mtp.safetensors").read_bytes()
    assert (out / "tiktoken" / "tokenizer.model").read_text() == "vocab"
    assert not (out / "model-00001-of-00001.safetensors").exists()
