import json
from argparse import Namespace
from contextlib import nullcontext
from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import torch

from miles.backends.megatron_utils.update_weight import hf_weight_iterator_bridge as bridge_iterator
from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import (
    HfWeightIteratorBridge,
    _load_checkpoint_name_remap,
    _process_conversion_tasks,
    _select_bridge_checkpoint,
)


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


def test_bridge_resolves_name_remap_from_checkpoint_metadata(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"architectures": ["DeepseekV4ForCausalLM"]}))
    weight_map = {"embed.weight": "model.safetensors"}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    remap = Mock()
    resolve = Mock(return_value=remap)
    monkeypatch.setattr(bridge_iterator, "get_param_name_remap", resolve)

    assert _load_checkpoint_name_remap(str(tmp_path)) is remap
    resolve.assert_called_once_with(str(config_path), weight_map)


def test_bridge_preserves_names_without_checkpoint_metadata(tmp_path, caplog):
    name = "model.layers.0.self_attn.q_proj.weight"
    assert _load_checkpoint_name_remap(str(tmp_path))(name) == name
    assert "preserving Bridge export names" in caplog.text


@pytest.mark.parametrize("native_names", [True, False])
@pytest.mark.parametrize("plain_export", [True, False])
def test_bridge_canonicalizes_names_before_postprocessing(native_names, plain_export, monkeypatch):
    weight = object()
    megatron_name = "decoder.layers.0.self_attention.linear_q_down_proj.weight"
    hf_name = "model.layers.0.self_attn.wq_a.weight"
    source_name = "layers.0.attn.wq_a.weight" if native_names else hf_name
    iterator = object.__new__(HfWeightIteratorBridge)
    iterator.args = Namespace(params_dtype=torch.bfloat16)
    iterator.model = []
    iterator._bridge = Mock()
    iterator._bridge.get_conversion_tasks.return_value = []
    iterator._bridge.export_hf_weights.return_value = iter([(source_name, weight, megatron_name)])
    if plain_export:
        iterator._bridge.get_conversion_tasks.return_value = [_ExportTask(param_weight=None)]

        def export_hf_weights(model, *, conversion_tasks, weight_dtype=None, **kwargs):
            assert weight_dtype is None
            assert all(task.weight_dtype == torch.float32 for task in conversion_tasks)
            return iter([(source_name, weight, megatron_name)])

        iterator._bridge.export_hf_weights = export_hf_weights
    iterator._remap_hf_name = Mock(return_value=hf_name)
    received = []

    def postprocess(named_weights, weight_type):
        assert weight_type == "base"
        received.extend(named_weights)
        return iter(received)

    iterator._postprocess_and_quantize = postprocess
    monkeypatch.setattr(bridge_iterator.megatron_bridge_utils, "patch_megatron_model", lambda model: nullcontext())
    monkeypatch.setattr(bridge_iterator, "_iter_mm_tower_units", lambda *args, **kwargs: iter(()))

    assert list(iterator._iter_hf_param_units({}, materialize=True)) == [[(hf_name, weight)]]
    assert received == [(hf_name, weight, megatron_name)]
    iterator._remap_hf_name.assert_called_once_with(source_name)


@dataclass
class _ExportTask:
    param_weight: object
    param_name: str = "weight"
    vp_stage: int | None = 0
    weight_dtype: torch.dtype | None = None


def test_plain_export_dtype_preserves_owner_and_receiver_tasks():
    original_weight = object()
    replacement_weight = Mock()
    replacement_weight.cuda.return_value = replacement_weight
    owner = _ExportTask(param_weight=original_weight)
    receiver = _ExportTask(param_weight=None, vp_stage=None)
    tasks = list(
        _process_conversion_tasks(
            [owner, receiver],
            {"vp_stages.0.weight": replacement_weight},
            weight_dtype=torch.float32,
        )
    )

    assert all(task.weight_dtype == torch.float32 for task in tasks)
    assert tasks[0].param_weight is replacement_weight
    assert tasks[1].param_weight is None
    assert owner.param_weight is original_weight
    assert owner.weight_dtype is receiver.weight_dtype is None
