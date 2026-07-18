"""Unit tests for HfWeightIteratorBase factory routing.

Validates that the right iterator subclass is selected based on megatron_to_hf_mode.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])


import json
from argparse import Namespace
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.megatron_utils.update_weight.hf_weight_iterator_base import HfWeightIteratorBase
from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import (
    HfWeightIteratorBridge,
    _process_adapter_conversion_tasks,
)
from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect

_BASE_MODULE = "miles.backends.megatron_utils.update_weight.hf_weight_iterator_base"


class _TestIterator(HfWeightIteratorBase):
    def get_hf_weight_chunks(self, megatron_local_weights, weight_type="base"):
        return iter(())


class TestHfWeightIteratorFactory:
    def _make_args(self, mode="bridge"):
        return Namespace(
            megatron_to_hf_mode=mode,
            hf_checkpoint="/fake/path",
            update_weight_buffer_size=1,
        )

    @patch(f"{_BASE_MODULE}.HfWeightIteratorBase.__init__", return_value=None)
    def test_bridge_mode_creates_bridge_iterator(self, mock_init):
        """Factory should select HfWeightIteratorBridge for 'bridge' mode."""
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import HfWeightIteratorBridge

        with patch.object(HfWeightIteratorBridge, "__init__", return_value=None):
            args = self._make_args("bridge")
            iterator = HfWeightIteratorBase.create(
                args=args, model=[MagicMock()], is_lora=True, model_name="qwen", quantization_config=None
            )
            assert isinstance(iterator, HfWeightIteratorBridge)

    @patch(f"{_BASE_MODULE}.HfWeightIteratorBase.__init__", return_value=None)
    def test_raw_mode_creates_direct_iterator(self, mock_init):
        """Factory should select HfWeightIteratorDirect for 'raw' mode."""
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect

        with patch.object(HfWeightIteratorDirect, "__init__", return_value=None):
            args = self._make_args("raw")
            iterator = HfWeightIteratorBase.create(
                args=args, model=[MagicMock()], is_lora=False, model_name="qwen", quantization_config=None
            )
            assert isinstance(iterator, HfWeightIteratorDirect)

    def test_invalid_mode_raises(self):
        args = self._make_args("invalid_mode")
        with pytest.raises(KeyError):
            HfWeightIteratorBase.create(
                args=args, model=[MagicMock()], is_lora=False, model_name="qwen", quantization_config=None
            )


def test_compressed_tensor_config_uses_checkpoint_packed_names(tmp_path):
    index = {
        "weight_map": {
            "model.layers.0.self_attention_res_proj.weight": "model.safetensors",
            "model.layers.0.experts.0.w1.weight_packed": "model.safetensors",
            "model.layers.0.experts.0.w1.weight_scale": "model.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    iterator = _TestIterator(
        Namespace(hf_checkpoint=str(tmp_path)),
        model=[],
        model_name="kimi_k3",
        quantization_config={"quant_method": "compressed-tensors"},
    )

    assert iterator.quantization_config["_miles_quantized_basenames"] == {"model.layers.0.experts.0.w1"}


def test_direct_kimi_lora_export_materializes_cpu_backup(monkeypatch):
    parameter = object()
    backup = MagicMock()
    cuda_backup = object()
    backup.cuda.return_value = cuda_backup
    seen = {}

    iterator = object.__new__(HfWeightIteratorDirect)
    iterator.args = Namespace()
    iterator.model = [object()]
    iterator.model_name = "kimi_k3"

    monkeypatch.setattr(
        "miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct.dist.get_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct.named_params_and_buffers",
        lambda _args, _model: iter((("module.module.decoder.layers.0.lora_adapter.weight", parameter),)),
    )

    def export(_model, *, materialize_parameter):
        seen["value"] = materialize_parameter(parameter)
        yield [("model.layers.0.lora_A.weight", seen["value"])]

    monkeypatch.setattr("miles_plugins.models.kimi_k3.lora.export_kimi_k3_lora_hf_chunks", export)

    chunks = list(
        iterator.get_hf_weight_chunks(
            {"module.module.decoder.layers.0.lora_adapter.weight": backup},
            weight_type="lora",
        )
    )

    assert chunks == [[("model.layers.0.lora_A.weight", cuda_backup)]]
    backup.cuda.assert_called_once_with()


@dataclass(frozen=True)
class _ConversionTask:
    param_name: str
    vp_stage: int
    param_weight: object


@dataclass(frozen=True)
class _AdapterTask:
    linear_in_task: _ConversionTask
    linear_out_task: _ConversionTask


def test_adapter_conversion_tasks_use_cpu_backup():
    stale_weight = object()
    cpu_in = MagicMock()
    cpu_out = MagicMock()
    cuda_in = object()
    cuda_out = object()
    cpu_in.cuda.return_value = cuda_in
    cpu_out.cuda.return_value = cuda_out
    task = _AdapterTask(
        linear_in_task=_ConversionTask("layer.adapter.linear_in.weight", 0, stale_weight),
        linear_out_task=_ConversionTask("layer.adapter.linear_out.weight", 0, stale_weight),
    )

    (processed,) = _process_adapter_conversion_tasks(
        [task],
        {
            "vp_stages.0.layer.adapter.linear_in.weight": cpu_in,
            "vp_stages.0.layer.adapter.linear_out.weight": cpu_out,
        },
    )

    assert processed.linear_in_task.param_weight is cuda_in
    assert processed.linear_out_task.param_weight is cuda_out
    cpu_in.cuda.assert_called_once_with()
    cpu_out.cuda.assert_called_once_with()


def test_adapter_conversion_tasks_require_cpu_backup():
    task = _AdapterTask(
        linear_in_task=_ConversionTask("layer.adapter.linear_in.weight", 0, object()),
        linear_out_task=_ConversionTask("layer.adapter.linear_out.weight", 0, object()),
    )

    with pytest.raises(AssertionError, match="Missing LoRA CPU backup"):
        _process_adapter_conversion_tasks([task], {})


def test_adapter_export_restores_bridge_materializer():
    stale_weight = object()
    cpu_in = MagicMock()
    cpu_out = MagicMock()
    cuda_in = object()
    cuda_out = object()
    cpu_in.cuda.return_value = cuda_in
    cpu_out.cuda.return_value = cuda_out
    task = _AdapterTask(
        linear_in_task=_ConversionTask("layer.adapter.linear_in.weight", 0, stale_weight),
        linear_out_task=_ConversionTask("layer.adapter.linear_out.weight", 0, stale_weight),
    )

    class _BridgeImpl:
        def __init__(self):
            self.seen = None

        def materialize_adapter_weights(self, tasks):
            self.seen = tasks
            return tasks

    class _AutoBridge:
        def __init__(self):
            self._model_bridge = _BridgeImpl()

        def export_adapter_weights(self, *_args, **_kwargs):
            self._model_bridge.materialize_adapter_weights([task])
            yield "weight"

    iterator = object.__new__(HfWeightIteratorBridge)
    iterator._bridge = _AutoBridge()
    iterator.model = [MagicMock()]

    assert list(
        iterator._export_adapter_weights_from_cpu_backup(
            {
                "vp_stages.0.layer.adapter.linear_in.weight": cpu_in,
                "vp_stages.0.layer.adapter.linear_out.weight": cpu_out,
            }
        )
    ) == ["weight"]
    processed = iterator._bridge._model_bridge.seen[0]
    assert processed.linear_in_task.param_weight is cuda_in
    assert processed.linear_out_task.param_weight is cuda_out
    assert "materialize_adapter_weights" not in iterator._bridge._model_bridge.__dict__
