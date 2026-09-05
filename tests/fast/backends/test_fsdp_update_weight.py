"""FSDP's HF weight iterator."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from miles.backends.fsdp_utils import hf_weight_iterator as iterator_module
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement

_ITERATOR_MODULE = "miles.backends.fsdp_utils.hf_weight_iterator"


def _model(state: dict, *, model_type: str = "", sync_dtypes: dict | None = None):
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type), state_dict=lambda: state)
    if sync_dtypes is not None:
        model._fsdp_sync_dtypes = sync_dtypes
    return model


def _iterator(model, *, model_name: str = "qwen3forcausallm"):
    return iterator_module.get_hf_weight_iterator(
        Namespace(update_weight_buffer_size=1 << 30),
        model,
        required_placement=WeightUpdatePlacement(gather_pp=True),
        model_name=model_name,
        quantization_config=None,
    )


def _units(iterator, *, materialize=True):
    with patch(f"{_ITERATOR_MODULE}.gather_full_param", side_effect=lambda p, async_op=False: p):
        return [dict(bucket) for bucket in iterator.iter_hf_weights(None, materialize=materialize)]


def test_every_parameter_streams_under_its_hf_name():
    state = {"model.embed_tokens.weight": torch.zeros(2), "lm_head.weight": torch.ones(2)}
    buckets = _units(_iterator(_model(state)))
    assert {name for bucket in buckets for name in bucket} == set(state)


def test_fp32_master_weights_are_cast_to_the_rollout_contract_dtype():
    value = torch.tensor([1.0 + 2**-20], dtype=torch.float32)
    state = {"fp32_weight": value, "bf16_weight": value.clone()}
    buckets = _units(
        _iterator(_model(state, sync_dtypes={"fp32_weight": torch.float32, "bf16_weight": torch.bfloat16}))
    )
    synced = {name: tensor for bucket in buckets for name, tensor in bucket.items()}
    assert synced["fp32_weight"].dtype is torch.float32 and torch.equal(synced["fp32_weight"], value)
    assert synced["bf16_weight"].dtype is torch.bfloat16
    assert not torch.equal(synced["bf16_weight"].to(torch.float32), value)


def test_non_materializing_ranks_join_every_gather_but_yield_nothing():
    state = {"a": torch.zeros(1), "b": torch.zeros(1)}
    with patch(f"{_ITERATOR_MODULE}.gather_full_param", side_effect=lambda p, async_op=False: p) as gather:
        buckets = list(_iterator(_model(state)).iter_hf_weights(None, materialize=False))
    assert buckets == []
    assert gather.call_count == 2


def test_batched_experts_are_unfused_through_the_registered_transform():
    """qwen3_moe keeps experts batched as [E, ...]; the engine wants one tensor per expert."""
    gate_up = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 2, 4)
    state = {"model.layers.0.mlp.experts.gate_up_proj": gate_up}
    model = _model(state, model_type="qwen3_moe")
    with (
        patch(f"{_ITERATOR_MODULE}.gather_full_param", side_effect=lambda p, async_op=False: p),
        patch(
            f"{_ITERATOR_MODULE}.get_param_transform",
            return_value=lambda name, full, model: ((f"expert.{i}", full[i]) for i in range(full.shape[0])),
        ),
    ):
        buckets = list(_iterator(model).iter_hf_weights(None))
    names = [name for bucket in buckets for name, _ in bucket]
    assert names == ["expert.0", "expert.1"]


def test_lora_adapters_are_refused():
    with pytest.raises(NotImplementedError, match="LoRA"):
        list(_iterator(_model({}))._iter_hf_adapter_units("lora", None, materialize=True))
