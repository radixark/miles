from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    ReplicatedMapping,
)

from miles.backends.megatron_utils import hf_export
from miles.backends.megatron_utils.bridge_lora_helpers import (
    _qualify_deepseek_v4_lora_targets,
    _setup_lora_model_via_bridge,
)
from miles.backends.megatron_utils.lora_utils import save_lora_checkpoint
from miles_plugins.megatron_bridge.deepseek_v4 import (
    _get_dsv4_explicit_mappings,
    is_deepseek_v4_config,
)


def test_dsv4_explicit_mapping_semantics():
    mappings = {mapping.megatron_param: mapping for mapping in _get_dsv4_explicit_mappings()}
    assert len(mappings) == len(_get_dsv4_explicit_mappings())

    for suffix in ("fn", "base", "scale"):
        assert isinstance(mappings[f"decoder.layers.*.hc_attn_{suffix}"], ReplicatedMapping)
        assert isinstance(mappings[f"decoder.layers.*.hc_ffn_{suffix}"], ReplicatedMapping)
        assert isinstance(mappings[f"decoder.hc_head_params.hc_head_{suffix}"], ReplicatedMapping)

    assert isinstance(mappings["decoder.layers.*.self_attention.attn_sink"], ColumnParallelMapping)
    for prefix in (
        "decoder.layers.*.self_attention.compressor",
        "decoder.layers.*.self_attention.indexer.compressor",
    ):
        for suffix in ("ape", "wkv.weight", "wgate.weight", "norm.weight"):
            assert isinstance(mappings[f"{prefix}.{suffix}"], ReplicatedMapping)

    assert isinstance(mappings["decoder.layers.*.self_attention.wq_b.weight"], AutoMapping)
    assert isinstance(
        mappings["decoder.layers.*.self_attention.indexer.linear_wq_b.weight"],
        AutoMapping,
    )
    assert isinstance(mappings["decoder.layers.*.mlp.router.tid2eid"], ReplicatedMapping)
    assert isinstance(mappings["decoder.layers.*.mlp.router.expert_bias"], ReplicatedMapping)


def test_dsv4_lora_construction_selects_native_provider():
    args = Namespace(hf_checkpoint="checkpoint")
    config = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
    sentinel = object()

    with (
        patch(
            "miles.backends.megatron_utils.bridge_lora_helpers.load_hf_config",
            return_value=config,
        ),
        patch(
            "miles.backends.megatron_utils.bridge_lora_helpers._setup_deepseek_v4_lora_model",
            return_value=sentinel,
        ) as setup_native,
    ):
        assert _setup_lora_model_via_bridge(args) is sentinel

    setup_native.assert_called_once_with(args)


def test_dsv4_main_attention_targets_are_disambiguated_from_indexer():
    assert _qualify_deepseek_v4_lora_targets(
        ["wq_a", "wq_b", "wkv", "wo_a", "wo_b", "indexer.wq_b", "linear_fc1"]
    ) == [
        "*.self_attention.wq_a",
        "*.self_attention.wq_b",
        "*.self_attention.wkv",
        "*.self_attention.wo_a",
        "*.self_attention.wo_b",
        "indexer.wq_b",
        "linear_fc1",
    ]


def test_dsv4_lora_portable_export_is_adapter_only():
    config = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
    args = Namespace(hf_checkpoint="checkpoint")
    model = [object()]

    with (
        patch.object(hf_export, "is_lora_model", return_value=True),
        patch.object(hf_export, "load_hf_config", return_value=config),
    ):
        assert hf_export._uses_adapter_only_export(args, model)


def test_dsv4_architecture_match_is_exact():
    assert is_deepseek_v4_config(SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]))
    assert not is_deepseek_v4_config(SimpleNamespace(architectures=["DeepseekV3ForCausalLM"]))


def test_dsv4_adapter_only_export_is_fail_closed(tmp_path):
    config = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
    args = Namespace(
        hf_checkpoint="checkpoint",
        save_hf=str(tmp_path / "rollout-{rollout_id}"),
    )
    model = [object()]

    with (
        patch.object(hf_export, "is_lora_model", return_value=True),
        patch.object(hf_export, "load_hf_config", return_value=config),
        patch.object(hf_export, "get_parallel_state") as parallel_state,
        patch.object(hf_export.torch.distributed, "get_rank", return_value=0),
        patch.object(hf_export, "save_lora_checkpoint", side_effect=RuntimeError("no mapping")) as save_adapter,
    ):
        parallel_state.return_value.effective_dp_cp.rank = 0
        parallel_state.return_value.tp.rank = 0
        with pytest.raises(RuntimeError, match="no mapping"):
            hf_export.save_hf_model(args, 0, model, raise_on_error=True)

    save_adapter.assert_called_once()
    assert save_adapter.call_args.kwargs["require_hf_export"] is True
    assert not (tmp_path / "rollout-0" / ".complete").exists()


def test_required_adapter_export_rejects_empty_bridge_result(tmp_path):
    model = SimpleNamespace(
        named_parameters=lambda: [("layer.adapter.linear_in.weight", torch.nn.Parameter(torch.ones(1)))]
    )
    parallel_state = SimpleNamespace(
        effective_dp=SimpleNamespace(rank=0),
        cp=SimpleNamespace(rank=0),
        tp=SimpleNamespace(rank=0),
        pp=SimpleNamespace(rank=0),
    )
    bridge = SimpleNamespace(export_adapter_weights=lambda *_args, **_kwargs: iter(()))

    with (
        patch(
            "miles.backends.megatron_utils.lora_utils.get_parallel_state",
            return_value=parallel_state,
        ),
        patch("megatron.bridge.AutoBridge.from_hf_pretrained", return_value=bridge),
        patch(
            "miles.utils.megatron_bridge_utils.patch_megatron_model",
            side_effect=lambda _model: nullcontext(),
        ),
    ):
        with pytest.raises(RuntimeError, match="Required HF PEFT adapter export failed"):
            save_lora_checkpoint(
                [model],
                Namespace(hf_checkpoint="checkpoint"),
                str(tmp_path),
                require_hf_export=True,
            )

    assert not (tmp_path / "adapter_model.bin").exists()
