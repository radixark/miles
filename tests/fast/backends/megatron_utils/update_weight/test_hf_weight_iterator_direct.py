from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
from tests.fast.fixtures.args_fixtures import replace_config_values
from tests.fast.fixtures.inkling_provider_fixtures import inkling_adapter_model, inkling_provider_env

from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect
from miles.utils.args.custom_function import CustomFunctionConfig

_ = inkling_adapter_model, inkling_provider_env


class TestNativeInklingExport:
    def test_typed_provider_exports_actual_adapter_tensors(
        self, inkling_provider_env: SimpleNamespace, inkling_adapter_model: torch.nn.Module
    ) -> None:
        """Typed Inkling providers export HF-named tensors through the native adapter exporter."""
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect

        iterator = object.__new__(HfWeightIteratorDirect)
        iterator.args = inkling_provider_env.args
        iterator.model = [inkling_adapter_model]
        iterator.model_name = "inkling"
        adapter = inkling_adapter_model.lora_lm_head_adapter

        named = dict(iterator._export_pp_local_lora(adapter=None))

        assert set(named) == {"language_model.lm_head.lora_A.weight", "language_model.lm_head.lora_B.weight"}
        torch.testing.assert_close(named["language_model.lm_head.lora_A.weight"], adapter.head_A.to(torch.bfloat16))
        torch.testing.assert_close(named["language_model.lm_head.lora_B.weight"], adapter.head_B.to(torch.bfloat16))

    @pytest.mark.parametrize("provider", [None, CustomFunctionConfig(path="models.other.provider")])
    def test_raw_adapter_export_does_not_depend_on_custom_provider(
        self,
        inkling_provider_env: SimpleNamespace,
        inkling_adapter_model: torch.nn.Module,
        provider: CustomFunctionConfig | None,
    ) -> None:
        """The attached adapter determines exported tensors independently of provider selection."""
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect

        iterator = object.__new__(HfWeightIteratorDirect)
        iterator.args = replace_config_values(inkling_provider_env.args, custom_model_provider_path=provider)
        iterator.model = [inkling_adapter_model]
        iterator.model_name = "other"

        named = dict(iterator._export_pp_local_lora(adapter=None))

        assert set(named) == {"language_model.lm_head.lora_A.weight", "language_model.lm_head.lora_B.weight"}
        adapter = inkling_adapter_model.lora_lm_head_adapter
        torch.testing.assert_close(named["language_model.lm_head.lora_A.weight"], adapter.head_A.to(torch.bfloat16))
        torch.testing.assert_close(named["language_model.lm_head.lora_B.weight"], adapter.head_B.to(torch.bfloat16))


class TestExpertOwnerPartition:
    @pytest.mark.parametrize("edp_rank,expert_ids", [(0, [0, 1]), (1, [2])])
    def test_typed_backend_expert_count_assigns_each_expert_to_one_replica(
        self,
        expert_owner_iterator: Callable[[int], HfWeightIteratorDirect],
        edp_rank: int,
        expert_ids: list[int],
    ) -> None:
        """An uneven expert count partitions converted expert work without duplication or omission."""
        iterator = expert_owner_iterator(edp_rank)

        names = [info.name for batch in iterator._expert_batches for info in batch.param_infos]

        assert names == [f"decoder.layers.0.mlp.experts.linear_fc1.weight{i}" for i in expert_ids]
