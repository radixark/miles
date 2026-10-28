from argparse import Namespace

import pytest

from tests.fast.backends.megatron_utils.update_weight.export_owner_fixtures import duplicate_param_model

from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import (
    HfWeightIteratorDirect,
    _check_and_fix_partition,
    _get_megatron_local_param_infos,
)


__all__ = ["duplicate_param_model"]


class TestExportBackendOptions:
    @pytest.mark.parametrize("rank,expected", [(None, []), (128, ["q_lora_a_proj"])])
    def test_atomic_export_groups_use_the_backend_query_adapter_rank(
        self, rank: int | None, expected: list[str]
    ) -> None:
        """The exporter keeps fused query and KV adapter projections atomic when backend Q-LoRA is enabled."""
        iterator = object.__new__(HfWeightIteratorDirect)
        iterator.args = Namespace(backend=Namespace(q_lora_rank=rank))
        iterator.model_name = "deepseekv3"

        groups = iterator._hf_atomic_update_groups()

        assert [group.key for group in groups] == expected
        if rank is not None:
            assert groups[0].suffixes == (
                ".self_attn.q_a_proj.weight",
                ".self_attn.kv_a_proj_with_mqa.weight",
            )

    @pytest.mark.parametrize(
        "swiglu,name,stride,dimension,expected",
        [
            (True, "linear_fc1.weight", 1, -1, (2, 0)),
            (True, "linear_fc1.weight", 2, 1, (2, 1)),
            (False, "linear_fc1.weight", 1, -1, (1, -1)),
            (True, "linear_fc2.weight", 1, 0, (1, 1)),
        ],
    )
    def test_export_partition_metadata_follows_backend_swiglu_without_rewriting_valid_dimensions(
        self, swiglu: bool, name: str, stride: int, dimension: int, expected: tuple[int, int]
    ) -> None:
        """SwiGLU fixes missing grouped-FC1 metadata while preserving an explicit partition dimension."""
        assert (
            _check_and_fix_partition(
                args=Namespace(backend=Namespace(swiglu=swiglu)),
                name=name,
                partition_stride=stride,
                partition_dim=dimension,
            )
            == expected
        )


class TestBackendMtpMetadata:
    @pytest.mark.parametrize("layers", [None, 1])
    def test_duplicate_pipeline_metadata_requires_backend_mtp_and_selects_the_lowest_owner(
        self, duplicate_param_model: object, layers: int | None
    ) -> None:
        """Duplicate PP metadata is legal only for MTP and resolves to one deterministic source rank."""
        args = Namespace(backend=Namespace(num_experts=None, mtp_num_layers=layers))
        if layers is None:
            with pytest.raises(AssertionError):
                _get_megatron_local_param_infos(args=args, model=[duplicate_param_model], gather_pp=True)
        else:
            infos, expert_infos = _get_megatron_local_param_infos(
                args=args, model=[duplicate_param_model], gather_pp=True
            )
            assert expert_infos == []
            assert len(infos) == 1
            assert infos[0].src_rank == 0
            assert tuple(infos[0].shape) == (2, 2)
