from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from miles.backends.megatron_utils import initialize
from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace


class TestBackendDistributedWiring:
    @pytest.mark.parametrize("dp_mapping,order", [(False, "tp-cp-ep-dp-pp"), (True, "tp-cp-ep-pp-dp")])
    def test_parallel_group_sdk_receives_backend_owned_topology(
        self, monkeypatch: pytest.MonkeyPatch, dp_mapping: bool, order: str
    ) -> None:
        """Model-parallel process groups use typed backend topology and mapping order."""
        args = SimpleNamespace(
            backend=MegatronArgsNamespace(
                tensor_model_parallel_size=2,
                pipeline_model_parallel_size=4,
                virtual_pipeline_model_parallel_size=3,
                pipeline_model_parallel_comm_backend="nccl",
                context_parallel_size=1,
                hierarchical_context_parallel_sizes=None,
                expert_model_parallel_size=8,
                num_distributed_optimizer_instances=2,
                expert_tensor_parallel_size=1,
                distributed_timeout_minutes=17,
                nccl_communicator_config_path="communicators.yaml",
                use_tp_pp_dp_mapping=dp_mapping,
                use_gloo_process_groups=False,
            )
        )
        sdk = Mock()
        embedding = object()
        positions = object()
        monkeypatch.setattr(initialize.mpu, "initialize_model_parallel", sdk)

        initialize._initialize_distributed(
            args=args, get_embedding_ranks=embedding, get_position_embedding_ranks=positions
        )

        sdk.assert_called_once_with(
            2,
            4,
            3,
            pipeline_model_parallel_comm_backend="nccl",
            context_parallel_size=1,
            hierarchical_context_parallel_sizes=None,
            expert_model_parallel_size=8,
            num_distributed_optimizer_instances=2,
            expert_tensor_parallel_size=1,
            distributed_timeout_minutes=17,
            nccl_communicator_config_path="communicators.yaml",
            order=order,
            get_embedding_ranks=embedding,
            get_position_embedding_ranks=positions,
            create_gloo_process_groups=False,
        )
