from miles.backends.fsdp_utils.config import FsdpArgsNamespace
from miles.utils.args.runtime import RolloutConfig
from miles.utils.dp_schedule import _calculate_workloads


class TestRolloutFlopsOwnership:
    def test_native_rollout_workloads_use_raw_fsdp_model_dimensions(self) -> None:
        """Native rollout balancing reads its FSDP owner without requiring Megatron model dimensions."""
        backend = FsdpArgsNamespace(
            hidden_size=2,
            num_attention_heads=1,
            num_query_groups=1,
            vocab_size=3,
            ffn_hidden_size=4,
            num_experts=None,
            num_layers=1,
            kv_channels=2,
            q_lora_rank=None,
            kv_lora_rank=None,
            qk_pos_emb_head_dim=0,
            qk_head_dim=2,
            v_head_dim=2,
        )
        args = RolloutConfig.model_construct(train_backend="fsdp", raw_fsdp=backend)
        before = vars(backend).copy()

        assert _calculate_workloads(step_lengths=[0, 1, 3], args=args) == [0, 96, 312]
        assert vars(backend) == before
