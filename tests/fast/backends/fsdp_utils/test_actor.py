import sys
from argparse import Namespace
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch
from tests.fast.fixtures.args_fixtures import make_trainer_args

from miles.backends.fsdp_utils import actor as actor_module
from miles.backends.fsdp_utils.adaptations.specs import resolve_arch_adapter
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.torch_native import actor as base_module
from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.utils import distributed_utils
from miles.utils.dp_schedule import TrainParallelConfig
from miles.utils.ft_utils.heartbeat_utils import SimpleHeartbeat
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.init_once import InitOnce


class TestFSDPInit:
    def test_fsdp_init_rejects_an_independent_dp_quorum_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FSDP initialization rejects a non-null independent-DP quorum store address."""

        def fail_if_initialization_continues() -> None:
            raise RuntimeError("FSDP initialization continued past quorum validation")

        def ignore_device(_device: object) -> None:
            return None

        def ignore_process_group(**_kwargs: object) -> None:
            return None

        def zero() -> int:
            return 0

        def one() -> int:
            return 1

        def false() -> bool:
            return False

        class FakeDumper:
            apply_source_patches = staticmethod(fail_if_initialization_continues)

        dumper_module = ModuleType("sglang.srt.debug_utils.dumper")
        dumper_module.dumper = FakeDumper()
        monkeypatch.setitem(sys.modules, "sglang.srt.debug_utils.dumper", dumper_module)
        monkeypatch.setattr(actor_module.torch.cuda, "set_device", ignore_device)
        monkeypatch.setattr(actor_module.dist, "init_process_group", ignore_process_group)
        monkeypatch.setattr(actor_module.dist, "get_rank", zero)
        monkeypatch.setattr(actor_module.dist, "get_world_size", one)
        monkeypatch.setattr(actor_module.dist, "is_initialized", false)
        monkeypatch.setattr(actor_module.dist, "new_group", lambda **_kwargs: object())
        monkeypatch.setattr(distributed_utils, "GLOO_GROUP", None)

        actor = object.__new__(actor_module.FSDPTrainRayActor)
        actor._rank = 0
        actor._heartbeat = SimpleHeartbeat()
        actor._init_once = InitOnce(type(actor).__name__)
        args = make_trainer_args(
            train_backend="fsdp",
            debug_deterministic_collective=False,
            distributed_backend="nccl",
            distributed_timeout_minutes=1,
            dumper_enable=True,
            env_report=None,
            num_gpus_per_node=1,
        )

        with pytest.raises(AssertionError):
            actor.init(
                args,
                "actor",
                indep_dp_info=IndepDPInfo.create_trivial(),
                indep_dp_store_addr="10.0.0.9:1234",
            )


class TestFSDPTrainExternalData:
    def test_fsdp_train_rejects_external_critic_data_before_starting_work(self) -> None:
        """FSDP rejects critic output before accessing state needed for training work."""
        actor = object.__new__(actor_module.FSDPTrainRayActor)
        external_data = TrainStepOutput(outcome=TrainStepOutcome.NORMAL)

        with pytest.raises(AssertionError, match="fsdp backend trains no critic"):
            actor.train(rollout_id=1, rollout_data_ref=object(), external_data=external_data)


class TestFSDPModelInputs:
    def test_packing_kwargs_follow_the_batch_boundaries_so_padding_stays_one_segment(self) -> None:
        """Pad tokens carry position 0; boundaries derived from `position_ids` would make each one a document."""
        actor = object.__new__(actor_module.FSDPTrainRayActor)
        actor.arch_adapter = resolve_arch_adapter(SimpleNamespace(model_type="qwen3_5_text"))
        cu_seqlens_host = (0, 3, 5, 8)  # two documents, then the 3-token pad segment get_batch appends
        batch = {
            "tokens": torch.zeros(1, 8, dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2, 0, 1, 0, 0, 0]]),
            "cu_seqlens": torch.tensor(cu_seqlens_host, dtype=torch.int32),
            "cu_seqlens_host": cu_seqlens_host,
            "max_seqlen": 3,
            "unconcat_tokens": [torch.zeros(3, dtype=torch.long), torch.zeros(2, dtype=torch.long)],
        }

        model_args = actor._get_model_inputs_args(batch)

        assert model_args["cu_seq_lens_q"] is batch["cu_seqlens"]
        assert model_args["max_length_q"] == 3
        assert model_args["seq_idx"].tolist() == [[0, 0, 0, 1, 1, 2, 2, 2]]

    def test_stock_hf_attention_gets_every_pad_token_as_its_own_segment(self) -> None:
        """A 4-token pad tail kept as one segment would exceed max_length 3, leaving its rows unwritten."""
        actor = object.__new__(actor_module.FSDPTrainRayActor)
        actor.arch_adapter = resolve_arch_adapter(SimpleNamespace(model_type="qwen3"))
        cu_seqlens_host = (0, 3, 5, 9)
        batch = {
            "tokens": torch.zeros(1, 9, dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2, 0, 1, 0, 0, 0, 0]]),
            "cu_seqlens": torch.tensor(cu_seqlens_host, dtype=torch.int32),
            "cu_seqlens_host": cu_seqlens_host,
            "max_seqlen": 4,
            "unconcat_tokens": [torch.zeros(3, dtype=torch.long), torch.zeros(2, dtype=torch.long)],
        }

        model_args = actor._get_model_inputs_args(batch)

        assert model_args["cu_seq_lens_q"].tolist() == [0, 3, 5, 6, 7, 8, 9]
        assert model_args["max_length_q"] == 3


class TestFSDPTrainParallelConfigWiring:
    def test_init_derives_a_config_without_schedule_support_and_train_forwards_it(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fsdp_three_rank_state: ParallelState,
        fsdp_debug_actor: actor_module.FSDPTrainRayActor,
    ) -> None:
        """FSDP init records its live DP layout without precomputed scheduling and train hands it to the loader."""
        args = make_trainer_args(
            train_backend="fsdp", dumper_enable=False, seed=0, offload_train=False, debug_rollout_only=True
        )
        received: list[TrainParallelConfig] = []

        def load_rollout_data(
            *,
            args: Namespace,
            rollout_data_ref: object,
            witness_info: object,
            train_parallel_config: TrainParallelConfig,
            output_store_read: object,
        ) -> tuple[dict[str, list], object]:
            received.append(train_parallel_config)
            return {"tokens": []}, nullcontext()

        monkeypatch.setattr(base_module, "get_rollout_data", load_rollout_data)

        fsdp_debug_actor.init(args, "actor", indep_dp_info=IndepDPInfo.create_trivial(), indep_dp_store_addr=None)
        fsdp_debug_actor.train(rollout_id=3, rollout_data_ref=object())

        assert received == [fsdp_debug_actor.train_parallel_config]
        assert received[0].dp_size == fsdp_three_rank_state.effective_dp.size == 3
        assert received[0].cp_size == 1
        assert not received[0].independent_dp
        assert not received[0].supports_precomputed_schedule
