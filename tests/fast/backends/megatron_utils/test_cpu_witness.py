from pathlib import Path
from typing import Literal

import pytest
from tests.fast.backends.megatron_utils.conftest import _CpuStepEnvironment

from miles.backends.megatron_utils.cpu_witness import log_cpu_witness
from miles.backends.megatron_utils.model import train_one_step
from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import TrainerCpuWitnessEvent, TrainerGroupMappingEvent
from miles.utils.audit_utils.witness.cpu import cpu_witnesses


class TestOptimizerCpuWitness:
    @pytest.mark.parametrize("reason,checkpoint_id", [("save", "disk-save-1"), ("transfer", None)])
    def test_save_boundary_emission_preserves_kind_identity_and_full_weight_history(
        self,
        cpu_step_environment: _CpuStepEnvironment,
        ownership_event_dir: Path,
        reason: Literal["save", "transfer"],
        checkpoint_id: str | None,
    ) -> None:
        """Disk saves and memory transfers serialize distinct boundaries with the complete rank witness."""
        env = cpu_step_environment
        for step_id in range(2):
            train_one_step(
                args=env.args,
                rollout_id=0,
                step_id=step_id,
                data_iterator=[env.iterator],
                model=[env.model],
                optimizer=env.optimizer,
                opt_param_scheduler=env.scheduler,
                num_microbatches=1,
                num_rollouts=1,
                witness_info=None,
                attempt=0,
            )
        log_cpu_witness(
            model=[env.model],
            lineage_id=None,
            trainer_model_id=env.args.trainer_model_id,
            rollout_id=0,
            reset=True,
            reason=reason,
            checkpoint_id=checkpoint_id,
        )

        event = [event for event in read_events(ownership_event_dir) if isinstance(event, TrainerCpuWitnessEvent)][-1]
        assert event.reason == reason
        assert event.checkpoint_id == checkpoint_id
        assert event.reset
        assert event.records == cpu_witnesses([env.model])[0].records
        assert len(event.records) == 2

    @pytest.mark.parametrize("disable_optimizer,invalid_gradient", [(False, False), (True, False), (False, True)])
    def test_only_successful_optimizer_updates_add_weight_evidence(
        self,
        cpu_step_environment: _CpuStepEnvironment,
        ownership_event_dir: Path,
        disable_optimizer: bool,
        invalid_gradient: bool,
    ) -> None:
        """NORMAL outcomes without an optimizer update cannot claim samples reached the weights."""
        env = cpu_step_environment
        env.args.debug_disable_optimizer = disable_optimizer
        env.optimizer.found_inf = invalid_gradient
        initial = env.model.weight.item()

        train_one_step(
            args=env.args,
            rollout_id=0,
            step_id=0,
            data_iterator=[env.iterator],
            model=[env.model],
            optimizer=env.optimizer,
            opt_param_scheduler=env.scheduler,
            num_microbatches=1,
            num_rollouts=1,
            witness_info=None,
            attempt=0,
        )

        applied = not disable_optimizer and not invalid_gradient
        assert (env.model.weight.item() != initial) == applied
        records = cpu_witnesses([env.model])[0].records
        assert len(records) == int(applied)
        if applied:
            assert records[0]["group_indices"] == [7]

    def test_multiple_optimizer_steps_report_only_their_consumed_microbatches(
        self, cpu_step_environment: _CpuStepEnvironment, ownership_event_dir: Path
    ) -> None:
        """A group split across optimizer steps does not certify unconsumed siblings."""
        env = cpu_step_environment
        for step_id in range(2):
            train_one_step(
                args=env.args,
                rollout_id=0,
                step_id=step_id,
                data_iterator=[env.iterator],
                model=[env.model],
                optimizer=env.optimizer,
                opt_param_scheduler=env.scheduler,
                num_microbatches=1,
                num_rollouts=1,
                witness_info=None,
                attempt=0,
            )

        mappings = [event for event in read_events(ownership_event_dir) if isinstance(event, TrainerGroupMappingEvent)]
        assert [event.groups for event in mappings] == [{7: [10]}, {7: [11]}]
        assert [record["step_id"] for record in cpu_witnesses([env.model])[0].records] == [0, 1]
        witness_events = [
            event for event in read_events(ownership_event_dir) if isinstance(event, TrainerCpuWitnessEvent)
        ]
        assert [event.reset for event in witness_events] == [True, False]
        assert [len(event.records) for event in witness_events] == [1, 1]

    def test_dynamic_schedule_uses_actual_rows_instead_of_a_contiguous_prefix(
        self, cpu_step_environment: _CpuStepEnvironment, ownership_event_dir: Path
    ) -> None:
        """Length balancing preserves group ownership when microbatches reorder rows."""
        env = cpu_step_environment
        env.iterator.micro_batch_indices = [[2, 0], [1]]
        env.iterator.micro_batch_size = None
        train_one_step(
            args=env.args,
            rollout_id=0,
            step_id=0,
            data_iterator=[env.iterator],
            model=[env.model],
            optimizer=env.optimizer,
            opt_param_scheduler=env.scheduler,
            num_microbatches=1,
            num_rollouts=2,
            witness_info=None,
            attempt=0,
        )

        [mapping] = [
            event for event in read_events(ownership_event_dir) if isinstance(event, TrainerGroupMappingEvent)
        ]
        assert mapping.groups == {8: [12], 7: [10]}
