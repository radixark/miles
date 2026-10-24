import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from tests.fast.backends.megatron_utils.sdk_config_fixtures import _ModelSdk, _SdkModel, model_sdk
from tests.fast.fixtures.args_fixtures import make_trainer_args

from miles.backends.training_utils.data.rollout import DataIterator
from miles.backends.training_utils.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger import logger as event_logger_module
from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity

__all__ = ["model_sdk"]


class TestBackendIterationHooks:
    @pytest.mark.parametrize(
        "chunks,overlap,align,param_overlap,param_align",
        [
            (1, True, True, False, False),
            (2, False, False, True, True),
            (2, True, True, True, True),
            (2, True, False, True, False),
            (1, False, False, False, False),
        ],
    )
    def test_backend_overlap_flags_select_single_or_virtual_pipeline_sync_hooks(
        self, model_sdk: _ModelSdk, chunks: int, overlap: bool, align: bool, param_overlap: bool, param_align: bool
    ) -> None:
        """Backend overlap and alignment flags install the external DDP hooks with the required scalar or list shape."""
        args = make_trainer_args(
            overlap_grad_reduce=overlap,
            align_grad_reduce=align,
            overlap_param_gather=param_overlap,
            align_param_gather=param_align,
        )
        models = [_SdkModel() for _ in range(chunks)]
        optimizer = Mock()

        config = model_sdk.module.setup_train_iteration_config(args, models, optimizer, disable_optimizer=False)

        assert config.grad_scale_func == optimizer.scale_loss
        assert config.timers is None
        expected_no_sync = [model.no_sync for model in models]
        expected_grad_sync = [model.start_grad_sync for model in models]
        expected_param_sync = [model.start_param_sync for model in models]
        if chunks == 1:
            expected_no_sync, expected_grad_sync, expected_param_sync = (
                expected_no_sync[0],
                expected_grad_sync[0],
                expected_param_sync[0],
            )
        assert config.no_sync_func == (expected_no_sync if overlap else None)
        assert config.grad_sync_func == (expected_grad_sync if overlap and align else None)
        assert config.param_sync_func == (expected_param_sync if param_overlap and param_align else None)
        assert config.finalize_model_grads_func is model_sdk.module.finalize_model_grads_with_empty_cache


class TestBackendTrainLoopOptions:
    @pytest.mark.parametrize(
        "per_token,manual_gc,expected",
        [(False, False, 2.0), (True, False, 8.0), (False, True, 2.0)],
    )
    def test_real_train_loop_uses_backend_gc_and_mtp_loss_scaling(
        self,
        model_sdk: _ModelSdk,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        per_token: bool,
        manual_gc: bool,
        expected: float,
    ) -> None:
        """Actual training logs the backend-selected MTP normalization and applies manual collection only when enabled."""
        from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

        args = make_trainer_args(
            manual_gc=manual_gc,
            manual_gc_interval=3,
            calculate_per_token_loss=per_token,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            align_param_gather=False,
            use_distributed_optimizer=False,
            debug_disable_optimizer=True,
            enable_mtp_training=True,
            reset_optimizer_states=False,
            seq_length=8,
            decoder_seq_length=8,
            micro_batch_size=1,
            enable_sample_ownership_checker=False,
            enable_witness=False,
            save_local_weight_checksum=False,
            dumper_enable=False,
            use_sampling_support_replay=False,
            ci_test=False,
            trainer_model_id=None,
        )
        disable, collect, reduced, cleaned = Mock(), Mock(), Mock(), Mock()
        monkeypatch.setattr(model_sdk.module.gc, "disable", disable)
        monkeypatch.setattr(model_sdk.module.gc, "collect", collect)
        monkeypatch.setattr(MTPLossLoggingHelper, "tracker", {"values": torch.tensor(8.0)})
        monkeypatch.setattr(MTPLossLoggingHelper, "reduce_loss_in_tracker", reduced)
        monkeypatch.setattr(MTPLossLoggingHelper, "clean_loss_in_tracker", cleaned)
        event_logger = EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"))
        monkeypatch.setattr(event_logger_module, "_event_logger", event_logger)

        outcome = model_sdk.module.train(
            args=args,
            rollout_id=0,
            model=[model_sdk.model],
            optimizer=None,
            opt_param_scheduler=None,
            data_iterator=[DataIterator({}, micro_batch_size=1)],
            num_microbatches=[4],
            num_rollouts=[8],
            witness_info=None,
            attempt=0,
        )

        records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        [metric] = [record for record in records if record["type"] == "metric"]
        assert metric["metrics"]["train/mtp_loss"] == expected
        assert outcome == TrainStepOutcome.NORMAL
        assert disable.call_count == collect.call_count == int(manual_gc)
        reduced.assert_called_once_with()
        cleaned.assert_called_once_with()

    def test_negative_backend_manual_gc_interval_is_rejected_before_pipeline_execution(
        self, model_sdk: _ModelSdk
    ) -> None:
        """An invalid manual collection interval fails before any training microbatch is submitted."""
        args = make_trainer_args(
            manual_gc=True,
            manual_gc_interval=-1,
            debug_disable_optimizer=True,
            reset_optimizer_states=False,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            align_param_gather=False,
        )

        with pytest.raises(AssertionError, match="Manual garbage collection interval"):
            model_sdk.module.train(
                args=args,
                rollout_id=0,
                model=[model_sdk.model],
                optimizer=None,
                opt_param_scheduler=None,
                data_iterator=[DataIterator({}, micro_batch_size=1)],
                num_microbatches=[1],
                num_rollouts=[1],
                witness_info=None,
                attempt=0,
            )

        assert model_sdk.pipeline_calls == []
