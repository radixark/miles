import pytest
import torch
from tests.fast.backends.megatron_utils.sdk_config_fixtures import _ModelSdk, model_sdk
from tests.fast.fixtures.args_fixtures import make_trainer_args, make_trainer_config

from miles.backends.training_utils.data.rollout import DataIterator
from miles.backends.training_utils.types import TrainStepOutcome

__all__ = ["model_sdk"]


class TestBackendSchedulerOptions:
    @pytest.mark.parametrize(
        ("decay", "fraction", "wsd", "expected_decay", "expected_warmup", "expected_wsd"),
        [(None, None, None, 120, 12, None), (7, 0.25, 2, 28, 7, 8)],
    )
    def test_scheduler_sdk_receives_backend_derived_sample_counts(
        self,
        model_sdk: _ModelSdk,
        decay: int | None,
        fraction: float | None,
        wsd: int | None,
        expected_decay: int,
        expected_warmup: int,
        expected_wsd: int | None,
    ) -> None:
        """Backend batch size derives scheduler steps while explicit decay and fractional warmup retain precedence."""
        args = make_trainer_config(
            num_rollout=10,
            rollout_batch_size=6,
            n_samples_per_prompt=2,
            global_batch_size=4,
            lr_decay_iters=decay,
            lr_warmup_fraction=fraction,
            lr_warmup_iters=3,
            lr_wsd_decay_iters=wsd,
            lr_warmup_init=0.01,
            lr=0.2,
            min_lr=0.003,
            lr_decay_style="cosine",
            start_weight_decay=0.02,
            end_weight_decay=0.08,
            weight_decay_incr_style="linear",
            use_checkpoint_opt_param_scheduler=True,
            override_opt_param_scheduler=False,
            lr_wsd_decay_style="linear",
        )
        optimizer = object()

        result = model_sdk.module.get_optimizer_param_scheduler(args, optimizer)

        model_sdk.scheduler.assert_called_once_with(
            optimizer,
            init_lr=0.01,
            max_lr=0.2,
            min_lr=0.003,
            lr_warmup_steps=expected_warmup,
            lr_decay_steps=expected_decay,
            lr_decay_style="cosine",
            start_wd=0.02,
            end_wd=0.08,
            wd_incr_steps=120,
            wd_incr_style="linear",
            use_checkpoint_opt_param_scheduler=True,
            override_opt_param_scheduler=False,
            wsd_decay_steps=expected_wsd,
            lr_wsd_decay_style="linear",
        )
        assert result is model_sdk.scheduler.return_value
        assert args.backend.train_iters == 30
        with pytest.raises(AttributeError, match="immutable"):
            args.backend.train_iters = 99


class TestBackendOptimizerOptions:
    @pytest.mark.parametrize("kind", ["adam", "dist_muon"])
    @pytest.mark.parametrize("use_gloo", [False, True])
    def test_optimizer_sdk_receives_backend_config_and_process_group_policy(
        self, model_sdk: _ModelSdk, kind: str, use_gloo: bool
    ) -> None:
        """Optimizer assembly selects the right external builder and forwards backend-owned numerical settings."""
        args = make_trainer_config(
            optimizer=kind,
            lr=0.031,
            min_lr=0.002,
            weight_decay=0.17,
            clip_grad=0.61,
            bf16=True,
            fp16=False,
            use_distributed_optimizer=True,
            use_gloo_process_groups=use_gloo,
            moe_use_upcycling=False,
            lora_rank=0,
            multi_lora=False,
            megatron_to_hf_mode="raw",
            debug_disable_optimizer=False,
            stream_optimizer_state_to_disk=False,
            custom_model_provider_path=None,
            global_batch_size=4,
            num_rollout=2,
            rollout_batch_size=4,
            n_samples_per_prompt=1,
            lr_decay_iters=None,
            lr_warmup_fraction=None,
            lr_warmup_iters=0,
            lr_wsd_decay_iters=None,
            lr_warmup_init=0.0,
            lr_decay_style="linear",
            start_weight_decay=0.17,
            end_weight_decay=0.17,
            weight_decay_incr_style="constant",
            use_checkpoint_opt_param_scheduler=False,
            override_opt_param_scheduler=False,
            lr_wsd_decay_style="linear",
        )

        models, optimizer, scheduler = model_sdk.module.setup_model_and_optimizer(args)

        selected = model_sdk.optimizer if kind == "adam" else model_sdk.muon
        unselected = model_sdk.muon if kind == "adam" else model_sdk.optimizer
        selected.assert_called_once()
        unselected.assert_not_called()
        kwargs = selected.call_args.kwargs
        config = kwargs["config"]
        assert (config.optimizer, config.lr, config.min_lr, config.weight_decay, config.clip_grad) == (
            kind,
            0.031,
            0.002,
            0.17,
            0.61,
        )
        assert config.bf16 and not config.fp16 and config.use_distributed_optimizer
        assert kwargs["model_chunks"] == models == [model_sdk.model]
        assert kwargs["use_gloo_process_groups"] is use_gloo
        if kind == "dist_muon":
            assert kwargs["layer_wise_distributed_optimizer"] is True
        assert optimizer is selected.return_value
        assert scheduler is model_sdk.scheduler.return_value


class TestBackendTrainingDimensions:
    @pytest.mark.parametrize("forward_only", [False, True])
    def test_real_train_step_passes_backend_dimensions_to_pipeline_sdk(
        self, model_sdk: _ModelSdk, forward_only: bool
    ) -> None:
        """Distinct sequence and batch dimensions reach the pipeline for training and scoring passes."""
        args = make_trainer_args(
            calculate_per_token_loss=False,
            seq_length=113,
            decoder_seq_length=47,
            micro_batch_size=3,
            debug_disable_optimizer=True,
            enable_sample_ownership_checker=False,
            enable_witness=False,
            save_local_weight_checksum=False,
            dumper_enable=False,
            use_sampling_support_replay=False,
            ci_test=False,
            enable_mtp_training=False,
            use_rollout_entropy=False,
        )
        iterator = DataIterator({}, micro_batch_size=3)
        if forward_only:
            model_sdk.module.forward_only(lambda *_args, **_kwargs: {}, args, [model_sdk.model], [iterator], [4], 0)
        else:
            model_sdk.module.train_one_step(
                args=args,
                rollout_id=0,
                step_id=0,
                data_iterator=[iterator],
                model=[model_sdk.model],
                optimizer=None,
                opt_param_scheduler=None,
                num_microbatches=4,
                num_rollouts=8,
                witness_info=None,
                attempt=0,
            )

        [call] = model_sdk.pipeline_calls
        assert (call["seq_length"], call["micro_batch_size"], call["num_microbatches"], call["forward_only"]) == (
            113,
            3,
            4,
            forward_only,
        )
        if not forward_only:
            assert call["decoder_seq_length"] == 47
        assert model_sdk.model.training

    @pytest.mark.parametrize("per_token,expected", [(False, 4.0), (True, 2.0)])
    def test_backend_loss_mode_selects_the_real_metric_denominator(
        self, model_sdk: _ModelSdk, per_token: bool, expected: float
    ) -> None:
        """Token-normalized metrics divide by reduced token count while rollout mode uses the rollout count."""
        args = make_trainer_args(
            calculate_per_token_loss=per_token,
            seq_length=8,
            decoder_seq_length=8,
            micro_batch_size=1,
            debug_disable_optimizer=True,
            enable_sample_ownership_checker=False,
            enable_witness=False,
            save_local_weight_checksum=False,
            dumper_enable=False,
            use_sampling_support_replay=False,
            ci_test=False,
            enable_mtp_training=False,
        )
        model_sdk.pipeline_outputs = [{"keys": ["loss"], "values": torch.tensor([8.0, 16.0])}]

        metrics, _, outcome = model_sdk.module.train_one_step(
            args=args,
            rollout_id=0,
            step_id=0,
            data_iterator=[DataIterator({}, micro_batch_size=1)],
            model=[model_sdk.model],
            optimizer=None,
            opt_param_scheduler=None,
            num_microbatches=1,
            num_rollouts=4,
            witness_info=None,
            attempt=0,
        )

        assert metrics == {"loss": expected}
        assert outcome == TrainStepOutcome.NORMAL


class TestBackendTrainingValidation:
    def test_upcycling_is_rejected_before_optimizer_creation(self, model_sdk: _ModelSdk) -> None:
        """Unsupported upcycling must fail without constructing an optimizer."""
        args = make_trainer_config(moe_use_upcycling=True)

        with pytest.raises(AssertionError):
            model_sdk.module.setup_model_and_optimizer(args)

        model_sdk.optimizer.assert_not_called()
        model_sdk.muon.assert_not_called()

    @pytest.mark.parametrize("check_nan", [False, True])
    def test_backend_nan_policy_controls_external_gradient_preparation(
        self, model_sdk: _ModelSdk, check_nan: bool
    ) -> None:
        """The explicit gradient check runs only when the backend optimizer does not own NaN checking."""
        from unittest.mock import Mock

        args = make_trainer_args(
            check_for_nan_in_loss_and_grad=check_nan,
            calculate_per_token_loss=False,
            seq_length=8,
            decoder_seq_length=8,
            micro_batch_size=1,
            debug_disable_optimizer=False,
            enable_sample_ownership_checker=False,
            enable_witness=False,
            save_local_weight_checksum=False,
            dumper_enable=False,
            use_sampling_support_replay=False,
            ci_test=False,
            enable_mtp_training=False,
        )
        optimizer = Mock(prepare_grads=Mock(return_value=True), step=Mock(return_value=(True, 1.0, 0)))
        scheduler = Mock()

        _, _, outcome = model_sdk.module.train_one_step(
            args=args,
            rollout_id=0,
            step_id=0,
            data_iterator=[DataIterator({}, micro_batch_size=1)],
            model=[model_sdk.model],
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            num_microbatches=1,
            num_rollouts=4,
            witness_info=None,
            attempt=0,
        )

        assert optimizer.prepare_grads.call_count == int(not check_nan)
        assert optimizer.step.call_count == int(check_nan)
        if check_nan:
            scheduler.step.assert_called_once_with(increment=4)
        else:
            scheduler.step.assert_not_called()
        assert outcome == TrainStepOutcome.NORMAL
