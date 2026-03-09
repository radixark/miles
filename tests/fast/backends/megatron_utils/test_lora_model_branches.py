"""Mock-based tests for LoRA branch logic in miles.backends.megatron_utils.model.

Validates that setup_model_and_optimizer, save, and save_hf_model correctly
route to LoRA-specific code paths depending on configuration — without GPU.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# _ensure_model_list
# ---------------------------------------------------------------------------


class TestEnsureModelList:
    def test_list_passthrough(self):
        from miles.backends.megatron_utils.model import _ensure_model_list

        models = [MagicMock(), MagicMock()]
        assert _ensure_model_list(models) is models

    def test_non_list_wrapped(self):
        from miles.backends.megatron_utils.model import _ensure_model_list

        model = MagicMock()
        result = _ensure_model_list(model)
        assert isinstance(result, list)
        assert result[0] is model


# ---------------------------------------------------------------------------
# should_disable_forward_pre_hook
# ---------------------------------------------------------------------------


class TestShouldDisableForwardPreHook:
    def test_both_true(self):
        from miles.backends.megatron_utils.model import should_disable_forward_pre_hook

        args = Namespace(use_distributed_optimizer=True, overlap_param_gather=True)
        assert should_disable_forward_pre_hook(args) is True

    def test_optimizer_false(self):
        from miles.backends.megatron_utils.model import should_disable_forward_pre_hook

        args = Namespace(use_distributed_optimizer=False, overlap_param_gather=True)
        assert should_disable_forward_pre_hook(args) is False

    def test_overlap_false(self):
        from miles.backends.megatron_utils.model import should_disable_forward_pre_hook

        args = Namespace(use_distributed_optimizer=True, overlap_param_gather=False)
        assert should_disable_forward_pre_hook(args) is False


# ---------------------------------------------------------------------------
# setup_model_and_optimizer — LoRA branch routing
# ---------------------------------------------------------------------------


_MODEL_MODULE = "miles.backends.megatron_utils.model"


class TestSetupModelAndOptimizerLoraBranch:
    """Verify that LoRA-enabled actor + bridge mode routes to _setup_lora_model_via_bridge."""

    def _make_args(self, lora_rank=32, role="actor", mode="bridge"):
        return Namespace(
            lora_rank=lora_rank,
            lora_adapter_path=None,
            megatron_to_hf_mode=mode,
            moe_use_upcycling=False,
            load="/some/path",
            pretrained_checkpoint=None,
            # optimizer fields
            num_rollout=10,
            rollout_batch_size=8,
            n_samples_per_prompt=8,
            global_batch_size=32,
            lr_decay_iters=None,
            lr_wsd_decay_iters=None,
            lr_warmup_fraction=None,
            lr_warmup_iters=0,
            lr_warmup_init=0,
            lr=1e-5,
            min_lr=0,
            lr_decay_style="constant",
            start_weight_decay=0,
            end_weight_decay=0,
            weight_decay_incr_style="constant",
            use_checkpoint_opt_param_scheduler=False,
            override_opt_param_scheduler=False,
            lr_wsd_decay_style="linear",
            enable_gloo_process_groups=False,
        )

    @patch(f"{_MODEL_MODULE}.get_optimizer_param_scheduler")
    @patch(f"{_MODEL_MODULE}.get_megatron_optimizer")
    @patch(f"{_MODEL_MODULE}._setup_lora_model_via_bridge")
    def test_lora_actor_bridge_routes_to_lora_setup(self, mock_lora_setup, mock_opt, mock_sched):
        from miles.backends.megatron_utils.model import setup_model_and_optimizer

        mock_lora_setup.return_value = [MagicMock()]
        mock_opt.return_value = MagicMock(param_groups=[])
        mock_sched.return_value = MagicMock()

        args = self._make_args(lora_rank=32, role="actor", mode="bridge")
        model, _, _ = setup_model_and_optimizer(args, role="actor")

        mock_lora_setup.assert_called_once_with(args)

    @patch(f"{_MODEL_MODULE}.get_optimizer_param_scheduler")
    @patch(f"{_MODEL_MODULE}.get_megatron_optimizer")
    @patch(f"{_MODEL_MODULE}.get_model")
    @patch(f"{_MODEL_MODULE}.get_model_provider_func")
    @patch(f"{_MODEL_MODULE}._setup_lora_model_via_bridge")
    def test_lora_critic_skips_lora_setup(self, mock_lora_setup, mock_provider, mock_get_model, mock_opt, mock_sched):
        from miles.backends.megatron_utils.model import setup_model_and_optimizer

        mock_get_model.return_value = [MagicMock()]
        mock_opt.return_value = MagicMock(param_groups=[])
        mock_sched.return_value = MagicMock()

        args = self._make_args(lora_rank=32, role="critic", mode="bridge")
        setup_model_and_optimizer(args, role="critic")

        mock_lora_setup.assert_not_called()
        mock_get_model.assert_called_once()

    @patch(f"{_MODEL_MODULE}.get_optimizer_param_scheduler")
    @patch(f"{_MODEL_MODULE}.get_megatron_optimizer")
    @patch(f"{_MODEL_MODULE}.get_model")
    @patch(f"{_MODEL_MODULE}.get_model_provider_func")
    @patch(f"{_MODEL_MODULE}._setup_lora_model_via_bridge")
    def test_non_lora_skips_lora_setup(self, mock_lora_setup, mock_provider, mock_get_model, mock_opt, mock_sched):
        from miles.backends.megatron_utils.model import setup_model_and_optimizer

        mock_get_model.return_value = [MagicMock()]
        mock_opt.return_value = MagicMock(param_groups=[])
        mock_sched.return_value = MagicMock()

        args = self._make_args(lora_rank=0, role="actor", mode="bridge")
        setup_model_and_optimizer(args, role="actor")

        mock_lora_setup.assert_not_called()
        mock_get_model.assert_called_once()

    @patch(f"{_MODEL_MODULE}.get_optimizer_param_scheduler")
    @patch(f"{_MODEL_MODULE}.get_megatron_optimizer")
    @patch(f"{_MODEL_MODULE}.get_model")
    @patch(f"{_MODEL_MODULE}._setup_lora_model_via_bridge")
    def test_lora_raw_mode_skips_bridge(self, mock_lora_setup, mock_get_model, mock_opt, mock_sched):
        from miles.backends.megatron_utils.model import setup_model_and_optimizer

        mock_get_model.return_value = [MagicMock()]
        mock_opt.return_value = MagicMock(param_groups=[])
        mock_sched.return_value = MagicMock()

        args = self._make_args(lora_rank=32, role="actor", mode="raw")
        setup_model_and_optimizer(args, role="actor")

        mock_lora_setup.assert_not_called()
        mock_get_model.assert_called_once()


# ---------------------------------------------------------------------------
# save — LoRA vs regular branch
# ---------------------------------------------------------------------------


class TestSaveLoRaBranch:
    @patch(f"{_MODEL_MODULE}.save_model_hashes")
    @patch(f"{_MODEL_MODULE}.enable_forward_pre_hook")
    @patch(f"{_MODEL_MODULE}.disable_forward_pre_hook")
    @patch(f"{_MODEL_MODULE}.should_disable_forward_pre_hook", return_value=False)
    @patch(f"{_MODEL_MODULE}.get_args")
    @patch(f"{_MODEL_MODULE}.save_checkpoint_with_lora")
    @patch(f"{_MODEL_MODULE}.is_lora_model", return_value=True)
    def test_lora_model_calls_lora_save(
        self, mock_is_lora, mock_save_lora, mock_get_args, mock_should, mock_disable, mock_enable, mock_save_hashes
    ):
        from miles.backends.megatron_utils.model import save

        model = [MagicMock()]
        save(42, model, MagicMock(), MagicMock())

        mock_save_lora.assert_called_once()

    @patch(f"{_MODEL_MODULE}.save_model_hashes")
    @patch(f"{_MODEL_MODULE}.enable_forward_pre_hook")
    @patch(f"{_MODEL_MODULE}.disable_forward_pre_hook")
    @patch(f"{_MODEL_MODULE}.should_disable_forward_pre_hook", return_value=False)
    @patch(f"{_MODEL_MODULE}.get_args")
    @patch(f"{_MODEL_MODULE}.save_checkpoint")
    @patch(f"{_MODEL_MODULE}.is_lora_model", return_value=False)
    def test_non_lora_model_calls_regular_save(
        self, mock_is_lora, mock_save_ckpt, mock_get_args, mock_should, mock_disable, mock_enable, mock_save_hashes
    ):
        from miles.backends.megatron_utils.model import save

        model = [MagicMock()]
        save(42, model, MagicMock(), MagicMock())

        mock_save_ckpt.assert_called_once()
