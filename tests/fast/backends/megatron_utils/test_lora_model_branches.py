"""Mock-based tests for LoRA branch logic in miles.backends.megatron_utils.model.

Validates that setup_model_and_optimizer, save, and save_hf_model correctly
route to LoRA-specific code paths depending on configuration — without GPU.
"""

import json
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.fast.fixtures.args_fixtures import make_trainer_args, make_trainer_config

from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

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

        args = make_trainer_args(use_distributed_optimizer=True, overlap_param_gather=True)
        assert should_disable_forward_pre_hook(args) is True

    def test_optimizer_false(self):
        from miles.backends.megatron_utils.model import should_disable_forward_pre_hook

        args = make_trainer_args(use_distributed_optimizer=False, overlap_param_gather=True)
        assert should_disable_forward_pre_hook(args) is False

    def test_overlap_false(self):
        from miles.backends.megatron_utils.model import should_disable_forward_pre_hook

        args = make_trainer_args(use_distributed_optimizer=True, overlap_param_gather=False)
        assert should_disable_forward_pre_hook(args) is False


# ---------------------------------------------------------------------------
# setup_model_and_optimizer — LoRA branch routing
# ---------------------------------------------------------------------------

_MODEL_MODULE = "miles.backends.megatron_utils.model"


class TestSetupModelAndOptimizerLoraBranch:
    """Verify that LoRA-enabled actor + bridge mode routes to _setup_lora_model_via_bridge."""

    def _make_args(self, lora_rank=32, role="actor", mode="bridge"):
        return make_trainer_config(
            lora_rank=lora_rank,
            lora_adapter_path=None,
            custom_model_provider_path=None,
            megatron_to_hf_mode=mode,
            model_name=None,
            moe_use_upcycling=False,
            debug_disable_optimizer=False,
            stream_optimizer_state_to_disk=False,
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
            use_gloo_process_groups=False,
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
    @patch(f"{_MODEL_MODULE}.get_model_provider_func")
    @patch(f"{_MODEL_MODULE}.wrap_model_provider_with_lora")
    @patch(f"{_MODEL_MODULE}._setup_lora_model_via_bridge")
    def test_raw_mode_lora_wraps_the_provider_with_native_lora(
        self, mock_lora_setup, mock_wrap, mock_provider, mock_get_model, mock_opt, mock_sched
    ):
        from miles.backends.megatron_utils.model import setup_model_and_optimizer

        mock_get_model.return_value = [MagicMock()]
        mock_opt.return_value = MagicMock(param_groups=[])
        mock_sched.return_value = MagicMock()

        args = self._make_args(lora_rank=32, role="actor", mode="raw")
        args = args.model_copy(update={"offload_train": False})
        setup_model_and_optimizer(args, role="actor")

        mock_lora_setup.assert_not_called()
        mock_wrap.assert_called_once_with(mock_provider.return_value, args)
        assert mock_get_model.call_args.args[0] is mock_wrap.return_value


# ---------------------------------------------------------------------------
# save — LoRA vs regular branch
# ---------------------------------------------------------------------------


class TestSaveLoRaBranch:
    @patch(f"{_MODEL_MODULE}.save_model_hashes")
    @patch(f"{_MODEL_MODULE}.enable_forward_pre_hook")
    @patch(f"{_MODEL_MODULE}.disable_forward_pre_hook")
    @patch(f"{_MODEL_MODULE}.should_disable_forward_pre_hook", return_value=False)
    @patch(f"{_MODEL_MODULE}.save_single_lora")
    @patch(f"{_MODEL_MODULE}.is_lora_model", return_value=True)
    def test_lora_model_calls_lora_save(
        self, mock_is_lora, mock_save_lora, mock_should, mock_disable, mock_enable, mock_save_hashes
    ):
        from miles.backends.megatron_utils.model import save

        model = [MagicMock()]
        slots = MagicMock()
        save(make_trainer_args(), 42, model, MagicMock(), MagicMock(), adapter_slots=slots, schedule_first_rollout=5)

        assert mock_save_lora.call_args.args[3] is slots

    @patch(f"{_MODEL_MODULE}.save_model_hashes")
    @patch(f"{_MODEL_MODULE}.enable_forward_pre_hook")
    @patch(f"{_MODEL_MODULE}.disable_forward_pre_hook")
    @patch(f"{_MODEL_MODULE}.should_disable_forward_pre_hook", return_value=False)
    @patch(f"{_MODEL_MODULE}.save_checkpoint")
    @patch(f"{_MODEL_MODULE}.is_lora_model", return_value=False)
    def test_non_lora_model_calls_regular_save(
        self, mock_is_lora, mock_save_ckpt, mock_should, mock_disable, mock_enable, mock_save_hashes
    ):
        from miles.backends.megatron_utils.model import save

        model = [MagicMock()]
        save(make_trainer_args(), 42, model, MagicMock(), MagicMock())

        mock_save_ckpt.assert_called_once()


# ---------------------------------------------------------------------------
# save_hf_model — raw-mode LoRA merges natively instead of going through Bridge
# ---------------------------------------------------------------------------

_HF_EXPORT_MODULE = "miles.backends.megatron_utils.hf_export"


class TestSaveHfModelRawLoRA:
    @patch(f"{_HF_EXPORT_MODULE}.write_sampler_export")
    @patch(f"{_HF_EXPORT_MODULE}._get_hf_bridge")
    @patch(f"{_HF_EXPORT_MODULE}.merge_lora_into_weights")
    @patch(f"{_HF_EXPORT_MODULE}.named_params_and_buffers")
    @patch(f"{_HF_EXPORT_MODULE}.is_lora_model", return_value=True)
    @patch(f"{_HF_EXPORT_MODULE}.write_checkpoint_dir", side_effect=lambda path, write, **_kwargs: write(path))
    @patch(f"{_HF_EXPORT_MODULE}.get_parallel_state")
    def test_raw_lora_writes_merged_weights_and_the_adapter(
        self, _parallel_state, _write_dir, _is_lora, named_weights, merge, get_bridge, sampler_view, tmp_path
    ):
        from miles.backends.megatron_utils.hf_export import save_hf_model

        named_weights.return_value = [("decoder.w", MagicMock())]
        publisher = MagicMock()
        model = [MagicMock()]
        slots = MagicMock(bf16_adapter_key_style="sampler")
        args = Namespace(
            megatron_to_hf_mode="raw",
            hf_checkpoint=_hf_dir(tmp_path),
            save_hf=None,
            lora_rank=8,
            lora_alpha=16.0,
            lora_type="canonical_lora",
            lora_dropout=0.05,
            experts_shared_outer_loras=False,
        )

        save_hf_model(args, 0, model, publisher=publisher, adapter_slots=slots, path=tmp_path, raise_on_error=True)

        merge.assert_called_once_with(model, dict(named_weights.return_value))
        publisher.write_model.assert_called_once_with(
            tmp_path, weights=merge.return_value, hf_checkpoint=args.hf_checkpoint
        )
        sampler_view.assert_called_once_with(
            tmp_path / "adapter", slots, 0, slots.parametrization.return_value, slots.targets, lora_dropout=0.05
        )
        get_bridge.assert_not_called()


class _SharedOuterNativeSlots:
    """Native's export of shared-outer experts: sglang's stacked names, the shared side ``[1, ...]``."""

    bf16_adapter_key_style = "sampler"
    targets = get_hf_lora_targets({"model_type": "qwen3_moe", "num_hidden_layers": 1, "num_experts": 4})
    experts = "model.layers.0.mlp.experts"

    def slot_spec(self, slot):
        return None

    def parametrization(self, **fields):
        modules = tuple(f"{self.experts}.{leaf}" for leaf in ("gate_proj", "up_proj", "down_proj"))
        return AdapterParametrization(**fields, target_modules=modules)

    def export_bf16_adapter(self, slot):
        for leaf, out_features in (("w1", 6), ("w3", 6)):
            yield f"{self.experts}.{leaf}.lora_A.weight", torch.ones(1, 2, 8)
            yield f"{self.experts}.{leaf}.lora_B.weight", torch.ones(4, out_features, 2)
        yield f"{self.experts}.w2.lora_A.weight", torch.ones(4, 2, 6)
        yield f"{self.experts}.w2.lora_B.weight", torch.ones(1, 8, 2)


@patch(f"{_HF_EXPORT_MODULE}.merge_lora_into_weights")
@patch(f"{_HF_EXPORT_MODULE}.named_params_and_buffers", return_value=[])
@patch(f"{_HF_EXPORT_MODULE}.is_lora_model", return_value=True)
@patch(f"{_HF_EXPORT_MODULE}.write_checkpoint_dir", side_effect=lambda path, write, **_kwargs: write(path))
@patch(f"{_HF_EXPORT_MODULE}.get_parallel_state")
def test_native_shared_outer_save_hf_writes_the_stacked_adapter_sglang_serves(
    _parallel_state, _write_dir, _is_lora, _named_weights, _merge, tmp_path
):
    """A per-expert 2D PEFT export has no name for the shared ``[1, ...]`` side; sglang loads the stacked one, with
    the run's rank, alpha and dropout as main wrote them."""
    from safetensors.torch import load_file

    from miles.backends.megatron_utils.hf_export import save_hf_model

    args = Namespace(
        megatron_to_hf_mode="raw",
        hf_checkpoint=_hf_dir(tmp_path),
        save_hf=None,
        lora_rank=2,
        lora_alpha=4.0,
        lora_type="lora",
        lora_dropout=0.05,
        experts_shared_outer_loras=True,
    )

    save_hf_model(
        args,
        0,
        [MagicMock()],
        publisher=MagicMock(),
        adapter_slots=_SharedOuterNativeSlots(),
        path=tmp_path,
        raise_on_error=True,
    )

    tensors = load_file(tmp_path / "adapter" / "adapter_model.safetensors")
    experts = _SharedOuterNativeSlots.experts
    assert {name: tuple(t.shape) for name, t in tensors.items()} == {
        f"{experts}.w1.lora_A.weight": (1, 2, 8),
        f"{experts}.w1.lora_B.weight": (4, 6, 2),
        f"{experts}.w3.lora_A.weight": (1, 2, 8),
        f"{experts}.w3.lora_B.weight": (4, 6, 2),
        f"{experts}.w2.lora_A.weight": (4, 2, 6),
        f"{experts}.w2.lora_B.weight": (1, 8, 2),
    }
    config = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert (config["r"], config["lora_alpha"], config["lora_dropout"]) == (2, 4.0, 0.05)
    # HF module names, which sglang normalizes
    assert config["target_modules"] == [f"{experts}.{leaf}" for leaf in ("down_proj", "gate_proj", "up_proj")]


def _hf_dir(tmp_path) -> str:
    """A base whose config.json --save-hf reads for the adapter's base fingerprint."""
    hf = tmp_path / "hf"
    hf.mkdir()
    config = {"model_type": "qwen3", "hidden_size": 8, "num_hidden_layers": 1, "vocab_size": 16}
    (hf / "config.json").write_text(json.dumps(config | {"num_attention_heads": 2, "num_key_value_heads": 1}))
    return str(hf)


def _fingerprint(hf: str) -> str:
    from miles.backends.training_utils.checkpoint.adapter.parametrization import base_model_fingerprint
    from miles.utils.hf_utils.config import load_hf_config

    return base_model_fingerprint(load_hf_config(hf).to_dict())


class TestSaveHfModelBridgeLoRA:
    @patch(f"{_HF_EXPORT_MODULE}.has_plain_peft_form", return_value=True)
    @patch(f"{_HF_EXPORT_MODULE}.write_sampler_export")
    @patch(f"{_HF_EXPORT_MODULE}.write_peft_export")
    @patch(f"{_HF_EXPORT_MODULE}.get_gloo_group")
    @patch(f"{_HF_EXPORT_MODULE}.torch")
    @patch(f"{_HF_EXPORT_MODULE}.patch_megatron_model")
    @patch(f"{_HF_EXPORT_MODULE}._get_hf_bridge")
    @patch(f"{_HF_EXPORT_MODULE}.is_lora_model", return_value=True)
    @patch(f"{_HF_EXPORT_MODULE}.write_checkpoint_dir", side_effect=lambda path, write, **_kwargs: write(path))
    @patch(f"{_HF_EXPORT_MODULE}.get_parallel_state")
    @pytest.mark.parametrize(
        "slot_spec,rank,alpha", [(None, 8, 16.0), ("spec", 4, 12.0)], ids=["single_lora", "multi_lora_slot_0"]
    )
    def test_bridge_lora_writes_slot_0_through_the_peft_export(
        self,
        _parallel_state,
        _write_dir,
        _is_lora,
        _get_bridge,
        _patch_model,
        torch_mod,
        _gloo,
        peft_export,
        sampler_view,
        plain_peft,
        tmp_path,
        slot_spec,
        rank,
        alpha,
    ):
        """Multi-LoRA describes slot 0 by its own rank and alpha, single LoRA by the run's."""
        from miles.backends.megatron_utils.hf_export import save_hf_model
        from miles.utils.lora.utils import AdapterSpec

        torch_mod.distributed.get_rank.return_value = 1
        publisher, slots = MagicMock(), MagicMock()
        slots.slot_spec.return_value = AdapterSpec(slot=0, rank=4, alpha=12.0) if slot_spec else None
        args = Namespace(
            megatron_to_hf_mode="bridge",
            hf_checkpoint=_hf_dir(tmp_path),
            save_hf=None,
            lora_rank=8,
            lora_alpha=16.0,
            lora_type="lora",
            lora_dropout=0.05,
            experts_shared_outer_loras=False,
        )

        save_hf_model(
            args, 0, [MagicMock()], publisher=publisher, adapter_slots=slots, path=tmp_path, raise_on_error=True
        )

        slots.slot_spec.assert_called_once_with(0)
        slots.parametrization.assert_called_once_with(
            base_model=args.hf_checkpoint,
            lora_rank=rank,
            lora_alpha=alpha,
            lora_type="lora",
            experts_shared_outer_loras=False,
            base_model_fingerprint=_fingerprint(args.hf_checkpoint),
        )
        peft_export.assert_called_once_with(
            tmp_path / "adapter", slots, 0, slots.parametrization.return_value, slots.targets, lora_dropout=0.05
        )
        sampler_view.assert_not_called()

        # canonical_lora's own A per fused component has no PEFT name: sglang's names, as main wrote
        plain_peft.return_value = False
        peft_export.reset_mock()
        save_hf_model(
            args, 0, [MagicMock()], publisher=publisher, adapter_slots=slots, path=tmp_path, raise_on_error=True
        )
        peft_export.assert_not_called()
        sampler_view.assert_called_once_with(
            tmp_path / "adapter", slots, 0, slots.parametrization.return_value, slots.targets, lora_dropout=0.05
        )
