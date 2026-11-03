import json
import os
import subprocess
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, LlamaConfig

from miles.utils.lora import arguments as lora_arguments
from miles.utils.lora.arguments import (
    _assert_adapter_layout_defined,
    _resolve_lora_targets,
    _validate_adapter_state_args,
    _validate_rollout_only_adapter,
    validate_lora_args,
    validate_lora_args_after_megatron,
    validate_multi_lora_args,
)


class TestLoraTargetOrdering:
    def test_exclusions_expand_to_stable_hf_module_order_without_changing_selectors(self) -> None:
        """Resolved adapter targets have reproducible order after exclusions expand module names."""
        config = LlamaConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=32,
        )
        args = Namespace(
            target_modules="all-linear",
            exclude_modules="q_proj",
            multi_lora=False,
            megatron_to_hf_mode="bridge",
            lora_type="canonical_lora",
        )

        hf_targets = _resolve_lora_targets(args, config)

        expected = [
            "model.layers.0.mlp.down_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.1.mlp.down_proj",
            "model.layers.1.mlp.gate_proj",
            "model.layers.1.mlp.up_proj",
            "model.layers.1.self_attn.k_proj",
            "model.layers.1.self_attn.o_proj",
            "model.layers.1.self_attn.v_proj",
        ]
        assert hf_targets == expected
        assert args.target_modules == "all-linear"
        assert args.exclude_modules == "q_proj"


def test_lora_arguments_import_no_megatron_to_hf():
    """megatron_to_hf loads every converter and SGLang's quantization kernels, which probe the GPU on ROCm (#4201)."""
    code = "import sys, miles.utils.lora.arguments; sys.exit('miles.backends.megatron_utils.megatron_to_hf' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


_NO_SNAPSHOTS = dict(
    save_hf=None,
    eval_hf_dir=None,
    eval_uses_snapshots=False,
    save=None,
    lora_rank=8,
    lora_adapter_path=None,
    debug_rollout_only=False,
)


@pytest.mark.parametrize(
    "overrides",
    [
        {"save": "/ckpt"},
        {"save_hf": "/hf/{rollout_id}"},
        {"eval_hf_dir": "/eval", "eval_uses_snapshots": True},
        {"lora_adapter_path": "/adapter"},
    ],
    ids=["save", "save_hf", "eval_snapshots", "lora_adapter_path"],
)
def test_adapter_slots_need_the_families_hf_lora_layout(overrides) -> None:
    """Native trains glm4, which has no HF LoRA layout; saving or loading its adapter would crash building the slots."""
    args = Namespace(**(_NO_SNAPSHOTS | overrides))
    with pytest.raises(AssertionError, match="model_type='glm4', which miles does not define"):
        _assert_adapter_layout_defined(args, AutoConfig.for_model("glm4"))
    _assert_adapter_layout_defined(args, AutoConfig.for_model("qwen3"))


def test_a_rollout_only_run_builds_no_slots_for_any_family() -> None:
    args = Namespace(**(_NO_SNAPSHOTS | {"lora_adapter_path": "/adapter", "debug_rollout_only": True}))
    _assert_adapter_layout_defined(args, AutoConfig.for_model("glm4"))


def test_an_eval_dir_without_eval_snapshots_builds_no_slots() -> None:
    """The actor publishes eval snapshots only under eval_uses_snapshots; validation must not refuse the run."""
    args = Namespace(**(_NO_SNAPSHOTS | {"eval_hf_dir": "/eval"}))
    _assert_adapter_layout_defined(args, AutoConfig.for_model("glm4"))


def test_a_run_without_adapter_snapshots_trains_any_family() -> None:
    _assert_adapter_layout_defined(Namespace(**_NO_SNAPSHOTS), AutoConfig.for_model("glm4"))


def _adapter_state_args(**overrides) -> Namespace:
    return Namespace(
        **{
            "save": "/ckpt",
            "lora_adapter_path": None,
            "debug_disable_optimizer": False,
            "no_save_optim": False,
            "no_load_optim": False,
            "optimizer": "sgd",
            "sgd_momentum": 0.0,
            "exp_avg_dtype": torch.float32,
            "exp_avg_sq_dtype": torch.float32,
            "main_params_dtype": torch.float32,
            "use_distributed_optimizer": False,
            "use_gloo_process_groups": True,
            "use_precision_aware_optimizer": False,
            "optimizer_cpu_offload": False,
            "overlap_param_gather": False,
        }
        | overrides
    )


@pytest.mark.parametrize(
    "overrides",
    [{}, {"no_save_optim": True, "lora_adapter_path": "/adapter"}],
    ids=["saves_optimizer", "loads_optimizer"],
)
def test_a_non_adam_optimizer_cannot_store_adapter_optimizer_state(overrides) -> None:
    with pytest.raises(AssertionError, match="store Adam state; got --optimizer sgd"):
        _validate_adapter_state_args(_adapter_state_args(**overrides))


@pytest.mark.parametrize("overrides", [{}, {"lora_adapter_path": "/adapter", "no_load_optim": True}])
def test_a_non_adam_optimizer_saves_and_loads_the_masters_alone(overrides) -> None:
    _validate_adapter_state_args(_adapter_state_args(no_save_optim=True, **overrides))


def test_sgd_with_momentum_cannot_save_the_adapter() -> None:
    # the masters-only checkpoint would drop the momentum buffer
    with pytest.raises(AssertionError, match="need --sgd-momentum 0; got 0.9"):
        _validate_adapter_state_args(_adapter_state_args(no_save_optim=True, sgd_momentum=0.9))


@pytest.mark.parametrize("optimizer", ["muon", "dist_muon", "lion"])
def test_an_optimizer_other_than_adam_or_sgd_cannot_save_the_adapter(optimizer) -> None:
    with pytest.raises(AssertionError, match=f"adam or sgd; got --optimizer {optimizer}"):
        _validate_adapter_state_args(_adapter_state_args(optimizer=optimizer, no_save_optim=True))


def test_convert_save_refuses_an_existing_destination(tmp_path):
    destination = tmp_path / "converted"
    destination.mkdir()
    args = Namespace(
        lora_type="lora",
        multi_lora_n_adapters=0,
        lora_rank=8,
        lora_alpha=16.0,
        lora_adapter_path=str(tmp_path),
        lora_adapter_convert_save=str(destination),
        train_backend="megatron",
        megatron_to_hf_mode="bridge",
        indep_dp=False,
    )
    with pytest.raises(AssertionError, match="already exists"):
        validate_lora_args(args)


def test_convert_save_without_lora_is_refused(tmp_path):
    args = Namespace(
        lora_type="lora",
        multi_lora_n_adapters=0,
        lora_rank=0,
        lora_adapter_path=None,
        lora_adapter_convert_save=str(tmp_path / "converted"),
    )
    with pytest.raises(AssertionError, match="converts --lora-adapter-path"):
        validate_lora_args(args)


def test_convert_save_is_refused_under_indep_dp(tmp_path):
    args = Namespace(
        lora_type="lora",
        multi_lora_n_adapters=0,
        lora_rank=8,
        lora_alpha=16.0,
        lora_adapter_path=str(tmp_path),
        lora_adapter_convert_save=str(tmp_path / "converted"),
        train_backend="megatron",
        megatron_to_hf_mode="bridge",
        indep_dp=True,
    )
    with pytest.raises(AssertionError, match="does not support --indep-dp"):
        validate_lora_args(args)


def test_convert_save_is_refused_under_rollout_only(tmp_path):
    args = Namespace(
        lora_type="lora",
        multi_lora_n_adapters=0,
        lora_rank=8,
        lora_alpha=16.0,
        lora_adapter_path=str(tmp_path),
        lora_adapter_convert_save=str(tmp_path / "converted"),
        train_backend="megatron",
        megatron_to_hf_mode="bridge",
        indep_dp=False,
        debug_rollout_only=True,
    )
    with pytest.raises(AssertionError, match="--debug-rollout-only"):
        validate_lora_args(args)


def test_a_relative_convert_save_is_made_absolute_on_the_driver(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = Namespace(
        lora_type="lora",
        multi_lora_n_adapters=0,
        lora_rank=8,
        lora_alpha=16.0,
        lora_adapter_path=str(tmp_path),
        lora_adapter_convert_save="converted",
        train_backend="megatron",
        megatron_to_hf_mode="bridge",
        indep_dp=False,
        debug_rollout_only=True,
    )
    # rollout-only stops validation right after the path is resolved
    with pytest.raises(AssertionError, match="--debug-rollout-only"):
        validate_lora_args(args)
    assert args.lora_adapter_convert_save == os.path.join(os.getcwd(), "converted")


def test_multi_lora_refuses_an_adapter_path():
    with pytest.raises(AssertionError, match="--lora-adapter-path is single LoRA only"):
        validate_multi_lora_args(Namespace(multi_lora_n_adapters=2, lora_rank=8, lora_adapter_path="/adapter"))


@pytest.mark.parametrize("save, lora_adapter_path", [("/save", None), (None, "/adapter")])
def test_a_disabled_optimizer_cannot_save_or_load_the_adapter_checkpoint(save, lora_adapter_path):
    args = Namespace(save=save, lora_adapter_path=lora_adapter_path, debug_disable_optimizer=True)
    with pytest.raises(AssertionError, match="--debug-disable-optimizer"):
        _validate_adapter_state_args(args)


def test_rollout_only_skips_the_adapter_checkpoint_checks():
    args = Namespace(
        lora_rank=8,
        lora_adapter_path="/adapter",
        save=None,
        multi_lora=False,
        megatron_to_hf_mode="bridge",
        debug_rollout_only=True,
        debug_disable_optimizer=True,
    )
    validate_lora_args_after_megatron(args)


def test_the_adapter_checkpoint_requires_fp32_masters():
    with pytest.raises(AssertionError, match="fp32 masters"):
        _validate_adapter_state_args(_adapter_state_args(no_save_optim=True, main_params_dtype=torch.float16))


class TestRolloutOnlyAdapter:
    @pytest.fixture(autouse=True)
    def _qwen3(self, monkeypatch):
        config = SimpleNamespace(to_dict=lambda: {"model_type": "qwen3", "num_hidden_layers": 1})
        monkeypatch.setattr(lora_arguments, "load_hf_config", lambda _: config)

    @staticmethod
    def _validate(path):
        _validate_rollout_only_adapter(Namespace(lora_adapter_path=str(path), hf_checkpoint="/hf"))

    def test_a_hub_id_goes_to_sglang_as_is(self):
        self._validate("org/adapter")

    def test_a_peft_dir_the_library_does_not_read_goes_to_sglang_as_is(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
        (tmp_path / "adapter_model.bin").write_bytes(b"")
        self._validate(tmp_path)

    def test_official_tinker_names_are_refused(self, tmp_path):
        save_file(
            {"base_model.model.model.unembed_tokens.lora_A.weight": torch.zeros(2, 2)},
            str(tmp_path / "adapter_model.safetensors"),
        )
        (tmp_path / "adapter_config.json").write_text("{}")
        with pytest.raises(AssertionError, match="official Tinker tensor names"):
            self._validate(tmp_path)

    def test_a_checkpoint_only_a_trainer_reads_is_refused(self, tmp_path):
        torch.save({}, tmp_path / "adapter_megatron_rank0.pt")
        with pytest.raises(AssertionError, match="only a trainer can read"):
            self._validate(tmp_path)
