import json
import os
import subprocess
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, LlamaConfig, PretrainedConfig

from miles.utils.lora import arguments as lora_arguments
from miles.utils.lora.arguments import (
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
    multi_lora=False,
)


def _check_layout(monkeypatch, args, hf_config):
    monkeypatch.setattr(lora_arguments, "load_hf_config", lambda path: hf_config)
    lora_arguments._assert_adapter_layout_defined(Namespace(**{**vars(args), "hf_checkpoint": "/hf"}))


class _MimoConfig(PretrainedConfig):
    """A family native trains with no HF LoRA layout; transformers ships no MiMo config."""

    model_type = "mimo"


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
def test_adapter_slots_need_the_families_hf_lora_layout(monkeypatch, overrides) -> None:
    """Native trains mimo, which has no HF LoRA layout; saving or loading its adapter would crash building the slots."""
    args = Namespace(**(_NO_SNAPSHOTS | overrides))
    with pytest.raises(AssertionError, match="model_type='mimo', which miles does not define"):
        _check_layout(monkeypatch, args, _MimoConfig())
    _check_layout(monkeypatch, args, AutoConfig.for_model("qwen3"))


def test_a_rollout_only_run_builds_no_slots_for_any_family(monkeypatch) -> None:
    args = Namespace(**(_NO_SNAPSHOTS | {"lora_adapter_path": "/adapter", "debug_rollout_only": True}))
    _check_layout(monkeypatch, args, _MimoConfig())


def test_an_eval_dir_without_eval_snapshots_builds_no_slots(monkeypatch) -> None:
    """The actor publishes eval snapshots only under eval_uses_snapshots; validation must not refuse the run."""
    args = Namespace(**(_NO_SNAPSHOTS | {"eval_hf_dir": "/eval"}))
    _check_layout(monkeypatch, args, _MimoConfig())


def test_a_run_without_adapter_snapshots_trains_any_family(monkeypatch) -> None:
    _check_layout(monkeypatch, Namespace(**_NO_SNAPSHOTS), _MimoConfig())


def _adapter_state_args(**overrides) -> Namespace:
    return Namespace(
        **{
            "save": "/ckpt",
            "lora_adapter_path": None,
            "debug_disable_optimizer": False,
            "optimizer": "adam",
            "exp_avg_dtype": torch.float32,
            "exp_avg_sq_dtype": torch.float32,
            "main_params_dtype": torch.float32,
            "use_distributed_optimizer": True,
            "use_gloo_process_groups": True,
            "use_precision_aware_optimizer": False,
            "optimizer_cpu_offload": False,
            "overlap_param_gather": False,
            "num_distributed_optimizer_instances": 1,
        }
        | overrides
    )


@pytest.mark.parametrize("optimizer", ["sgd", "muon", "dist_muon", "lion"])
def test_an_optimizer_other_than_adam_cannot_save_the_adapter(optimizer) -> None:
    with pytest.raises(AssertionError, match=f"store Adam state; got --optimizer {optimizer}"):
        _validate_adapter_state_args(_adapter_state_args(optimizer=optimizer))


def test_the_adapter_checkpoint_requires_the_distributed_optimizer():
    with pytest.raises(AssertionError, match="distributed optimizer"):
        _validate_adapter_state_args(_adapter_state_args(use_distributed_optimizer=False))


@pytest.mark.parametrize("train_backend, megatron_to_hf_mode", [("torchtitan", "bridge")], ids=["torchtitan"])
def test_convert_save_is_refused_outside_megatron_single_lora(tmp_path, train_backend, megatron_to_hf_mode):
    args = Namespace(
        lora_type="lora",
        multi_lora_n_adapters=0,
        lora_rank=8,
        lora_alpha=16.0,
        lora_adapter_path=str(tmp_path),
        lora_adapter_convert_save=str(tmp_path / "converted"),
        train_backend=train_backend,
        megatron_to_hf_mode=megatron_to_hf_mode,
        debug_rollout_only=False,
    )
    with pytest.raises(AssertionError, match="--lora-adapter-convert-save is implemented only"):
        validate_lora_args(args)


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
        debug_rollout_only=False,
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
        debug_rollout_only=False,
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


def test_a_relative_adapter_path_is_made_absolute_before_alpha_is_resolved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "source").mkdir()
    (tmp_path / "source" / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32}))
    args = Namespace(
        lora_type="lora",
        multi_lora_n_adapters=0,
        lora_rank=8,
        lora_alpha=None,
        lora_adapter_path="source",
        lora_adapter_convert_save=str(tmp_path / "converted"),
        train_backend="megatron",
        megatron_to_hf_mode="bridge",
        indep_dp=True,
        debug_rollout_only=False,
    )
    # --indep-dp stops validation right after alpha is resolved
    with pytest.raises(AssertionError, match="does not support --indep-dp"):
        validate_lora_args(args)
    assert args.lora_adapter_path == os.path.join(os.getcwd(), "source")
    assert args.lora_alpha == 32


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


@pytest.mark.parametrize(
    "fault_tolerance", [dict(indep_dp=True), dict(non_persistent_ckpt_type="local")], ids=["peer_heal", "in_memory"]
)
def test_lora_refuses_the_checkpoints_that_cannot_carry_its_adapter(fault_tolerance):
    """save() writes a LoRA adapter to its adapter dir: a healed cell or an in-memory rollback would lose it."""
    args = Namespace(
        lora_rank=8,
        multi_lora=False,
        megatron_to_hf_mode="raw",
        debug_rollout_only=False,
        **{"indep_dp": False, "non_persistent_ckpt_type": None, **fault_tolerance},
    )
    with pytest.raises(AssertionError, match="remove 'train' from --ft-components"):
        validate_lora_args_after_megatron(args)


def test_the_adapter_checkpoint_requires_fp32_masters():
    with pytest.raises(AssertionError, match="fp32 masters"):
        _validate_adapter_state_args(_adapter_state_args(main_params_dtype=torch.float16))


def test_loading_the_adapter_checkpoint_refuses_several_distributed_optimizer_instances():
    args = _adapter_state_args(save=None, lora_adapter_path="/adapter", num_distributed_optimizer_instances=2)
    with pytest.raises(AssertionError, match="--num-distributed-optimizer-instances"):
        _validate_adapter_state_args(args)


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


def _bridge_args_without_layout(monkeypatch, *, multi_lora=False, **overrides):
    hf_config = SimpleNamespace(model_type="gpt2", to_dict=lambda: {"model_type": "gpt2"})
    monkeypatch.setattr(lora_arguments, "load_hf_config", lambda path: hf_config)
    monkeypatch.setattr(
        lora_arguments, "validate_multi_lora_args", lambda args: setattr(args, "multi_lora", multi_lora)
    )
    monkeypatch.setattr(lora_arguments, "_resolve_lora_targets", lambda args, config: ["h.0.attn.c_attn"])
    monkeypatch.setattr(
        "miles.backends.megatron_utils.lora.target_modules.check_fused_lora_selection", lambda *a, **k: None
    )
    fields = dict(
        train_backend="megatron",
        megatron_to_hf_mode="bridge",
        hf_checkpoint="/hf",
        lora_rank=8,
        lora_alpha=16.0,
        lora_adapter_path=None,
        lora_type="lora",
        experts_shared_outer_loras=False,
        lora_adapter_convert_save=None,
        debug_rollout_only=False,
        save_hf=None,
        eval_hf_dir=None,
        eval_uses_snapshots=False,
        save=None,
        indep_dp=False,
        non_persistent_ckpt_type=None,
    )
    return Namespace(**(fields | overrides))


def _validate(args):
    lora_arguments.validate_lora_args(args)
    lora_arguments.validate_lora_args_after_megatron(args)


def test_only_runs_that_read_the_adapter_slots_need_the_hf_lora_layout(monkeypatch):
    """The adapter slots, which need the layout, are built for HF snapshot exports and the --save adapter dir."""
    monkeypatch.setattr(lora_arguments, "_validate_adapter_state_args", lambda args: None)
    _validate(_bridge_args_without_layout(monkeypatch))
    for reader in (dict(save_hf="/hf_out"), dict(save="/save")):
        with pytest.raises(AssertionError, match="HF LoRA target layout"):
            _validate(_bridge_args_without_layout(monkeypatch, **reader))


def test_an_eval_hf_dir_without_eval_snapshots_needs_no_layout(monkeypatch):
    """Only eval snapshots export to --eval-hf-dir; the actor publishes nothing for it otherwise."""
    _validate(_bridge_args_without_layout(monkeypatch, eval_hf_dir="/eval"))
    with pytest.raises(AssertionError, match="HF LoRA target layout"):
        _validate(_bridge_args_without_layout(monkeypatch, eval_hf_dir="/eval", eval_uses_snapshots=True))


def test_multi_lora_bridge_refuses_a_model_without_hf_lora_layout(monkeypatch):
    """Multi-LoRA builds the adapter slots its sampler exports read, even without --save-hf / --save."""
    with pytest.raises(AssertionError, match="HF LoRA target layout"):
        _validate(_bridge_args_without_layout(monkeypatch, multi_lora=True))
