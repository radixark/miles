import json
import os
import subprocess
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig

from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    AdapterParametrization,
    write_parametrization,
)
from miles.utils.lora import arguments as lora_arguments
from miles.utils.lora.arguments import (
    _resolve_lora_alpha,
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


@pytest.mark.parametrize(
    "train_backend, megatron_to_hf_mode", [("torchtitan", "bridge"), ("megatron", "raw")], ids=["torchtitan", "native"]
)
def test_convert_save_is_refused_outside_bridge_single_lora(tmp_path, train_backend, megatron_to_hf_mode):
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


class _ReachedTheModel(Exception):
    """Raised where validation first reads the base model: every path and alpha decision is made by then."""


def _validate_until_the_model(args, monkeypatch):
    def reached(path):
        raise _ReachedTheModel

    monkeypatch.setattr(lora_arguments, "load_hf_config", reached)
    with pytest.raises(_ReachedTheModel):
        validate_lora_args(args)


def _single_lora_args(**overrides):
    args = dict(
        lora_type="lora",
        multi_lora_n_adapters=0,
        lora_rank=8,
        lora_alpha=16.0,
        lora_adapter_path=None,
        lora_adapter_convert_save=None,
        train_backend="megatron",
        megatron_to_hf_mode="bridge",
        indep_dp=False,
        debug_rollout_only=False,
        debug_skip_weight_update=False,
        hf_checkpoint="/hf",
    )
    return Namespace(**(args | overrides))


def test_a_relative_convert_save_is_made_absolute_on_the_driver(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = _single_lora_args(lora_adapter_path=str(tmp_path), lora_adapter_convert_save="converted")
    _validate_until_the_model(args, monkeypatch)
    assert args.lora_adapter_convert_save == os.path.join(os.getcwd(), "converted")


def test_a_relative_adapter_path_is_made_absolute_before_alpha_is_resolved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "source").mkdir()
    (tmp_path / "source" / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32}))
    args = _single_lora_args(lora_alpha=None, lora_adapter_path="source")
    _validate_until_the_model(args, monkeypatch)
    assert args.lora_adapter_path == os.path.join(os.getcwd(), "source")
    assert args.lora_alpha == 32


def test_multi_lora_refuses_an_adapter_path():
    with pytest.raises(AssertionError, match="--lora-adapter-path is single LoRA only"):
        validate_multi_lora_args(Namespace(multi_lora_n_adapters=2, lora_rank=8, lora_adapter_path="/adapter"))


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
    args = Namespace(
        save="/save",
        lora_adapter_path=None,
        debug_disable_optimizer=False,
        optimizer="adam",
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        main_params_dtype=torch.float16,
        use_distributed_optimizer=True,
        use_gloo_process_groups=True,
        use_precision_aware_optimizer=True,
        optimizer_cpu_offload=True,
    )
    with pytest.raises(AssertionError, match="fp32 masters"):
        _validate_adapter_state_args(args)


def test_loading_the_adapter_checkpoint_refuses_several_distributed_optimizer_instances():
    args = Namespace(
        save=None,
        lora_adapter_path="/adapter",
        debug_disable_optimizer=False,
        optimizer="adam",
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        main_params_dtype=torch.float32,
        use_distributed_optimizer=True,
        use_gloo_process_groups=True,
        use_precision_aware_optimizer=False,
        overlap_param_gather=False,
        num_distributed_optimizer_instances=2,
    )
    with pytest.raises(AssertionError, match="--num-distributed-optimizer-instances"):
        _validate_adapter_state_args(args)


def test_without_an_optimizer_an_adapter_path_loads_weights_only():
    args = Namespace(save=None, lora_adapter_path="/adapter", lora_adapter_convert_save=None, debug_disable_optimizer=True)
    _validate_adapter_state_args(args)


@pytest.mark.parametrize("saved", [dict(save="/save"), dict(lora_adapter_convert_save="/converted")])
def test_without_an_optimizer_nothing_that_stores_fp32_masters_is_accepted(saved):
    args = Namespace(
        **{"save": None, "lora_adapter_path": "/adapter", "lora_adapter_convert_save": None, **saved},
        debug_disable_optimizer=True,
    )
    with pytest.raises(AssertionError, match="--debug-disable-optimizer does not build"):
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

    def test_an_adapter_dir_with_no_plain_peft_form_is_refused_without_a_convert_save_remedy(self, tmp_path):
        gdn = "model.layers.0.linear_attn"
        components = tuple(f"{gdn}.{leaf}" for leaf in ("in_proj_q", "in_proj_k", "in_proj_v"))
        write_parametrization(
            tmp_path,
            AdapterParametrization(
                base_model="/hf",
                lora_rank=2,
                lora_alpha=4.0,
                lora_type="canonical_lora",
                target_modules=components,
                experts_shared_outer_loras=False,
            ),
        )
        with pytest.raises(AssertionError, match="no plain-PEFT form") as raised:
            self._validate(tmp_path)
        assert "--lora-adapter-convert-save" not in str(raised.value)


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


# --lora-alpha is the source adapter's when there is one; the flag may only repeat it
def _alpha_args(path, alpha):
    return SimpleNamespace(lora_adapter_path=None if path is None else str(path), lora_alpha=alpha)


def test_without_a_source_the_default_is_16():
    assert _resolve_lora_alpha(_alpha_args(None, None)) == 16.0
    assert _resolve_lora_alpha(_alpha_args(None, 8.0)) == 8.0


def test_a_source_alpha_is_adopted_and_a_different_flag_refused(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 32}))
    assert _resolve_lora_alpha(_alpha_args(tmp_path, None)) == 32
    assert _resolve_lora_alpha(_alpha_args(tmp_path, 32.0)) == 32
    with pytest.raises(AssertionError, match="differs"):
        _resolve_lora_alpha(_alpha_args(tmp_path, 16.0))


def test_a_source_without_alpha_needs_the_flag(tmp_path):
    with pytest.raises(AssertionError, match="pass --lora-alpha"):
        _resolve_lora_alpha(_alpha_args(tmp_path, None))
    assert _resolve_lora_alpha(_alpha_args(tmp_path, 4.0)) == 4.0


def test_a_hub_id_takes_the_flag_or_the_default():
    assert _resolve_lora_alpha(_alpha_args("org/adapter", None)) == 16.0
    assert _resolve_lora_alpha(_alpha_args("org/adapter", 8.0)) == 8.0


_TINY_QWEN3_5_TEXT = dict(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    layer_types=["linear_attention", "full_attention"],
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=8,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=4,
    linear_value_head_dim=4,
)


def _gdn_export_args(monkeypatch, targets, **overrides):
    from transformers import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(**_TINY_QWEN3_5_TEXT)
    monkeypatch.setattr(lora_arguments, "load_hf_config", lambda path: config)
    fields = dict(
        hf_checkpoint="/hf",
        lora_type="canonical_lora",
        hf_lora_targets=targets,
        lora_rank=4,
        lora_alpha=8.0,
        experts_shared_outer_loras=False,
        save_hf="/hf_out",
        eval_hf_dir=None,
        eval_uses_snapshots=False,
        multi_lora=False,
        save=None,
        lora_adapter_path=None,
    )
    return Namespace(**(fields | overrides))


@pytest.mark.parametrize("targets", [["in_proj_qkv"], ["in_proj_*"], ["*.linear_attn.in_proj_*"]])
def test_canonical_gdn_components_refuse_hf_exports_at_launch(monkeypatch, targets):
    """The check resolves the targets to the base's modules, so a wildcard selecting in_proj_qkv is caught too."""
    with pytest.raises(AssertionError, match=r"no PEFT adapter can name: use --lora-type lora"):
        lora_arguments._assert_adapter_layout_defined(_gdn_export_args(monkeypatch, targets))
    # no HF export: nothing to refuse
    lora_arguments._assert_adapter_layout_defined(_gdn_export_args(monkeypatch, targets, save_hf=None))
    # one shared lora_A per in_proj_qkv has a PEFT form
    lora_arguments._assert_adapter_layout_defined(_gdn_export_args(monkeypatch, targets, lora_type="lora"))
    # multi-LoRA reads the slots but writes no PEFT view
    lora_arguments._assert_adapter_layout_defined(
        _gdn_export_args(monkeypatch, targets, save_hf=None, multi_lora=True)
    )


def _gpt_oss_export_args(monkeypatch, **overrides):
    from transformers import GptOssConfig

    config = GptOssConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        layer_types=["sliding_attention", "full_attention"],
    )
    monkeypatch.setattr(lora_arguments, "load_hf_config", lambda path: config)
    fields = dict(
        hf_checkpoint="/hf",
        lora_type="lora",
        hf_lora_targets=["gate_up_proj", "down_proj"],
        lora_rank=4,
        lora_alpha=8.0,
        experts_shared_outer_loras=True,
        save_hf="/hf_out",
        eval_hf_dir=None,
        eval_uses_snapshots=False,
        multi_lora=False,
        save=None,
        lora_adapter_path=None,
    )
    return Namespace(**(fields | overrides))


def test_a_packed_expert_shared_outer_adapter_refuses_hf_exports_at_launch(monkeypatch):
    """Bridge names a packed family's per-expert side once for all experts, so the export keeps only the last."""
    with pytest.raises(AssertionError, match="would keep only the last expert's per-expert side"):
        lora_arguments._assert_adapter_layout_defined(_gpt_oss_export_args(monkeypatch))
    lora_arguments._assert_adapter_layout_defined(_gpt_oss_export_args(monkeypatch, experts_shared_outer_loras=False))
    lora_arguments._assert_adapter_layout_defined(_gpt_oss_export_args(monkeypatch, save_hf=None))
