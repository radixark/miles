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
    megatron_to_hf_mode="raw",
    save_hf=None,
    eval_hf_dir=None,
    eval_uses_snapshots=False,
    save=None,
    lora_rank=8,
    lora_adapter_path=None,
    debug_rollout_only=False,
    multi_lora=False,
    lora_adapter_targets=[],
)


def _check_layout(monkeypatch, args, hf_config):
    monkeypatch.setattr(lora_arguments, "load_hf_config", lambda path: hf_config)
    lora_arguments._assert_adapter_layout_defined(Namespace(**{**vars(args), "hf_checkpoint": "/hf"}))


class _LayoutlessConfig(PretrainedConfig):
    """A family native trains with no HF LoRA layout: no published checkpoint names qwen3_6."""

    model_type = "qwen3_6"


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
    """Native trains qwen3_6, which has no HF LoRA layout; saving or loading its adapter would crash building the
    slots."""
    args = Namespace(**(_NO_SNAPSHOTS | overrides))
    with pytest.raises(AssertionError, match="model_type='qwen3_6', which miles does not define"):
        _check_layout(monkeypatch, args, _LayoutlessConfig())
    _check_layout(monkeypatch, args, AutoConfig.for_model("qwen3"))


@pytest.mark.parametrize(
    "overrides",
    [{}, {"lora_adapter_path": "/adapter", "debug_rollout_only": True}, {"eval_hf_dir": "/eval"}],
    ids=["no_snapshots", "rollout_only", "eval_dir_without_eval_snapshots"],
)
def test_a_run_that_builds_no_slots_trains_any_family(monkeypatch, overrides) -> None:
    """The actor publishes eval snapshots only under eval_uses_snapshots; validation must not refuse the run."""
    _check_layout(monkeypatch, Namespace(**(_NO_SNAPSHOTS | overrides)), _LayoutlessConfig())


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


def test_a_relative_adapter_path_is_made_absolute_before_alpha_is_resolved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "source").mkdir()
    (tmp_path / "source" / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 32}))
    args = _single_lora_args(lora_alpha=None, lora_adapter_path="source")
    _validate_until_the_model(args, monkeypatch)
    assert args.lora_adapter_path == os.path.join(os.getcwd(), "source")
    assert args.lora_alpha == 32


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


def test_lora_refuses_a_load_holding_adapter_checkpoints_without_an_adapter_path(tmp_path):
    """Megatron finds no checkpoint there and would start from the base model, overwriting those adapters."""
    (tmp_path / "iter_0000003" / "adapter").mkdir(parents=True)
    args = Namespace(
        lora_rank=8,
        multi_lora=False,
        megatron_to_hf_mode="raw",
        debug_rollout_only=False,
        indep_dp=False,
        non_persistent_ckpt_type=None,
        load=str(tmp_path),
        lora_adapter_path=None,
    )
    with pytest.raises(AssertionError, match=r"holds LoRA adapter checkpoints \(iter_\*/adapter\)"):
        validate_lora_args_after_megatron(args)


def test_without_an_optimizer_an_adapter_path_loads_weights_only():
    args = Namespace(
        save=None, lora_adapter_path="/adapter", lora_adapter_convert_save=None, debug_disable_optimizer=True
    )
    _validate_adapter_state_args(args)


class TestRolloutOnlyAdapter:
    @pytest.fixture(autouse=True)
    def _qwen3(self, monkeypatch):
        config = SimpleNamespace(to_dict=lambda: {"model_type": "qwen3", "num_hidden_layers": 1})
        monkeypatch.setattr(lora_arguments, "load_hf_config", lambda _: config)

    @staticmethod
    def _validate(path):
        _validate_rollout_only_adapter(Namespace(lora_adapter_path=str(path), hf_checkpoint="/hf"))

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

    def test_a_sampler_export_of_an_adapter_with_no_plain_peft_form_goes_to_sglang(self, tmp_path):
        gdn = "model.layers.0.linear_attn"
        components = tuple(f"{gdn}.{leaf}" for leaf in ("in_proj_q", "in_proj_k", "in_proj_v"))
        parametrization = AdapterParametrization(
            base_model="/hf",
            lora_rank=2,
            lora_alpha=4.0,
            lora_type="canonical_lora",
            target_modules=components,
            experts_shared_outer_loras=False,
        )
        # what write_sampler_adapter commits: sglang-named tensors and the parametrization
        save_file(
            {f"base_model.model.{gdn}.in_proj_q.lora_A.weight": torch.zeros(2, 4)},
            str(tmp_path / "adapter_model.safetensors"),
        )
        (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4.0}))
        write_parametrization(tmp_path, parametrization)
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
        load=None,
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


def test_a_megatron_bridge_source_is_refused_before_a_raw_model_is_built(tmp_path, monkeypatch):
    import torch.distributed.checkpoint as dcp

    key = "decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight"
    dcp.save({key: torch.zeros(2)}, checkpoint_id=tmp_path, no_dist=True)
    (tmp_path / "run_config.yaml").write_text("peft:\n  alpha: 16\n")
    source = dict(lora_adapter_path=str(tmp_path), debug_skip_weight_update=False)
    with pytest.raises(AssertionError, match="only --megatron-to-hf-mode bridge loads"):
        lora_arguments.validate_lora_args(
            _bridge_args_without_layout(monkeypatch, megatron_to_hf_mode="raw", **source)
        )
    lora_arguments.validate_lora_args(_bridge_args_without_layout(monkeypatch, **source))


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
        megatron_to_hf_mode="bridge",
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
        debug_rollout_only=False,
        lora_adapter_path=None,
    )
    return Namespace(**(fields | overrides))


def test_a_packed_expert_shared_outer_adapter_refuses_hf_exports_at_launch(monkeypatch):
    """Bridge names a packed family's per-expert side once for all experts, so the export keeps only the last."""
    with pytest.raises(AssertionError, match="would keep only the last expert's per-expert side"):
        lora_arguments._assert_adapter_layout_defined(_gpt_oss_export_args(monkeypatch))
    lora_arguments._assert_adapter_layout_defined(_gpt_oss_export_args(monkeypatch, experts_shared_outer_loras=False))
    lora_arguments._assert_adapter_layout_defined(_gpt_oss_export_args(monkeypatch, save_hf=None))
