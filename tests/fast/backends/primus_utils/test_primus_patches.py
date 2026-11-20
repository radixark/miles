import argparse
import sys
import textwrap

import pytest

PRIMUS_YAML = """
work_group: test
user_name: test
exp_name: primus_patches_test
workspace: {workspace}

modules:
  pre_trainer:
    framework: megatron
    config: pre_trainer.yaml
    model: llama3.1_8B.yaml
    overrides:
      train_iters: 50
      enable_primus_turbo: true
"""


@pytest.fixture(autouse=True)
def _require_primus():
    pytest.importorskip("primus.core.patches")
    pytest.importorskip("megatron.training.arguments")


class _Parsed(Exception):
    pass


def _parse(monkeypatch, tmp_path, extra_argv):
    """The argparse namespace as the bridge leaves it, before Miles validates and freezes it."""
    import miles.utils.arguments as miles_arguments

    def _capture(args):
        raise _Parsed(args)

    monkeypatch.setattr(miles_arguments, "miles_validate_args", _capture)

    config = tmp_path / "primus_exp.yaml"
    config.write_text(textwrap.dedent(PRIMUS_YAML).format(workspace=tmp_path / "output"))

    argv = [
        "pytest",
        "--train-backend",
        "megatron",
        "--rollout-batch-size",
        "1",
        "--primus-config",
        str(config),
    ] + extra_argv
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(_Parsed) as parsed:
        miles_arguments.parse_args()
    return parsed.value.args[0]


@pytest.fixture
def restore_megatron_globals():
    """Patches rewrite module-level Megatron state, so undo it for the rest of the suite."""
    from megatron.core.extensions import transformer_engine_spec_provider
    from megatron.core.models.gpt import gpt_layer_specs, moe_module_specs
    from megatron.core.transformer import multi_token_prediction

    modules = [transformer_engine_spec_provider, gpt_layer_specs, moe_module_specs, multi_token_prediction]
    # Megatron only binds TESpecProvider when Transformer Engine imports, and other tests
    # in the suite can leave TE stubbed out. There is nothing to swap in that case.
    if not all(hasattr(module, "TESpecProvider") for module in modules):
        pytest.skip("Transformer Engine is unavailable here, so Megatron exposes no TESpecProvider")

    saved = [(module, module.TESpecProvider) for module in modules]
    yield
    for module, spec_provider in saved:
        module.TESpecProvider = spec_provider
    from primus.backends.megatron.training import global_vars

    global_vars.destroy_global_vars()


def test_compat_shim_provides_grouped_mlp_submodules():
    from megatron.core.transformer.moe import experts

    from miles.backends.primus_utils.megatron_compat import install_megatron_compat_shims

    install_megatron_compat_shims()

    # Either Megatron defines it or the shim aliases MLPSubmodules; Primus imports it eagerly.
    assert hasattr(experts, "TEGroupedMLPSubmodules")
    submodules = experts.TEGroupedMLPSubmodules(linear_fc1=object, linear_fc2=object)
    assert submodules.linear_fc1 is not None and submodules.linear_fc2 is not None
    # Already-installed shims are not reported a second time.
    assert install_megatron_compat_shims() == []


def test_turbo_spec_provider_builds_megatron_moe_spec(monkeypatch, tmp_path, restore_megatron_globals):
    """Megatron's own spec builder must accept the Primus provider and pick Turbo grouped GEMM."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

    from miles.backends.primus_utils.patches import PrimusPatchRunner

    argv = ["--primus-override", "use_turbo_grouped_gemm=true", "use_turbo_rms_norm=true"]
    runner = PrimusPatchRunner.create_if_enabled(_parse(monkeypatch, tmp_path, argv))
    runner.run("before_train")

    spec = get_gpt_layer_with_transformer_engine_spec(num_experts=8, moe_grouped_gemm=True, qk_layernorm=True)
    mlp = spec.submodules.mlp
    mlp_submodules = mlp.keywords["submodules"] if hasattr(mlp, "keywords") else mlp.submodules
    experts = mlp_submodules.experts
    assert experts.func.__name__ == "PrimusGroupedMLP"
    assert experts.keywords["submodules"].linear_fc1.__name__ == "PrimusTurboColumnParallelGroupedLinear"
    assert experts.keywords["submodules"].linear_fc2.__name__ == "PrimusTurboRowParallelGroupedLinear"


def test_primus_topk_router_is_registered_for_hf_bridge():
    from megatron.bridge.models.conversion.param_mapping import AutoMapping

    from miles.backends.primus_utils.patches import register_primus_bridge_module_types

    register_primus_bridge_module_types()
    assert "PrimusTopKRouter" in AutoMapping._MODULE_TYPE_REGISTRY["replicated"]
    assert "PrimusTurboRMSNorm" in AutoMapping._MODULE_TYPE_REGISTRY["replicated"]
    assert "PrimusTurboColumnParallelGroupedLinear" in AutoMapping._MODULE_TYPE_REGISTRY["column"]
    assert "PrimusTurboRowParallelGroupedLinear" in AutoMapping._MODULE_TYPE_REGISTRY["row"]


def test_hf_bridge_leaves_packed_expert_storage_out_of_conversion():
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

    from miles.backends.primus_utils.patches import install_primus_hf_bridge_fixes

    install_primus_hf_bridge_fixes()
    is_skipped = MegatronModelBridge._is_adapter_param_name
    assert is_skipped(None, "decoder.layers.0.mlp.experts.linear_fc1.weights")
    assert not is_skipped(None, "decoder.layers.0.mlp.experts.linear_fc1.weight3")
    assert is_skipped(None, "decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight")


def test_turbo_grouped_linear_packed_weights_are_not_expert_indexed():
    from miles.backends.primus_utils.patches import _is_expert_indexed_param

    assert _is_expert_indexed_param("decoder.layers.0.mlp.experts.linear_fc1.weight0")
    assert _is_expert_indexed_param("decoder.layers.0.mlp.experts.linear_fc2.bias3")
    assert not _is_expert_indexed_param("decoder.layers.0.mlp.experts.linear_fc1.weights")
    assert not _is_expert_indexed_param("decoder.layers.0.self_attention.linear_qkv.weight")


def test_turbo_spec_provider_is_installed(monkeypatch, tmp_path, restore_megatron_globals):
    """The spec provider is how every Primus-Turbo kernel reaches the model."""
    from megatron.core.models.gpt import gpt_layer_specs

    from miles.backends.primus_utils.patches import PrimusPatchRunner

    runner = PrimusPatchRunner.create_if_enabled(_parse(monkeypatch, tmp_path, []))
    assert "megatron.turbo.te_spec_provider" in runner.inventory()["before_train"]
    runner.run("before_train")

    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        PrimusTurboSpecProvider,
    )

    assert gpt_layer_specs.TESpecProvider is PrimusTurboSpecProvider


def test_turbo_gemm_reaches_the_model_spec(monkeypatch, tmp_path, restore_megatron_globals):
    """Turbo GEMM is the kernel path Miles enables; it must reach the spec."""
    from miles.backends.primus_utils.patches import PrimusPatchRunner

    argv = ["--primus-override", "use_turbo_gemm=true"]
    runner = PrimusPatchRunner.create_if_enabled(_parse(monkeypatch, tmp_path, argv))
    runner.run("before_train")

    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        PrimusTurboSpecProvider,
    )

    provider = PrimusTurboSpecProvider()
    assert provider.linear().__name__ == "PrimusTurboLinear"
    assert provider.column_parallel_linear().__name__ == "PrimusTurboColumnParallelLinear"
    assert provider.row_parallel_linear().__name__ == "PrimusTurboRowParallelLinear"


def test_turbo_attention_is_refused(monkeypatch, tmp_path):
    """Its forward is correct and its backward is not, so it must fail loudly at startup."""
    from miles.backends.primus_utils.patches import PrimusPatchRunner

    argv = ["--primus-override", "use_turbo_attention=true"]
    with pytest.raises(RuntimeError, match="use_turbo_attention"):
        PrimusPatchRunner.create_if_enabled(_parse(monkeypatch, tmp_path, argv))


def test_turbo_attention_can_be_forced_for_investigation(monkeypatch, tmp_path, restore_megatron_globals):
    from miles.backends.primus_utils.patches import _ALLOW_UNSAFE_ENV, PrimusPatchRunner

    monkeypatch.setenv(_ALLOW_UNSAFE_ENV, "1")
    argv = ["--primus-override", "use_turbo_attention=true"]
    runner = PrimusPatchRunner.create_if_enabled(_parse(monkeypatch, tmp_path, argv))
    runner.run("before_train")

    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        PrimusTurboSpecProvider,
    )

    assert PrimusTurboSpecProvider().core_attention().__name__ == "PrimusTurboAttention"


def test_turbo_linears_are_refused_under_tensor_parallelism(monkeypatch, tmp_path):
    """Every Primus-Turbo linear asserts tp_size == 1, so catch it before model build."""
    from miles.backends.primus_utils.patches import PrimusPatchRunner

    argv = [
        "--tensor-model-parallel-size",
        "2",
        "--primus-override",
        "use_turbo_gemm=true",
    ]
    with pytest.raises(RuntimeError, match="tensor parallel size == 1"):
        PrimusPatchRunner.create_if_enabled(_parse(monkeypatch, tmp_path, argv))


def test_turbo_linears_are_allowed_without_tensor_parallelism(monkeypatch, tmp_path):
    from miles.backends.primus_utils.patches import PrimusPatchRunner

    argv = ["--tensor-model-parallel-size", "1", "--primus-override", "use_turbo_gemm=true"]
    assert PrimusPatchRunner.create_if_enabled(_parse(monkeypatch, tmp_path, argv)) is not None


def test_params_view_is_bound_into_megatron_globals(monkeypatch, tmp_path, restore_megatron_globals):
    """Primus-Turbo reads its flags off Megatron's args global, not Primus' own."""
    from megatron.training import global_vars

    from miles.backends.primus_utils.patches import PrimusPatchRunner

    args = _parse(monkeypatch, tmp_path, [])
    runner = PrimusPatchRunner.create_if_enabled(args)

    global_vars.set_args(args)
    runner.bind_megatron_global_args()

    bound = global_vars.get_args()
    # A Primus-only name that Miles' Megatron never defines must now resolve.
    assert bound.enable_primus_turbo is True
    assert bound.num_layers == args.num_layers


def test_spec_provider_changes_nothing_until_turbo_is_asked_for(monkeypatch, tmp_path, restore_megatron_globals):
    """Swapping the provider in must be inert while the Turbo flags are off."""
    from miles.backends.primus_utils.patches import PrimusPatchRunner

    runner = PrimusPatchRunner.create_if_enabled(_parse(monkeypatch, tmp_path, []))
    runner.run("before_train")

    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        PrimusTurboSpecProvider,
    )

    provider = PrimusTurboSpecProvider()
    assert provider.core_attention().__name__ == "TEDotProductAttention"
    assert provider.linear().__name__ == "TELinear"
    assert provider.column_parallel_linear().__name__ == "TEColumnParallelLinear"
    assert provider.row_parallel_linear().__name__ == "TERowParallelLinear"


def test_patches_are_off_without_primus_config():
    from miles.backends.primus_utils.patches import PrimusPatchRunner, primus_patches_enabled

    args = argparse.Namespace(primus_config=None, primus_patches="kernels")
    assert primus_patches_enabled(args) is False
    assert PrimusPatchRunner.create_if_enabled(args) is None


def test_patches_disabled_by_selection(monkeypatch, tmp_path):
    from miles.backends.primus_utils.patches import PrimusPatchRunner

    args = _parse(monkeypatch, tmp_path, ["--primus-patches", "none"])
    assert PrimusPatchRunner.create_if_enabled(args) is None


def test_default_selection_is_kernels_only(monkeypatch, tmp_path):
    from miles.backends.primus_utils.patches import PrimusPatchRunner

    runner = PrimusPatchRunner.create_if_enabled(_parse(monkeypatch, tmp_path, []))
    selected = set(runner._enabled_ids)

    assert "megatron.turbo.rms_norm" in selected
    assert "megatron.env.cuda_device_max_connections" in selected
    # Miles owns orchestration: these must never be selected by default.
    for patch_id in [
        "megatron.args.tensorboard_path",
        "megatron.args.wandb_config",
        "megatron.args.checkpoint_path",
        "megatron.checkpoint.save_checkpoint",
        "megatron.training.evaluate",
        "megatron.tokenizer.build_tokenizer_override",
    ]:
        assert patch_id not in selected


def test_incompatible_patches_are_never_selected(monkeypatch, tmp_path):
    from miles.backends.primus_utils.patches import _INCOMPATIBLE_PATCH_IDS, PrimusPatchRunner

    # The list is empty while every known fork gap has a shim; it should stay honoured.
    for selection in ["kernels", "all"]:
        runner = PrimusPatchRunner.create_if_enabled(_parse(monkeypatch, tmp_path, ["--primus-patches", selection]))
        assert not set(runner._enabled_ids) & set(_INCOMPATIBLE_PATCH_IDS)


def test_unknown_patch_id_is_rejected(monkeypatch, tmp_path):
    from miles.backends.primus_utils.patches import PrimusPatchRunner

    with pytest.raises(ValueError, match="unregistered patch id"):
        PrimusPatchRunner.create_if_enabled(
            _parse(monkeypatch, tmp_path, ["--primus-patches", "megatron.not.a.patch"])
        )


def test_selected_patches_apply_cleanly(monkeypatch, tmp_path):
    """The default set must apply without error, since a failed patch aborts the run."""
    from miles.backends.primus_utils.patches import PrimusPatchRunner

    runner = PrimusPatchRunner.create_if_enabled(_parse(monkeypatch, tmp_path, []))
    applied = sum(runner.run(phase) for phase in ("setup", "build_args", "before_train"))
    assert applied > 0


def test_inventory_reports_only_patches_that_would_fire(monkeypatch, tmp_path):
    from miles.backends.primus_utils.patches import PrimusPatchRunner

    runner = PrimusPatchRunner.create_if_enabled(_parse(monkeypatch, tmp_path, []))
    inventory = runner.inventory()

    assert set(inventory) == {"setup", "build_args", "before_train", "after_train"}
    fired = {patch_id for ids in inventory.values() for patch_id in ids}
    assert fired <= set(runner._enabled_ids)
    assert "megatron.env.cuda_device_max_connections" in inventory["setup"]


def test_params_view_spans_both_namespaces(monkeypatch, tmp_path):
    from miles.backends.primus_utils.patches import PrimusPatchRunner

    args = _parse(monkeypatch, tmp_path, [])
    view = PrimusPatchRunner.create_if_enabled(args)._params_view

    # A Megatron argument and a Primus-only parameter both resolve through one namespace.
    assert view.num_layers == args.num_layers
    assert view.enable_primus_turbo is True
    # Miles' own flag keeps its value even though the Primus config also names it.
    assert view.offload == args.offload
    # Writes to Megatron arguments reach the real args object.
    view.tensor_model_parallel_size = 8
    assert args.tensor_model_parallel_size == 8
    with pytest.raises(AttributeError):
        _ = view.definitely_not_a_parameter
