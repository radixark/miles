"""Launch-time recompute guards for multi-LoRA (``validate_multi_lora_args``).

A checkpointed region is replayed grad-enabled only when its input requires
grad. Multi-LoRA trains adapter-only (frozen base), so recompute shapes that
checkpoint the adapters themselves — 'full' granularity always, selective
'moe' with expert-only targets — depend on Megatron-Bridge's PEFT input-grad
patch recognizing multi-LoRA ``.adapters.<slot>.`` params
(radixark/Megatron-Bridge#27, branch bridge @ 688d34b8). On an UNFIXED bridge
those shapes silently zero every adapter gradient (4xH200 GPT-OSS 20B
evidence, 2026-08-12: grad_norm=0.0 on every step, zero trainer logprob
delta) and must be refused at launch; on a FIXED bridge they train real
gradients (4xH200 re-validation on bridge @ 688d34b8) and must pass through.
These tests pin both guard directions, the shapes that never probe the
bridge, and the source probe itself.
"""

import importlib.util
import sys
from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import pytest

import miles.utils.multi_lora as multi_lora_module
from miles.utils.multi_lora import (
    _bridge_recompute_patch_recognizes_multi_lora,
    _recompute_source_recognizes_adapters,
    validate_multi_lora_args,
)


def _args(**overrides) -> SimpleNamespace:
    """Args rich enough to pass validate_multi_lora_args, mirroring
    test_tinker_predicates._full_args."""
    base = dict(
        tinker_backend=True,
        multi_lora_n_adapters=2,
        lora_rank=8,
        target_modules=["linear_qkv"],
        train_backend="megatron",
        pipeline_model_parallel_size=1,
        qkv_format="thd",
        experts_shared_outer_loras=False,
        optimizer="adam",
        colocate=False,
        indep_dp=False,
        ft_components=[],
        offload_train=False,
        enable_witness=False,
        sglang_tokenizer_worker_num=1,
        calculate_per_token_loss=False,
        disable_rollout_trim_samples=False,
        use_dynamic_global_batch_size=False,
        megatron_to_hf_mode="bridge",
        rollout_global_dataset=False,
        recompute_granularity=None,
        recompute_modules=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


EXPERT_TARGETS = ["gate_proj", "up_proj", "down_proj"]

PROBE_NAME = "_bridge_recompute_patch_recognizes_multi_lora"


@pytest.fixture
def unfixed_bridge(monkeypatch):
    """The installed bridge does NOT recognize .adapters. in its recompute patch."""
    monkeypatch.setattr(multi_lora_module, PROBE_NAME, lambda: False)


@pytest.fixture
def fixed_bridge(monkeypatch):
    """The installed bridge DOES recognize .adapters. in its recompute patch."""
    monkeypatch.setattr(multi_lora_module, PROBE_NAME, lambda: True)


@pytest.fixture
def probe_must_not_run(monkeypatch):
    """Supported recompute shapes must never import/probe the bridge at all."""

    def _boom():
        raise AssertionError("bridge probe ran for a recompute shape that never needs it")

    monkeypatch.setattr(multi_lora_module, PROBE_NAME, _boom)


class TestUnfixedBridgeRefusals:
    def test_full_recompute_is_refused_for_any_targets(self, unfixed_bridge):
        validate_multi_lora_args(_args())
        with pytest.raises(AssertionError, match="recompute-granularity full"):
            validate_multi_lora_args(_args(recompute_granularity="full"))

    def test_full_recompute_refusal_points_at_the_bridge_fix_and_selective(self, unfixed_bridge):
        with pytest.raises(AssertionError, match="Megatron-Bridge#27"):
            validate_multi_lora_args(_args(recompute_granularity="full", target_modules=EXPERT_TARGETS))
        with pytest.raises(AssertionError, match="selective"):
            validate_multi_lora_args(_args(recompute_granularity="full", target_modules=EXPERT_TARGETS))

    def test_refusal_happens_at_launch_not_after_gpu_time(self, unfixed_bridge):
        # The guard must live in validate_multi_lora_args (driver launch), not in
        # the trainer: a refused config should never reach model build.
        args = _args(recompute_granularity="full")
        with pytest.raises(AssertionError):
            validate_multi_lora_args(args)

    def test_moe_module_with_expert_targets_is_refused(self, unfixed_bridge):
        with pytest.raises(AssertionError, match="moe_act"):
            validate_multi_lora_args(
                _args(
                    recompute_granularity="selective",
                    recompute_modules=["core_attn", "moe"],
                    target_modules=EXPERT_TARGETS,
                )
            )

    def test_moe_refusal_points_at_the_bridge_fix(self, unfixed_bridge):
        with pytest.raises(AssertionError, match="Megatron-Bridge#27"):
            validate_multi_lora_args(
                _args(
                    recompute_granularity="selective",
                    recompute_modules=["core_attn", "moe"],
                    target_modules=EXPERT_TARGETS,
                )
            )


class TestFixedBridgePassThrough:
    def test_full_recompute_is_allowed(self, fixed_bridge):
        validate_multi_lora_args(_args(recompute_granularity="full"))

    def test_full_recompute_is_allowed_for_expert_targets(self, fixed_bridge):
        validate_multi_lora_args(_args(recompute_granularity="full", target_modules=EXPERT_TARGETS))

    def test_moe_module_with_expert_targets_is_allowed(self, fixed_bridge):
        validate_multi_lora_args(
            _args(
                recompute_granularity="selective",
                recompute_modules=["core_attn", "moe"],
                target_modules=EXPERT_TARGETS,
            )
        )

    def test_pass_through_still_runs_the_rest_of_validation(self, fixed_bridge):
        with pytest.raises(AssertionError, match="qkv-format thd"):
            validate_multi_lora_args(_args(recompute_granularity="full", qkv_format="bshd"))


class TestShapesThatNeverProbeTheBridge:
    def test_no_recompute_is_allowed(self, probe_must_not_run):
        validate_multi_lora_args(_args(target_modules=EXPERT_TARGETS))

    def test_selective_default_modules_is_allowed(self, probe_must_not_run):
        # recompute_modules=None defaults to ['core_attn'] downstream.
        validate_multi_lora_args(_args(recompute_granularity="selective", target_modules=EXPERT_TARGETS))

    def test_selective_core_attn_moe_act_is_allowed_for_expert_targets(self, probe_must_not_run):
        validate_multi_lora_args(
            _args(
                recompute_granularity="selective",
                recompute_modules=["core_attn", "moe_act"],
                target_modules=EXPERT_TARGETS,
            )
        )

    def test_moe_module_without_expert_targets_is_allowed(self, probe_must_not_run):
        # Attention-only adapters sit outside the checkpointed MoE region; 'moe'
        # recompute is then a legitimate memory saver on ANY bridge.
        validate_multi_lora_args(
            _args(
                recompute_granularity="selective",
                recompute_modules=["core_attn", "moe"],
                target_modules=["linear_qkv"],
            )
        )

    def test_absent_recompute_attrs_do_not_break_validation(self, probe_must_not_run):
        args = _args()
        del args.recompute_granularity
        del args.recompute_modules
        validate_multi_lora_args(args)


def _load_module_file(tmp_path, name: str, body: str):
    path = tmp_path / f"{name}.py"
    path.write_text(body)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSourceProbe:
    """The probe inspects the REAL installed function's source: these tests run
    it against file-backed stand-ins for the fixed/unfixed bridge shapes."""

    FIXED_BODY = (
        "def maybe_enable_recompute_inputs_grad(model):\n"
        '    names = ["x.adapter.w", "x.adapters.0.w"]\n'
        '    return any(".adapter." in n or ".adapters." in n for n in names)\n'
    )
    UNFIXED_BODY = (
        "def maybe_enable_recompute_inputs_grad(model):\n"
        '    names = ["x.adapter.w"]\n'
        '    return any(".adapter." in n for n in names)\n'
    )

    def test_fixed_source_is_recognized(self, tmp_path):
        module = _load_module_file(tmp_path, "probe_fixed_bridge_recompute", self.FIXED_BODY)
        assert _recompute_source_recognizes_adapters(module) is True

    def test_unfixed_source_is_not_recognized(self, tmp_path):
        module = _load_module_file(tmp_path, "probe_unfixed_bridge_recompute", self.UNFIXED_BODY)
        assert _recompute_source_recognizes_adapters(module) is False

    def test_module_without_the_patch_function_fails_closed(self, tmp_path):
        module = _load_module_file(tmp_path, "probe_empty_bridge_recompute", "X = 1\n")
        assert _recompute_source_recognizes_adapters(module) is False

    def test_unimportable_bridge_fails_closed(self, monkeypatch):
        # sys.modules[name] = None makes any import of that name raise: the
        # probe must report 'unfixed' rather than crash arg validation.
        monkeypatch.setitem(sys.modules, "megatron.bridge.peft", None)
        monkeypatch.delitem(sys.modules, "megatron.bridge.peft.recompute", raising=False)
        assert _bridge_recompute_patch_recognizes_multi_lora() is False
