from types import SimpleNamespace

import pytest
from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config

from miles.backends.megatron_utils.megatron_to_hf.processors import quantizer_fp8


class TestResolvedFp8ScaleFormat:
    @pytest.mark.parametrize(
        "runner,a2a,expected",
        [
            ("deep_gemm", "none", "ue8m0"),
            ("auto", "deepep", "ue8m0"),
            ("auto", "mooncake", "ue8m0"),
            ("auto", "none", None),
            ("flashinfer_trtllm", "deepep", None),
        ],
    )
    def test_expert_scale_format_uses_resolved_runner_and_communication_backend(
        self, monkeypatch: pytest.MonkeyPatch, runner: str, a2a: str, expected: str | None
    ) -> None:
        """MoE FP8 scales follow resolved SGLang runner ownership without requiring flat CLI fields."""
        monkeypatch.setattr(quantizer_fp8, "should_deepgemm_weight_requant_ue8m0", lambda **kwargs: True)
        args = SimpleNamespace(sglang=make_sglang_config(moe_runner_backend=runner, moe_a2a_backend=a2a))

        result = quantizer_fp8._get_scale_format(args, "model.layers.0.mlp.experts.0.weight", [128, 128])

        assert result == expected

    def test_server_group_override_controls_scale_format_instead_of_the_base_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit server-group runner override controls the format used by its training weight sender."""
        monkeypatch.setattr(quantizer_fp8, "should_deepgemm_weight_requant_ue8m0", lambda **kwargs: True)
        config = make_sglang_config(moe_runner_backend="deep_gemm", moe_a2a_backend="none")
        [model] = config.models
        [group] = model.server_groups
        overridden = group.model_copy(update={"overrides": {"moe_runner_backend": "flashinfer_trtllm"}})
        config = config.model_copy(update={"models": [model.model_copy(update={"server_groups": [overridden]})]})

        result = quantizer_fp8._get_scale_format(
            SimpleNamespace(sglang=config), "model.layers.0.mlp.experts.0.weight", [128, 128]
        )

        assert result is None
        assert config.base_args["moe_runner_backend"] == "deep_gemm"

    def test_inconsistent_group_runners_are_rejected_instead_of_selecting_the_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A weight sender cannot choose one scale protocol for heterogeneous runner requirements."""
        monkeypatch.setattr(quantizer_fp8, "should_deepgemm_weight_requant_ue8m0", lambda **kwargs: True)
        config = make_sglang_config(moe_runner_backend="deep_gemm", moe_a2a_backend="none")
        [model] = config.models
        [first] = model.server_groups
        second = first.model_copy(update={"overrides": {"moe_runner_backend": "auto"}})
        config = config.model_copy(update={"models": [model.model_copy(update={"server_groups": [first, second]})]})

        with pytest.raises(AssertionError, match="moe_runner_backend"):
            quantizer_fp8._get_scale_format(
                SimpleNamespace(sglang=config), "model.layers.0.mlp.experts.0.weight", [128, 128]
            )
