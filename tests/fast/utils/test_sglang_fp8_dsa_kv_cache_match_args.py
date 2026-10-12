"""Launch check for --match-sglang-fp8-dsa-kv-cache-values: only sampler caches it reproduces are accepted."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import argparse
from argparse import Namespace

import pytest

from miles.utils.arguments import get_miles_extra_args_provider, validate_match_sglang_fp8_dsa_kv_cache_values_args


def _glm53_args(**overrides) -> Namespace:
    """Trainer and sampler settings of the GLM-5.3 W4A16 recipe."""
    values = dict(
        match_sglang_fp8_dsa_kv_cache_values=True,
        train_backend="megatron",
        dsa_impl="megatron",
        rollout_external=False,
        sglang_config=None,
        sglang_dtype="bfloat16",
        sglang_kv_cache_dtype="fp8_e4m3",
        sglang_attention_backend="dsa",
        sglang_prefill_attention_backend=None,
        sglang_decode_attention_backend=None,
        sglang_dsa_prefill_backend="flashmla_sparse",
        sglang_dsa_decode_backend="flashmla_kv",
        bf16=True,
    )
    return Namespace(**(values | overrides))


def test_flag_defaults_off_and_can_be_enabled():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    required = ["--rollout-batch-size", "64"]
    assert parser.parse_args(required).match_sglang_fp8_dsa_kv_cache_values is False
    assert (
        parser.parse_args([*required, "--match-sglang-fp8-dsa-kv-cache-values"]).match_sglang_fp8_dsa_kv_cache_values
        is True
    )


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="glm53-recipe"),
        pytest.param({"sglang_attention_backend": "nsa"}, id="nsa-alias"),
        pytest.param(
            {
                "sglang_attention_backend": None,
                "sglang_prefill_attention_backend": "dsa",
                "sglang_decode_attention_backend": "dsa",
            },
            id="split-dsa-backends",
        ),
        pytest.param({"sglang_dsa_prefill_backend": "flashmla_kv"}, id="flashmla-kv-prefill"),
    ],
)
def test_accepts_samplers_storing_the_scaled_fp8_cache(overrides):
    validate_match_sglang_fp8_dsa_kv_cache_values_args(_glm53_args(**overrides))


def test_flag_off_checks_nothing():
    validate_match_sglang_fp8_dsa_kv_cache_values_args(
        _glm53_args(match_sglang_fp8_dsa_kv_cache_values=False, train_backend="fsdp", sglang_kv_cache_dtype="auto")
    )


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"train_backend": "fsdp"}, id="fsdp-trainer"),
        pytest.param({"dsa_impl": "miles"}, id="miles-dsa-attention"),
        pytest.param({"rollout_external": True}, id="external-engine"),
        pytest.param({"sglang_config": "/tmp/groups.yaml"}, id="per-group-overrides"),
        pytest.param({"sglang_dtype": "auto"}, id="auto-sampler-dtype"),
        pytest.param({"sglang_dtype": "float16"}, id="fp16-sampler"),
        pytest.param({"sglang_kv_cache_dtype": "auto"}, id="bf16-cache"),
        pytest.param({"sglang_kv_cache_dtype": "fp8_e5m2"}, id="e5m2-cache"),
        pytest.param({"sglang_attention_backend": None}, id="auto-attention-backend"),
        pytest.param({"sglang_prefill_attention_backend": "fa3"}, id="split-prefill-backend"),
        pytest.param({"sglang_decode_attention_backend": "fa3"}, id="split-decode-backend"),
        pytest.param({"sglang_dsa_prefill_backend": None}, id="auto-prefill-backend"),
        pytest.param({"sglang_dsa_prefill_backend": "trtllm"}, id="trtllm-cache-layout"),
        pytest.param({"sglang_dsa_prefill_backend": "flashmla_sparse_q8"}, id="requantizing-prefill"),
        pytest.param({"sglang_dsa_decode_backend": None}, id="auto-decode-backend"),
        pytest.param({"sglang_dsa_decode_backend": "trtllm"}, id="trtllm-decode"),
        pytest.param({"bf16": False}, id="non-bf16-trainer"),
    ],
)
def test_rejects_samplers_whose_cache_it_does_not_reproduce(overrides):
    with pytest.raises(ValueError):
        validate_match_sglang_fp8_dsa_kv_cache_values_args(_glm53_args(**overrides))
