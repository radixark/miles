import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from miles.backends.megatron_utils import model_provider as subject
from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.args.runtime import TrainerConfig


@pytest.mark.parametrize("kind", ["local", "te", "moe", "custom_callable", "custom_object"])
@pytest.mark.parametrize("role,post_process", [("actor", True), ("critic", True), ("critic", False)])
def test_native_provider_builds_from_backend_and_installs_companion(
    monkeypatch: pytest.MonkeyPatch, kind: str, role: str, post_process: bool
) -> None:
    """Native construction forwards backend shape and selects the requested layer consumer."""
    args = _args(kind=kind)
    calls, config = _patch_sdks(monkeypatch=monkeypatch)
    custom_spec = object()

    def custom_factory(received_args: TrainerConfig, received_config: Any, stage: int) -> object:
        assert received_args is args
        assert received_config is config
        assert stage == 3
        return custom_spec

    monkeypatch.setattr(
        subject, "import_module", lambda path: custom_factory if kind == "custom_callable" else custom_spec
    )
    provider = subject.get_model_provider_func(args=args, role=role)
    model = provider(pre_process=False, post_process=post_process, vp_stage=3)

    assert calls["config_args"] is args.backend
    assert model.kwargs == {
        "config": config,
        "transformer_layer_spec": custom_spec if kind.startswith("custom") else kind,
        "vocab_size": 173,
        "max_sequence_length": 257,
        "pre_process": False,
        "post_process": post_process,
        "fp16_lm_cross_entropy": True,
        "parallel_output": True,
        "share_embeddings_and_output_weights": role != "critic",
        "position_embedding_type": "rope",
        "rotary_percent": 0.75,
        "rotary_base": 12345,
        "rope_scaling": True,
        "vp_stage": 3,
    }
    if kind == "local":
        assert calls["local"] == dict(
            num_experts=None,
            moe_grouped_gemm=True,
            qk_layernorm=True,
            multi_latent_attention=True,
            normalization="RMSNorm",
            use_kitchen=True,
            use_kitchen_attention=True,
            kitchen_attention_backend="sdpa",
        )
    elif kind == "te":
        assert calls["te"] == dict(
            num_experts=None, moe_grouped_gemm=True, qk_layernorm=True, multi_latent_attention=True
        )
    elif kind == "moe":
        assert calls["moe"] == dict(use_transformer_engine=False, vp_stage=3)
    assert model.model_companion.chunk_index == 3
    assert model.model_companion.pipeline_rank == 2
    assert model.model_companion.replica_id == (2, 2, 2)
    if role == "critic" and post_process:
        assert isinstance(model.output_layer, subject.LinearForLastLayer)
        assert model.output_layer.weight.shape == (1, 8)
    else:
        assert not hasattr(model, "output_layer")


@pytest.mark.parametrize("replay_enabled", [False, True])
@pytest.mark.parametrize("preserve_supported", [False, True])
def test_mtp_and_fp8_build_preserve_replay_state_and_sdk_context(
    monkeypatch: pytest.MonkeyPatch, replay_enabled: bool, preserve_supported: bool
) -> None:
    """MTP builds outside routing replay and FP8 construction preserves supported initialization values."""
    args = _args(kind="local", mtp_num_layers=2, fp8_param_gather=True)
    args = args.model_copy(update={"use_rollout_routing_replay": True})
    calls, config = _patch_sdks(monkeypatch=monkeypatch)
    monkeypatch.setattr(subject.routing_replay_manager, "enabled", replay_enabled)
    events = []

    def mtp(received_config: Any, layer: str, **kwargs: Any) -> str:
        assert received_config is config
        assert layer == "local"
        assert subject.routing_replay_manager.enabled is False
        assert kwargs == {"use_transformer_engine": False}
        return "mtp"

    @contextmanager
    def fp8_context(enabled: bool, preserve_high_precision_init_val: bool = False) -> Iterator[None]:
        events.append((enabled, preserve_high_precision_init_val))
        yield
        events.append("exit")

    @contextmanager
    def legacy_fp8_context(enabled: bool) -> Iterator[None]:
        events.append((enabled, False))
        yield
        events.append("exit")

    monkeypatch.setattr("megatron.core.models.gpt.gpt_layer_specs.get_gpt_mtp_block_spec", mtp)
    monkeypatch.setattr(
        "transformer_engine.pytorch.fp8_model_init", fp8_context if preserve_supported else legacy_fp8_context
    )
    model = subject.get_model_provider_func(args=args)()

    assert model.kwargs["mtp_block_spec"] == "mtp"
    assert "vp_stage" not in model.kwargs
    assert subject.routing_replay_manager.enabled is replay_enabled
    assert events == [(True, preserve_supported), "exit"]
    assert calls["config_args"] is args.backend


def _args(kind: str, **overrides: Any) -> TrainerConfig:
    values = dict(
        transformer_impl="transformer_engine" if kind == "te" else "local",
        spec="custom.spec" if kind.startswith("custom") else None,
        num_experts=4 if kind == "moe" else None,
        moe_grouped_gemm=True,
        qk_layernorm=True,
        multi_latent_attention=True,
        normalization="RMSNorm",
        fp8_param_gather=False,
        padded_vocab_size=173,
        max_position_embeddings=257,
        fp16_lm_cross_entropy=True,
        untie_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_percent=0.75,
        rotary_base=12345,
        use_rope_scaling=True,
        mtp_num_layers=0,
    )
    values.update(overrides)
    return TrainerConfig.model_construct(
        backend=MegatronArgsNamespace(**values),
        custom_model_provider_path=None,
        megatron_to_hf_mode="raw",
        enable_witness=False,
        dsa_impl="tilelang",
        use_rollout_routing_replay=False,
    )


def _patch_sdks(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], SimpleNamespace]:
    te = ModuleType("transformer_engine")
    te_pytorch = ModuleType("transformer_engine.pytorch")
    te.pytorch = te_pytorch
    te_pytorch.fp8_model_init = None
    monkeypatch.setitem(sys.modules, "transformer_engine", te)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te_pytorch)
    calls: dict[str, Any] = {}
    config = SimpleNamespace(
        hidden_size=8,
        sequence_parallel=False,
        use_kitchen=True,
        use_kitchen_attention=True,
        kitchen_attention_backend="sdpa",
    )

    def convert(backend: MegatronArgsNamespace) -> SimpleNamespace:
        calls["config_args"] = backend
        return config

    class Model(torch.nn.Module):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            self.kwargs = kwargs
            self.config = kwargs["config"]

    def spec(name: str, **kwargs: Any) -> str:
        calls[name] = kwargs
        return name

    monkeypatch.setattr(subject, "core_transformer_config_from_args", convert)
    monkeypatch.setattr(subject, "GPTModel", Model)
    monkeypatch.setattr(subject, "get_gpt_layer_local_spec", lambda **kwargs: spec("local", **kwargs))
    monkeypatch.setattr(subject, "get_gpt_layer_with_transformer_engine_spec", lambda **kwargs: spec("te", **kwargs))
    monkeypatch.setattr(subject, "get_gpt_decoder_block_spec", lambda config, **kwargs: spec("moe", **kwargs))
    group = SimpleNamespace(rank=2)
    monkeypatch.setattr(
        "miles.backends.training_utils.parallel.get_parallel_state",
        lambda: SimpleNamespace(pp=group, tp=group, cp=group, intra_dp=group),
    )
    return calls, config


@pytest.mark.parametrize("untied", [False, True])
def test_actor_embedding_sharing_uses_backend_and_rejects_external_config(
    monkeypatch: pytest.MonkeyPatch, untied: bool
) -> None:
    """Actor sharing follows the backend while caller-supplied config is rejected before conversion."""
    args = _args(kind="local", untie_embeddings_and_output_weights=untied)
    calls, _ = _patch_sdks(monkeypatch=monkeypatch)
    provider = subject.get_model_provider_func(args=args)

    with pytest.raises(AssertionError, match="expects config to be None"):
        provider(config=SimpleNamespace())
    assert calls == {}

    model = provider()

    assert model.kwargs["share_embeddings_and_output_weights"] is not untied
    assert "vp_stage" not in model.kwargs
    assert model.model_companion.chunk_index == 0


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_native_construction_installs_real_witness_buffers(
    monkeypatch: pytest.MonkeyPatch, sequence_parallel: bool
) -> None:
    """Enabled witness construction installs both zeroed buffers with the model sequence layout."""
    args = _args(kind="local").model_copy(update={"enable_witness": True, "witness_buffer_size": 7})
    _, config = _patch_sdks(monkeypatch=monkeypatch)
    config.sequence_parallel = sequence_parallel

    model = subject.get_model_provider_func(args=args)()

    assert model.local_head_witness._sequence_parallel is sequence_parallel
    assert model.local_tail_witness._sequence_parallel is sequence_parallel
    assert model.local_head_witness.witness.weight.shape == (7, 1)
    assert model.local_tail_witness.witness.weight.shape == (7, 1)
    assert torch.count_nonzero(model.local_head_witness.witness.weight) == 0
    assert torch.count_nonzero(model.local_tail_witness.witness.weight) == 0
    assert model.model_companion.chunk_index == 0


def test_fp8_gather_without_sdk_initializer_reports_required_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FP8 parameter gather reports its missing SDK initializer instead of constructing a model."""
    args = _args(kind="local", fp8_param_gather=True)
    _patch_sdks(monkeypatch=monkeypatch)
    monkeypatch.delattr("transformer_engine.pytorch.fp8_model_init")

    with pytest.raises(RuntimeError, match="requires `fp8_model_init`") as error:
        subject.get_model_provider_func(args=args)()

    assert isinstance(error.value.__cause__, ImportError)
