import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from tests.fast.fixtures.args_fixtures import make_trainer_config
from tests.fast.miles_plugins.models.qwen3_8_next.conftest import gdn

from miles_plugins.models import glm4

__all__ = ["gdn"]


class TestBackendOwnedSpecs:
    @pytest.mark.parametrize("enabled", [False, True])
    def test_glm_spec_passes_all_six_backend_settings_to_the_sdk(
        self, monkeypatch: pytest.MonkeyPatch, enabled: bool
    ) -> None:
        """GLM's real spec entrypoint forwards each backend-owned SDK setting."""
        settings = dict(
            num_experts=4 if enabled else None,
            moe_grouped_gemm=enabled,
            qk_layernorm=not enabled,
            multi_latent_attention=enabled,
            post_self_attn_layernorm=not enabled,
            post_mlp_layernorm=enabled,
        )
        args = make_trainer_config(**settings)
        original = args.model_dump()
        calls = []
        result = object()

        def sdk_spec(**kwargs: Any) -> object:
            calls.append(kwargs)
            return result

        monkeypatch.setattr(glm4, "get_gpt_layer_with_transformer_engine_spec", sdk_spec)

        assert glm4.get_glm_spec(args=args, config=object(), vp_stage=2) is result
        assert calls == [settings]
        assert args.model_dump() == original

    @pytest.mark.parametrize("family", ["qwen3_5", "qwen3_next", "qwen3_8_next.qwen3_8_next"])
    @pytest.mark.parametrize("num_experts", [None, 4])
    def test_qwen_spec_uses_backend_expert_count_before_building_the_real_layer_plan(
        self, monkeypatch: pytest.MonkeyPatch, family: str, num_experts: int | None, gdn: ModuleType
    ) -> None:
        """Dense Qwen models reset the MoE cadence before SDK block construction."""
        module = importlib.import_module(f"miles_plugins.models.{family}")
        args = make_trainer_config(num_experts=num_experts, hf_checkpoint="/qwen-checkpoint")
        original = args.model_dump()
        config = SimpleNamespace(num_layers=2, moe_layer_freq=[1, 0], pipeline_model_parallel_layout=None)
        text = SimpleNamespace(
            layer_types=["linear_attention", "full_attention"],
            hc_count=4,
            hc_lowrank=2,
            ple_layer_ids=[],
            ple_embed_dim=8,
            ngram_size=2,
            heads_per_ngram=1,
            ngram_vocab_size_base=17,
            split_ngram_parts=1,
            ple_conv_kernel_size=3,
            eos_token_id=7,
            indexer_budget=11,
            indexer_compress_ratio=2,
            indexer_n_heads=2,
            indexer_head_dim=4,
            indexer_kv_heads=1,
        )
        original_attention = SimpleNamespace(params={"marker": 13}, submodules=SimpleNamespace())
        layers = [SimpleNamespace(submodules=SimpleNamespace(self_attention=original_attention)) for _ in range(2)]
        block = SimpleNamespace(layer_specs=layers)
        observed = []

        def sdk_block(received: Any, **kwargs: Any) -> SimpleNamespace:
            assert received is config
            observed.append((list(received.moe_layer_freq), kwargs))
            return block

        monkeypatch.setattr(module, "get_gpt_decoder_block_spec", sdk_block)
        monkeypatch.setattr(module, "get_num_layers_to_build", lambda config, vp_stage: 2)
        monkeypatch.setattr(module, "get_transformer_layer_offset", lambda config, vp_stage: 0)
        if family.startswith("qwen3_8"):
            monkeypatch.setattr(module, "load_hf_config", lambda path: text)
        else:
            monkeypatch.setattr(module.AutoConfig, "from_pretrained", lambda path, **kwargs: text)

        result = (
            module.get_qwen3_8_next_spec(args=args, config=config, vp_stage=3)
            if family.startswith("qwen3_8")
            else (
                module.get_qwen3_5_spec(args=args, config=config, vp_stage=3)
                if family == "qwen3_5"
                else module.get_qwen3_next_spec(args=args, config=config, vp_stage=3)
            )
        )

        assert result is block
        assert observed == [
            ([0, 0] if num_experts is None else [1, 0], {"use_transformer_engine": True, "vp_stage": 3})
        ]
        assert result.layer_specs[0].submodules.self_attention.params == {"args": args}
        assert result.layer_specs[0].submodules.self_attention.module is (
            module.Qwen38NextLinearAttention if family.startswith("qwen3_8") else module.Attention
        )
        assert args.model_dump() == original


@pytest.mark.parametrize("family", ["qwen3_5", "qwen3_next"])
@pytest.mark.parametrize("backend", ["megatron", "cake"])
def test_the_qwen_full_attention_layers_take_the_cake_core_attention(
    monkeypatch: pytest.MonkeyPatch, family: str, backend: str, gdn: ModuleType
) -> None:
    """--full-attention-backend cake swaps the full-attention layer specs (never the GDN ones) through
    apply_cake_full_attention; the default leaves them as Megatron built them."""
    module = importlib.import_module(f"miles_plugins.models.{family}")
    args = make_trainer_config(full_attention_backend=backend, num_experts=None, hf_checkpoint="/qwen-checkpoint")
    config = SimpleNamespace(num_layers=2, moe_layer_freq=[0, 0], pipeline_model_parallel_layout=None)
    text = SimpleNamespace(layer_types=["linear_attention", "full_attention"], num_hidden_layers=2)
    full_attention = SimpleNamespace(submodules=SimpleNamespace(self_attention="TE attention"))
    block = SimpleNamespace(
        layer_specs=[SimpleNamespace(submodules=SimpleNamespace(self_attention="TE attention")), full_attention]
    )
    cake = ModuleType("miles_plugins.models.cake_full_attention")
    cake.apply_cake_full_attention = lambda spec, args: ("cake", spec, args)
    monkeypatch.setitem(sys.modules, "miles_plugins.models.cake_full_attention", cake)
    monkeypatch.setattr(module, "get_gpt_decoder_block_spec", lambda config, **kwargs: block)
    monkeypatch.setattr(module, "get_num_layers_to_build", lambda config, vp_stage: 2)
    monkeypatch.setattr(module, "get_transformer_layer_offset", lambda config, vp_stage: 0)
    monkeypatch.setattr(module.AutoConfig, "from_pretrained", lambda path, **kwargs: text)

    spec_fn = module.get_qwen3_5_spec if family == "qwen3_5" else module.get_qwen3_next_spec
    result = spec_fn(args=args, config=config, vp_stage=None)

    assert result.layer_specs[0].submodules.self_attention.module is module.Attention
    if backend == "cake":
        tag, swapped, passed = result.layer_specs[1]
        assert tag == "cake" and swapped is full_attention and passed is args
    else:
        assert result.layer_specs[1] is full_attention


def _fake_rmsnorm_ext(monkeypatch: pytest.MonkeyPatch, calls: list[tuple]) -> type[torch.nn.Module]:
    """Stand-ins for ``miles_ext_kernels`` (its ``configure``), ``miles_ext_kernels.cake`` (its ``prebuild``) and the
    op module ``miles_ext_kernels.cake.rmsnorm`` (its ``ZeroCenteredRMSNorm``), each recording its calls."""

    class ZeroCenteredRMSNorm(torch.nn.Module):
        def __init__(
            self, hidden_size: int, eps: float, *, params_dtype: torch.dtype, sequence_parallel: bool | None = None
        ) -> None:
            super().__init__()
            calls.append(("build", hidden_size, eps, params_dtype, sequence_parallel))
            self.weight = torch.nn.Parameter(torch.zeros(hidden_size, dtype=params_dtype))

    package = ModuleType("miles_ext_kernels")
    package.configure = lambda values: calls.append(("configure", values))
    cake = ModuleType("miles_ext_kernels.cake")
    cake.prebuild = lambda op, **params: calls.append(("prebuild", op, params))
    rmsnorm = ModuleType("miles_ext_kernels.cake.rmsnorm")
    rmsnorm.ZeroCenteredRMSNorm = ZeroCenteredRMSNorm
    package.cake, cake.rmsnorm = cake, rmsnorm
    for module in (package, cake, rmsnorm):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return ZeroCenteredRMSNorm


@pytest.mark.parametrize("sequence_parallel", [None, True])
def test_a_cake_gdn_input_norm_is_the_kernel_package_norm(
    monkeypatch: pytest.MonkeyPatch, gdn: ModuleType, sequence_parallel: bool | None
) -> None:
    """--linear-attention-input-norm cake: the Qwen3.5 / Qwen3-Next GDN input norm is the package's zero-centred
    RMSNorm, configured with the run's values, built with the layer's sequence-parallel setting and holding the single
    weight parameter of the te and hf norms; its libraries for the hidden size are built with it."""
    qwen3_5 = importlib.import_module("miles_plugins.models.qwen3_5")
    calls: list[tuple] = []
    norm_cls = _fake_rmsnorm_ext(monkeypatch, calls)
    args = SimpleNamespace(miles_ext_kernels_args={"cake_rmsnorm_variant": "det"})

    norm = qwen3_5.gdn_input_layernorm(
        "cake", 4096, 1e-6, torch.bfloat16, args=args, sequence_parallel=sequence_parallel
    )

    assert isinstance(norm, norm_cls)
    assert list(norm.state_dict()) == ["weight"]
    assert calls == [
        ("configure", {"cake_rmsnorm_variant": "det"}),
        ("build", 4096, 1e-6, torch.bfloat16, sequence_parallel),
        ("prebuild", "rmsnorm", {"hidden_size": 4096}),
    ]


def test_the_te_gdn_input_norm_is_built_with_the_sequence_parallel_setting(
    monkeypatch: pytest.MonkeyPatch, gdn: ModuleType
) -> None:
    qwen3_5 = importlib.import_module("miles_plugins.models.qwen3_5")
    built: list[dict[str, Any]] = []
    monkeypatch.setattr(qwen3_5, "_TEZeroCenteredRMSNorm", lambda hidden_size, **kwargs: built.append(kwargs))

    qwen3_5.gdn_input_layernorm("te", 4096, 1e-6, torch.bfloat16, sequence_parallel=True)

    assert built == [dict(eps=1e-6, zero_centered_gamma=True, params_dtype=torch.bfloat16, sequence_parallel=True)]


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_the_gdn_layer_builds_its_input_norm_with_its_sequence_parallel_setting(
    monkeypatch: pytest.MonkeyPatch, gdn: ModuleType, sequence_parallel: bool
) -> None:
    """The Qwen3.5 GDN layer hands ``config.sequence_parallel`` to its input norm's constructor."""
    qwen3_5 = importlib.import_module("miles_plugins.models.qwen3_5")
    norms: list[dict[str, Any]] = []
    text_config = SimpleNamespace(linear_conv_kernel_dim=4, rms_norm_eps=1e-6, hidden_act="silu", hidden_size=4096)
    monkeypatch.setattr(qwen3_5, "load_hf_config", lambda path: text_config)
    monkeypatch.setattr(qwen3_5, "gdn_heads", lambda config: None)
    monkeypatch.setattr(qwen3_5.Attention, "core_cls", lambda *args, **kwargs: None)
    monkeypatch.setattr(qwen3_5, "gdn_input_layernorm", lambda *args, **kwargs: norms.append(kwargs))
    monkeypatch.setattr(qwen3_5.LinearAttentionLayer, "__init__", lambda self, *args, **kwargs: None)
    args = SimpleNamespace(
        hf_checkpoint="checkpoint",
        linear_attention_backend="fla",
        linear_attention_input_norm="cake",
        allgather_cp=False,
    )
    config = SimpleNamespace(sequence_parallel=sequence_parallel, params_dtype=torch.bfloat16)

    qwen3_5.Attention(args, config, layer_number=1, pg_collection=SimpleNamespace(tp=None))

    assert norms == [{"args": args, "sequence_parallel": sequence_parallel}]


def test_a_cake_gdn_input_norm_built_on_the_meta_device_keeps_its_sequence_parallel_mark(
    monkeypatch: pytest.MonkeyPatch, gdn: ModuleType
) -> None:
    """The real norm, built on the meta device with ``sequence_parallel=True``: after ``to_empty()`` and
    ``reset_parameters()``, Megatron's rule for sequence-parallel gradients still selects its weight."""
    pytest.importorskip("miles_ext_kernels.cake.rmsnorm", reason="the miles-ext-kernels submodule is not checked out")
    from miles_ext_kernels import cake

    qwen3_5 = importlib.import_module("miles_plugins.models.qwen3_5")
    monkeypatch.setattr(cake, "prebuild", lambda op, **params: [])  # builds for a GPU, which this test has none of
    with torch.device("meta"):
        norm = qwen3_5.gdn_input_layernorm("cake", 4096, 1e-6, torch.bfloat16, sequence_parallel=True)
    norm.to_empty(device="cpu")
    norm.reset_parameters()

    marked = [name for name, param in norm.named_parameters() if getattr(param, "sequence_parallel", False)]
    assert marked == ["weight"]
