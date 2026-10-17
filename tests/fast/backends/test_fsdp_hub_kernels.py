"""Hub compute kernels for the FSDP backend (CPU-only, no GPU, no network).

`kernels` is stubbed through sys.modules throughout, so these never touch the Hub. What they pin:

  * `--kernel-backend native` (the default) resolves nothing and never imports `kernels`;
  * the numerics guard that keeps hub kernels out of --true-on-policy-mode / --deterministic-mode;
  * strict vs. fallback when a repo or a function is missing from the build;
  * and, the reason this feature exists, that transformers' own Qwen3.5 GatedDeltaNet calls the
    Hub-bound kernels and still hands them the packed-document `seq_idx` / `cu_seqlens` from the
    arch adapter's packing kwargs.
"""

import importlib
import itertools
import sys
import types
from argparse import Namespace

import pytest
import torch
import torch.nn as nn
from tests.fast.fixtures.args_fixtures import make_trainer_config

from miles.backends.fsdp_utils.arguments import validate_kernel_backend_args
from miles.backends.fsdp_utils.plugins.hf_kernels import HubKernels, HubKernelSpec
from miles.backends.fsdp_utils.plugins.hf_kernels.binders import _GDN_KERNEL_SLOTS
from miles.backends.fsdp_utils.plugins.hf_kernels.loader import load_mapping
from miles.backends.fsdp_utils.plugins.hf_kernels.presets import (
    CAUSAL_CONV1D,
    SLOT_CAUSAL_CONV1D,
    SLOT_FLASH_ATTN_VARLEN,
    SLOT_GATED_DELTA_RULE,
)


def _make_raw_args(**overrides) -> Namespace:
    base = dict(
        kernel_backend="hub",
        kernel_mapping_path="",
        kernel_strict=False,
        true_on_policy_mode=False,
        deterministic_mode=False,
    )
    base.update(overrides)
    return Namespace(**base)


def _make_args(**overrides: object) -> Namespace:
    values = vars(_make_raw_args(**overrides))
    return Namespace(true_on_policy_mode=values.pop("true_on_policy_mode"), backend=Namespace(**values))


def _stub_kernels(monkeypatch, *, module=None, modules=None, raises=None, calls=None):
    """Install a fake `kernels` package.

    `module` serves every repo; `modules` maps repo_id -> module so a test can make one repo
    resolve and another 404, which is how the per-slot independence is exercised.
    """

    loaded = []

    def get_kernel(repo_id, revision=None, version=None, user_agent=None):
        if calls is not None:
            calls.append((repo_id, revision, version))
        if raises is not None:
            raise raises
        if modules is not None:
            if repo_id not in modules:
                raise FileNotFoundError(f"no stub build for {repo_id}")
            result = modules[repo_id]
        else:
            result = module
        loaded.append(
            types.SimpleNamespace(
                module=result,
                repo_info=types.SimpleNamespace(repo_id=repo_id, revision=revision or f"commit-v{version}"),
                metadata=types.SimpleNamespace(id=f"build:{repo_id}"),
            )
        )
        return result

    fake = types.ModuleType("kernels")
    fake.get_kernel = get_kernel
    fake.get_loaded_kernels = lambda: loaded
    monkeypatch.setitem(sys.modules, "kernels", fake)


def _kernel_module(**functions):
    mod = types.ModuleType("fake_hub_kernel")
    for name, fn in functions.items():
        setattr(mod, name, fn)
    return mod


def _noop(*args, **kwargs):
    return None


def _hub_function(name):
    """A distinct no-op per name, so a check can tell which Hub function landed where."""

    def function(*args, **kwargs):
        return None

    function.__name__ = name
    return function


def _all_hub_modules(**overrides):
    """One stub module per repo in the shipped mapping, with every advertised function present."""
    modules = {
        repo: _kernel_module(**{name: _hub_function(name) for name in names})
        for repo, names in (
            ("kernels-community/fla", ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule")),
            ("kernels-community/causal-conv1d", ("causal_conv1d_fn", "causal_conv1d_update")),
            ("kernels-community/flash-attn2", ("flash_attn_varlen_func",)),
        )
    }
    modules.update(overrides)
    return modules


# --------------------------------------------------------------------------------------- args


def test_native_is_the_default_and_stays_inert(monkeypatch):
    """A typed FSDP trainer config keeps native kernels independent of Hub dependencies."""
    args = make_trainer_config(train_backend="fsdp", kernel_backend="native")
    monkeypatch.delitem(sys.modules, "kernels", raising=False)

    hub = HubKernels.prepare(args)

    assert hub.bind(nn.Sequential(nn.Linear(4, 4))) == {}
    assert hub.resolve_slot(SLOT_CAUSAL_CONV1D) is None
    # The whole point of the lazy import: a native run must not need `kernels` to exist at all.
    assert "kernels" not in sys.modules


@pytest.mark.parametrize("mode", ["true_on_policy_mode", "deterministic_mode"])
def test_hub_backend_is_rejected_in_the_bit_exact_modes(mode):
    with pytest.raises(ValueError, match="incompatible with --true-on-policy-mode"):
        validate_kernel_backend_args(_make_raw_args(**{mode: True}))


@pytest.mark.parametrize("mode", ["true_on_policy_mode", "deterministic_mode"])
def test_default_mapping_is_empty_in_the_bit_exact_modes(mode):
    """Defence in depth: even if validation is bypassed, the preset hands back nothing to bind."""
    assert load_mapping(_make_args(**{mode: True})) == {}


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="--kernel-backend must be one of"):
        validate_kernel_backend_args(_make_raw_args(kernel_backend="hubb"))


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(kernel_strict=True), "--kernel-strict only applies"),
        (dict(kernel_mapping_path="pkg.mod.fn"), "--kernel-mapping-path only applies"),
    ],
)
def test_hub_only_flags_are_rejected_under_native(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_kernel_backend_args(_make_raw_args(kernel_backend="native", **overrides))


def test_native_and_hub_both_validate_clean():
    validate_kernel_backend_args(_make_raw_args(kernel_backend="native"))
    validate_kernel_backend_args(_make_raw_args(kernel_backend="hub", kernel_strict=True))


# ------------------------------------------------------------------------------------ presets


def test_default_mapping_pins_the_expected_repos():
    mapping = load_mapping(_make_args())

    assert set(mapping) == {SLOT_GATED_DELTA_RULE, SLOT_CAUSAL_CONV1D, SLOT_FLASH_ATTN_VARLEN}
    assert (mapping[SLOT_GATED_DELTA_RULE].repo_id, mapping[SLOT_GATED_DELTA_RULE].version) == (
        "kernels-community/fla",
        1,
    )
    assert (mapping[SLOT_CAUSAL_CONV1D].repo_id, mapping[SLOT_CAUSAL_CONV1D].version) == (
        "kernels-community/causal-conv1d",
        1,
    )
    assert (mapping[SLOT_FLASH_ATTN_VARLEN].repo_id, mapping[SLOT_FLASH_ATTN_VARLEN].version) == (
        "kernels-community/flash-attn2",
        2,
    )
    # Every mapped function is one a binder actually reaches for.
    assert mapping[SLOT_GATED_DELTA_RULE].functions == (
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
    )
    assert "causal_conv1d_fn" in mapping[SLOT_CAUSAL_CONV1D].functions
    assert "flash_attn_varlen_func" in mapping[SLOT_FLASH_ATTN_VARLEN].functions


def _custom_mapping(args):
    return {
        SLOT_CAUSAL_CONV1D: HubKernelSpec(
            repo_id="me/my-conv1d", revision="main", functions=("causal_conv1d_fn", "causal_conv1d_update")
        )
    }


def _conv_only_mapping(args):
    return {SLOT_CAUSAL_CONV1D: CAUSAL_CONV1D}


def _broken_mapping(args):
    return {SLOT_CAUSAL_CONV1D: "kernels-community/causal-conv1d"}


def test_kernel_mapping_path_substitutes_the_whole_mapping():
    args = _make_args(kernel_mapping_path=f"{__name__}._custom_mapping")
    mapping = load_mapping(args)

    assert set(mapping) == {SLOT_CAUSAL_CONV1D}
    assert mapping[SLOT_CAUSAL_CONV1D].repo_id == "me/my-conv1d"


def test_kernel_mapping_path_rejects_a_non_spec_value():
    args = _make_args(kernel_mapping_path=f"{__name__}._broken_mapping")
    with pytest.raises(TypeError, match="must be a HubKernelSpec"):
        load_mapping(args)


def test_spec_rejects_both_pins_and_no_functions():
    with pytest.raises(ValueError, match="not both"):
        HubKernelSpec(repo_id="a/b", version=1, revision="main", functions=("f",))
    with pytest.raises(ValueError, match="at least one function"):
        HubKernelSpec(repo_id="a/b", version=1)


# ---------------------------------------------------------------------------------- resolution


def test_prepare_downloads_once_and_resolve_reads_the_cache(monkeypatch):
    calls = []
    _stub_kernels(
        monkeypatch,
        module=_kernel_module(causal_conv1d_fn=lambda **kw: None, causal_conv1d_update=lambda: None),
        calls=calls,
    )
    hub = HubKernels.prepare(_make_args(kernel_mapping_path=f"{__name__}._conv_only_mapping"))

    first = hub.resolve_slot(SLOT_CAUSAL_CONV1D)
    second = hub.resolve_slot(SLOT_CAUSAL_CONV1D)

    assert set(first) == {"causal_conv1d_fn", "causal_conv1d_update"}
    assert first == second
    # One download for the policy model and the ref model together.
    assert calls == [("kernels-community/causal-conv1d", None, 1)]


def test_unresolvable_repo_falls_back_to_the_native_kernel(monkeypatch, caplog):
    calls = []
    _stub_kernels(monkeypatch, raises=FileNotFoundError("no build for torch2.9-cu130"), calls=calls)

    with caplog.at_level("WARNING"):
        hub = HubKernels.prepare(_make_args(kernel_mapping_path=f"{__name__}._conv_only_mapping"))

    assert "keeping the native kernel" in caplog.text
    # The failure is remembered: re-resolving per model would just re-pay the Hub round trip.
    assert hub.resolve_slot(SLOT_CAUSAL_CONV1D) is None
    assert hub.resolve_slot(SLOT_CAUSAL_CONV1D) is None
    assert len(calls) == 1


def test_unresolvable_repo_raises_under_strict(monkeypatch):
    _stub_kernels(monkeypatch, raises=FileNotFoundError("no build for torch2.9-cu130"))

    with pytest.raises(RuntimeError, match="--kernel-strict"):
        HubKernels.prepare(_make_args(kernel_strict=True))


def test_a_build_missing_the_function_falls_back(monkeypatch, caplog):
    _stub_kernels(monkeypatch, module=_kernel_module(causal_conv1d_fn=lambda: None))  # no causal_conv1d_update

    with caplog.at_level("WARNING"):
        hub = HubKernels.prepare(_make_args(kernel_mapping_path=f"{__name__}._conv_only_mapping"))
    assert hub.resolve_slot(SLOT_CAUSAL_CONV1D) is None
    assert "does not expose callable" in caplog.text

    with pytest.raises(RuntimeError, match="does not expose callable"):
        HubKernels.prepare(_make_args(kernel_mapping_path=f"{__name__}._conv_only_mapping", kernel_strict=True))


def test_prepare_is_inert_under_native(monkeypatch):
    calls = []
    _stub_kernels(monkeypatch, module=_kernel_module(), calls=calls)

    HubKernels.prepare(_make_args(kernel_backend="native"))

    assert calls == []


def test_prepare_warms_every_mapped_repo_once(monkeypatch):
    calls = []
    _stub_kernels(monkeypatch, modules=_all_hub_modules(), calls=calls)

    hub = HubKernels.prepare(_make_args())
    for slot in (SLOT_GATED_DELTA_RULE, SLOT_CAUSAL_CONV1D, SLOT_FLASH_ATTN_VARLEN):
        assert hub.resolve_slot(slot) is not None

    # Downloaded once each by prepare; the per-slot resolves are all cache hits.
    assert sorted(repo for repo, _, _ in calls) == [
        "kernels-community/causal-conv1d",
        "kernels-community/fla",
        "kernels-community/flash-attn2",
    ]


_mapping_provider_calls = []


def _counting_mapping(args):
    _mapping_provider_calls.append(None)
    return {SLOT_CAUSAL_CONV1D: CAUSAL_CONV1D}


def test_binding_uses_the_mapping_agreed_in_prepare(monkeypatch):
    """The provider runs once, in prepare; binding must not re-invoke a possibly drifting provider."""
    _mapping_provider_calls.clear()
    _stub_kernels(monkeypatch, modules=_all_hub_modules())
    hub = HubKernels.prepare(_make_args(kernel_mapping_path=f"{__name__}._counting_mapping"))

    assert hub.bind(_build_gdn_model(monkeypatch)) == {"gated_deltanet": 2}
    assert hub.resolve_slot(SLOT_CAUSAL_CONV1D) is not None
    assert len(_mapping_provider_calls) == 1


# ------------------------------------------------------------------------------ GatedDeltaNet

_fake_modeling_ids = itertools.count()


def _torch_conv_stand_in(x, weight, bias=None, activation=None, **kwargs):
    """transformers' torch `causal_conv1d_fn`: takes no seq_idx, so packed documents bleed into each other."""
    return x


def _torch_chunk_stand_in(query, key=None, value=None, g=None, beta=None, **kwargs):
    """transformers' torch GDN fallback: swallows cu_seqlens and never resets per document."""
    return query


def _build_gdn_model(monkeypatch, native_conv=_torch_conv_stand_in, native_chunk=_torch_chunk_stand_in, n_layers=2):
    """GatedDeltaNet layers that call the kernels of their modeling module, as transformers' forward does.

    Each model gets its own stand-in modeling module in `sys.modules`, where the binder rebinds the
    kernels, so a binding never outlives the test.
    """
    modeling = types.ModuleType(f"fake_modeling_gated_deltanet_{next(_fake_modeling_ids)}")
    modeling.torch_chunk_gated_delta_rule = native_chunk
    modeling.torch_recurrent_gated_delta_rule = native_chunk
    modeling.causal_conv1d_fn = native_conv
    modeling.causal_conv1d_update = native_conv
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)

    class FakeGatedDeltaNet(nn.Module):
        def forward(self, x, **kwargs):
            x = modeling.causal_conv1d_fn(x, None, None, activation="silu", **kwargs)
            cu_seqlens = kwargs.pop("cu_seq_lens_q", None)
            return modeling.torch_chunk_gated_delta_rule(x, x, x, g=None, beta=None, cu_seqlens=cu_seqlens, **kwargs)

    FakeGatedDeltaNet.__module__ = modeling.__name__
    model = nn.Module()
    model.layers = nn.ModuleList([FakeGatedDeltaNet() for _ in range(n_layers)])
    return model


def _modeling(model):
    return sys.modules[type(model.layers[0]).__module__]


def test_binder_rebinds_the_kernels_every_gated_deltanet_calls(monkeypatch):
    _stub_kernels(monkeypatch, modules=_all_hub_modules())
    model = _build_gdn_model(monkeypatch)

    assert HubKernels.prepare(_make_args()).bind(model) == {"gated_deltanet": 2}
    fla = sys.modules["kernels"].get_kernel("kernels-community/fla")
    conv = sys.modules["kernels"].get_kernel("kernels-community/causal-conv1d")
    modeling = _modeling(model)
    # transformers names the delta-rule kernels after their torch fallbacks; the slot maps across.
    assert modeling.torch_chunk_gated_delta_rule.__wrapped__ is fla.chunk_gated_delta_rule
    assert modeling.torch_recurrent_gated_delta_rule.__wrapped__ is fla.fused_recurrent_gated_delta_rule
    assert modeling.causal_conv1d_fn.__wrapped__ is conv.causal_conv1d_fn
    assert modeling.causal_conv1d_update.__wrapped__ is conv.causal_conv1d_update


def test_gated_deltanet_slots_resolve_independently(monkeypatch):
    """One Hub repo being unavailable must not cost the run the other slot."""
    modules = _all_hub_modules()
    del modules["kernels-community/causal-conv1d"]
    _stub_kernels(monkeypatch, modules=modules)
    model = _build_gdn_model(monkeypatch)

    assert HubKernels.prepare(_make_args()).bind(model) == {"gated_deltanet": 2}
    fla = sys.modules["kernels"].get_kernel("kernels-community/fla")
    assert _modeling(model).torch_chunk_gated_delta_rule.__wrapped__ is fla.chunk_gated_delta_rule
    assert _modeling(model).causal_conv1d_fn is _torch_conv_stand_in  # untouched: no build to bind


def test_binder_is_inert_under_native(monkeypatch):
    _stub_kernels(monkeypatch, modules=_all_hub_modules())
    model = _build_gdn_model(monkeypatch)

    assert HubKernels.prepare(_make_args(kernel_backend="native")).bind(model) == {}
    assert _modeling(model).causal_conv1d_fn is _torch_conv_stand_in
    assert _modeling(model).torch_chunk_gated_delta_rule is _torch_chunk_stand_in


def test_binder_leaves_the_native_kernels_when_every_hub_repo_fails(monkeypatch):
    def native(**kwargs):
        return None

    _stub_kernels(monkeypatch, raises=FileNotFoundError("nope"))
    model = _build_gdn_model(monkeypatch, native_conv=native)

    assert HubKernels.prepare(_make_args()).bind(model) == {}
    assert _modeling(model).causal_conv1d_fn is native
    assert _modeling(model).torch_chunk_gated_delta_rule is _torch_chunk_stand_in


@pytest.mark.parametrize("strict", [False, True])
def test_a_slot_whose_kernels_the_modeling_module_lacks_is_never_reported_bound(monkeypatch, caplog, strict):
    """transformers moving these kernels once turned this binder into a no-op that still logged success."""
    _stub_kernels(monkeypatch, modules=_all_hub_modules())
    model = _build_gdn_model(monkeypatch)
    del _modeling(model).causal_conv1d_update
    hub = HubKernels.prepare(_make_args(kernel_strict=strict))

    if strict:
        with pytest.raises(RuntimeError, match="no module-level GatedDeltaNet kernel"):
            hub.bind(model)
        return
    with caplog.at_level("INFO"):
        assert hub.bind(model) == {"gated_deltanet": 2}  # the delta-rule slot still binds
    assert "causal_conv1d_update'] to rebind; keeping the native kernel" in caplog.text
    assert "GatedDeltaNet gated_delta_rule served from the Hub" in caplog.text
    assert _modeling(model).causal_conv1d_fn is _torch_conv_stand_in  # never half-bound


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe", "qwen3_next", "qwen4_exp", "olmo_hybrid"])
def test_transformers_gated_deltanet_modeling_modules_expose_the_kernels_the_binder_rebinds(model_type):
    modeling = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
    for _, name_to_function in _GDN_KERNEL_SLOTS:
        for name in name_to_function:
            assert callable(getattr(modeling, name, None)), f"{modeling.__name__}.{name}"


def test_hub_kernels_serve_qwen3_5_gated_deltanet_with_the_packed_document_boundaries(monkeypatch, caplog):
    """The reason this feature exists, on transformers' own Qwen3.5.

    Its GatedDeltaNet hands the actor's padding-free kwargs to the module-level kernels of
    `modeling_qwen3_5`; the torch fallbacks drop `cu_seqlens` and `seq_idx`. Bound from the Hub, the
    kernels must be the ones called, receive both boundaries, and keep the arguments they compute on.
    """
    import torch.nn.functional as F
    from tests.fast.backends.test_fsdp_hf_compat import _TINY_QWEN3_5_TEXT, _adapter, _packed_boundaries
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5

    # Binding is process-wide; restore the originals once the test is done.
    for _, name_to_function in _GDN_KERNEL_SLOTS:
        for name in name_to_function:
            monkeypatch.setattr(modeling_qwen3_5, name, getattr(modeling_qwen3_5, name))
    native_chunk = modeling_qwen3_5.torch_chunk_gated_delta_rule
    seen = {}

    # Signatures of the Hub builds, which declare no **kwargs.
    def hub_chunk(
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
    ):
        seen["cu_seqlens"] = cu_seqlens
        return native_chunk(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    def hub_conv(
        x,
        weight,
        bias=None,
        seq_idx=None,
        initial_states=None,
        return_final_states=False,
        final_states_out=None,
        activation=None,
    ):
        seen["seq_idx"] = seq_idx
        out = F.conv1d(F.pad(x, (weight.shape[-1] - 1, 0)), weight.unsqueeze(1), bias, groups=x.shape[1])
        return F.silu(out) if activation in ("silu", "swish") else out

    _stub_kernels(
        monkeypatch,
        modules=_all_hub_modules(
            **{
                "kernels-community/fla": _kernel_module(
                    chunk_gated_delta_rule=hub_chunk, fused_recurrent_gated_delta_rule=_noop
                ),
                "kernels-community/causal-conv1d": _kernel_module(
                    causal_conv1d_fn=hub_conv, causal_conv1d_update=_noop
                ),
            }
        ),
    )

    cfg = Qwen3_5TextConfig(**_TINY_QWEN3_5_TEXT)
    cfg._attn_implementation = "eager"
    model = modeling_qwen3_5.Qwen3_5ForCausalLM(cfg).eval()
    inputs = dict(
        input_ids=torch.arange(5).view(1, 5),
        position_ids=torch.tensor([[0, 1, 2, 0, 1]]),
        **_adapter("qwen3_5_text").packing_kwargs(**_packed_boundaries(3, 2)),
    )
    with torch.no_grad():
        native_logits = model(**inputs).logits
        assert seen == {}
        with caplog.at_level("INFO"):
            assert HubKernels.prepare(_make_args()).bind(model) == {"gated_deltanet": 1}
        hub_logits = model(**inputs).logits

    assert seen["cu_seqlens"].tolist() == [0, 3, 5]
    assert seen["seq_idx"].tolist() == [[0, 0, 0, 1, 1]]
    # The stand-ins compute what the fallbacks do, so equal logits mean no argument was lost on the way.
    torch.testing.assert_close(hub_logits, native_logits)
    assert "served from the Hub in transformers.models.qwen3_5.modeling_qwen3_5 (1 module(s))" in caplog.text


# --------------------------------------------------------------------------------- NemotronH


class FakeAttnMixer(nn.Module):
    pass


class FakeAttnBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_type = "attention"
        self.mixer = FakeAttnMixer()


def test_binder_stashes_varlen_on_each_nemotron_attention_mixer(monkeypatch):
    from miles.backends.fsdp_utils.models.nemotron_h import HUB_VARLEN_ATTR

    def varlen(*args, **kwargs):
        return None

    _stub_kernels(
        monkeypatch,
        modules=_all_hub_modules(**{"kernels-community/flash-attn2": _kernel_module(flash_attn_varlen_func=varlen)}),
    )

    model = nn.Module()
    model.blocks = nn.ModuleList([FakeAttnBlock() for _ in range(3)])

    assert HubKernels.prepare(_make_args()).bind(model) == {"nemotron_h": 3}
    for block in model.blocks:
        assert getattr(block.mixer, HUB_VARLEN_ATTR) is varlen


def test_binders_skip_architectures_that_are_not_present(monkeypatch):
    _stub_kernels(monkeypatch, modules=_all_hub_modules())
    plain = nn.Sequential(nn.Linear(4, 4))

    assert HubKernels.prepare(_make_args()).bind(plain) == {}


class FakeNemotronAttn(nn.Module):
    """Enough of the NemotronH attention mixer for `_patch_attn_forward` to rewrite its forward."""

    def __init__(self, hidden=8, heads=2):
        super().__init__()
        self.head_dim = hidden // heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.dense_calls = 0

    def forward(self, hidden_states, *args, **kwargs):
        self.dense_calls += 1  # the un-patched dense path
        return hidden_states, None


def _patch_nemotron_attn(monkeypatch, native_varlen):
    """Install `_patch_attn_forward` on a fresh class with `flash_attn` stubbed to `native_varlen`."""
    from miles.backends.fsdp_utils.models import nemotron_h as nemotron_h_module

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.flash_attn_varlen_func = native_varlen
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)

    cls = type("PatchTargetAttn", (FakeNemotronAttn,), {})
    nemotron_h_module._patch_attn_forward(cls)
    return cls


def _run_packed_forward(mixer):
    mixer._packing_cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
    mixer._packing_max_seqlen = 2
    return mixer(torch.zeros(1, 4, 8))


def test_nemotron_attention_still_uses_the_native_varlen_kernel(monkeypatch):
    """The Hub handle is an override, not a replacement: with no --kernel-backend hub the wheel wins."""
    used = []

    def native(*args, **kwargs):
        used.append("native")
        return torch.zeros(4, 2, 4)

    mixer = _patch_nemotron_attn(monkeypatch, native)()
    _run_packed_forward(mixer)

    assert used == ["native"]
    assert mixer.dense_calls == 0


def test_nemotron_attention_prefers_the_hub_varlen_kernel_when_bound(monkeypatch):
    from miles.backends.fsdp_utils.models.nemotron_h import HUB_VARLEN_ATTR

    used = []

    def native(*args, **kwargs):
        used.append("native")
        return torch.zeros(4, 2, 4)

    def from_hub(*args, **kwargs):
        used.append("hub")
        return torch.zeros(4, 2, 4)

    mixer = _patch_nemotron_attn(monkeypatch, native)()
    setattr(mixer, HUB_VARLEN_ATTR, from_hub)
    _run_packed_forward(mixer)

    assert used == ["hub"]
    assert mixer.dense_calls == 0


def test_nemotron_attention_falls_back_to_dense_when_no_varlen_kernel_exists(monkeypatch):
    """Pre-existing behaviour, unchanged: no wheel and no hub binding means the dense forward."""
    mixer = _patch_nemotron_attn(monkeypatch, None)()
    _run_packed_forward(mixer)

    assert mixer.dense_calls == 1


# ------------------------------------------------------------------------------ custom mappings


def _incomplete_mapping(args):
    return {args.backend.test_slot: HubKernelSpec(repo_id="me/custom", version=1, functions=("causal_conv1d_fn",))}


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("slot", [SLOT_CAUSAL_CONV1D, SLOT_GATED_DELTA_RULE, SLOT_FLASH_ATTN_VARLEN])
def test_incomplete_custom_slot_is_rejected_before_downloading(monkeypatch, strict, slot):
    calls = []
    _stub_kernels(monkeypatch, modules=_all_hub_modules(), calls=calls)
    args = _make_args(kernel_mapping_path=f"{__name__}._incomplete_mapping", kernel_strict=strict, test_slot=slot)

    with pytest.raises(ValueError, match="must declare all required functions"):
        HubKernels.prepare(args)

    assert calls == []


def test_unknown_custom_slot_is_rejected():
    args = _make_args(kernel_mapping_path=f"{__name__}._incomplete_mapping", test_slot="causal_conv1d_typo")
    with pytest.raises(ValueError, match="unknown kernel mapping slot"):
        load_mapping(args)


def test_complete_custom_mapping_binds_policy_and_reference_without_changing_weights(monkeypatch):
    conv, update = lambda **kwargs: None, lambda **kwargs: None
    calls = []
    _stub_kernels(monkeypatch, module=_kernel_module(causal_conv1d_fn=conv, causal_conv1d_update=update), calls=calls)
    hub = HubKernels.prepare(_make_args(kernel_mapping_path=f"{__name__}._custom_mapping"))

    for _ in range(2):
        model = _build_gdn_model(monkeypatch)
        model.proj = nn.Linear(4, 4)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        assert hub.bind(model) == {"gated_deltanet": 2}
        assert _modeling(model).causal_conv1d_fn.__wrapped__ is conv
        assert _modeling(model).causal_conv1d_update.__wrapped__ is update
        assert _modeling(model).torch_chunk_gated_delta_rule is _torch_chunk_stand_in
        assert model.state_dict().keys() == before.keys()
        for key, expected in before.items():
            torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)
    assert calls == [("me/my-conv1d", "main", None)]


@pytest.mark.parametrize("strict", [False, True])
def test_prepare_rejects_kernels_without_hub_provenance(monkeypatch, strict):
    _stub_kernels(monkeypatch, modules=_all_hub_modules())
    sys.modules["kernels"].get_loaded_kernels = lambda: []
    args = _make_args(kernel_strict=strict)
    if strict:
        with pytest.raises(RuntimeError, match="cannot establish Hub provenance"):
            HubKernels.prepare(args)
    else:
        hub = HubKernels.prepare(args)
        assert hub.bind(_build_gdn_model(monkeypatch)) == {}
