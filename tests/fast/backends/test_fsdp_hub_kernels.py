"""Module-level Hub compute kernels for the FSDP backend (CPU-only, no GPU, no network).

`kernels` is stubbed through sys.modules throughout, so these never touch the Hub. What they pin:

  * `--kernel-backend native` (the default) resolves nothing and never imports `kernels`;
  * the numerics guard that keeps hub kernels out of --true-on-policy-mode / --deterministic-mode;
  * strict vs. fallback when a repo or a function is missing from the build;
  * and, the reason this feature exists, that a Hub-bound `causal_conv1d_fn` still receives the
    packed-document `seq_idx` that `models/qwen3_5.py` injects -- i.e. the binder composes with the
    packing wrapper rather than defeating it.
"""

import sys
import types
from argparse import Namespace

import pytest
import torch
import torch.nn as nn

from miles.backends.fsdp_utils.arguments import validate_kernel_backend_args
from miles.backends.fsdp_utils.kernels import hub
from miles.backends.fsdp_utils.kernels.hub import HubKernelSpec, hub_kernels_enabled, load_module_kernels, resolve_slot
from miles.backends.fsdp_utils.kernels.module_patches import apply_hub_module_kernels
from miles.backends.fsdp_utils.kernels.presets import (
    SLOT_CAUSAL_CONV1D,
    SLOT_FLASH_ATTN_VARLEN,
    SLOT_GATED_DELTA_RULE,
)


def _make_args(**overrides) -> Namespace:
    base = dict(
        kernel_backend="hub",
        kernel_mapping_path="",
        kernel_strict=False,
        true_on_policy_mode=False,
        deterministic_mode=False,
    )
    base.update(overrides)
    return Namespace(**base)


@pytest.fixture(autouse=True)
def clear_kernel_cache():
    """The resolved-module cache is process-global by design; don't leak it between tests."""
    hub._RESOLVED.clear()
    yield
    hub._RESOLVED.clear()


def _stub_kernels(monkeypatch, *, module=None, modules=None, raises=None, calls=None):
    """Install a fake `kernels` package.

    `module` serves every repo; `modules` maps repo_id -> module so a test can make one repo
    resolve and another 404, which is how the per-slot independence is exercised.
    """

    def get_kernel(repo_id, revision=None, version=None, user_agent=None):
        if calls is not None:
            calls.append((repo_id, revision, version))
        if raises is not None:
            raise raises
        if modules is not None:
            if repo_id not in modules:
                raise FileNotFoundError(f"no stub build for {repo_id}")
            return modules[repo_id]
        return module

    fake = types.ModuleType("kernels")
    fake.get_kernel = get_kernel
    monkeypatch.setitem(sys.modules, "kernels", fake)


def _kernel_module(**functions):
    mod = types.ModuleType("fake_hub_kernel")
    for name, fn in functions.items():
        setattr(mod, name, fn)
    return mod


def _noop(*args, **kwargs):
    return None


def _all_hub_modules(**overrides):
    """One stub module per repo in the shipped mapping, with every advertised function present."""
    modules = {
        "kernels-community/fla": _kernel_module(chunk_gated_delta_rule=_noop, fused_recurrent_gated_delta_rule=_noop),
        "kernels-community/causal-conv1d": _kernel_module(causal_conv1d_fn=_noop, causal_conv1d_update=_noop),
        "kernels-community/flash-attn2": _kernel_module(flash_attn_varlen_func=_noop),
    }
    modules.update(overrides)
    return modules


# --------------------------------------------------------------------------------------- args


def test_native_is_the_default_and_stays_inert(monkeypatch):
    args = _make_args(kernel_backend="native")
    monkeypatch.delitem(sys.modules, "kernels", raising=False)

    assert hub_kernels_enabled(args) is False
    assert load_module_kernels(args) == {}
    assert resolve_slot(args, SLOT_CAUSAL_CONV1D) is None
    # The whole point of the lazy import: a native run must not need `kernels` to exist at all.
    assert "kernels" not in sys.modules


@pytest.mark.parametrize("mode", ["true_on_policy_mode", "deterministic_mode"])
def test_hub_backend_is_rejected_in_the_bit_exact_modes(mode):
    with pytest.raises(ValueError, match="incompatible with --true-on-policy-mode"):
        validate_kernel_backend_args(_make_args(**{mode: True}))


@pytest.mark.parametrize("mode", ["true_on_policy_mode", "deterministic_mode"])
def test_default_mapping_is_empty_in_the_bit_exact_modes(mode):
    """Defence in depth: even if validation is bypassed, the preset hands back nothing to bind."""
    assert load_module_kernels(_make_args(**{mode: True})) == {}


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="--kernel-backend must be one of"):
        validate_kernel_backend_args(_make_args(kernel_backend="hubb"))


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(kernel_strict=True), "--kernel-strict only applies"),
        (dict(kernel_mapping_path="pkg.mod.fn"), "--kernel-mapping-path only applies"),
    ],
)
def test_hub_only_flags_are_rejected_under_native(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_kernel_backend_args(_make_args(kernel_backend="native", **overrides))


def test_native_and_hub_both_validate_clean():
    validate_kernel_backend_args(_make_args(kernel_backend="native"))
    validate_kernel_backend_args(_make_args(kernel_backend="hub", kernel_strict=True))


# ------------------------------------------------------------------------------------ presets


def test_default_mapping_pins_the_expected_repos():
    mapping = load_module_kernels(_make_args())

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
        SLOT_CAUSAL_CONV1D: HubKernelSpec(repo_id="me/my-conv1d", revision="main", functions=("causal_conv1d_fn",))
    }


def _broken_mapping(args):
    return {SLOT_CAUSAL_CONV1D: "kernels-community/causal-conv1d"}


def test_kernel_mapping_path_substitutes_the_whole_mapping():
    args = _make_args(kernel_mapping_path=f"{__name__}._custom_mapping")
    mapping = load_module_kernels(args)

    assert set(mapping) == {SLOT_CAUSAL_CONV1D}
    assert mapping[SLOT_CAUSAL_CONV1D].repo_id == "me/my-conv1d"


def test_kernel_mapping_path_rejects_a_non_spec_value():
    args = _make_args(kernel_mapping_path=f"{__name__}._broken_mapping")
    with pytest.raises(TypeError, match="must be a HubKernelSpec"):
        load_module_kernels(args)


def test_spec_rejects_both_pins_and_no_functions():
    with pytest.raises(ValueError, match="not both"):
        HubKernelSpec(repo_id="a/b", version=1, revision="main", functions=("f",))
    with pytest.raises(ValueError, match="at least one function"):
        HubKernelSpec(repo_id="a/b", version=1)


# ---------------------------------------------------------------------------------- resolution


def test_resolve_slot_passes_the_pin_through_and_caches_the_module(monkeypatch):
    calls = []
    _stub_kernels(
        monkeypatch,
        module=_kernel_module(causal_conv1d_fn=lambda **kw: None, causal_conv1d_update=lambda: None),
        calls=calls,
    )
    args = _make_args()

    first = resolve_slot(args, SLOT_CAUSAL_CONV1D)
    second = resolve_slot(args, SLOT_CAUSAL_CONV1D)

    assert set(first) == {"causal_conv1d_fn", "causal_conv1d_update"}
    assert first == second
    # One download for the policy model and the ref model together.
    assert calls == [("kernels-community/causal-conv1d", None, 1)]


def test_unresolvable_repo_falls_back_to_the_native_kernel(monkeypatch, caplog):
    _stub_kernels(monkeypatch, raises=FileNotFoundError("no build for torch2.9-cu130"))

    with caplog.at_level("WARNING"):
        assert resolve_slot(_make_args(), SLOT_CAUSAL_CONV1D) is None
    assert "keeping the native kernel" in caplog.text


def test_unresolvable_repo_raises_under_strict(monkeypatch):
    _stub_kernels(monkeypatch, raises=FileNotFoundError("no build for torch2.9-cu130"))

    with pytest.raises(RuntimeError, match="--kernel-strict"):
        resolve_slot(_make_args(kernel_strict=True), SLOT_CAUSAL_CONV1D)


def test_a_failed_repo_is_not_retried_per_model(monkeypatch):
    calls = []
    _stub_kernels(monkeypatch, raises=FileNotFoundError("nope"), calls=calls)
    args = _make_args()

    assert resolve_slot(args, SLOT_CAUSAL_CONV1D) is None
    assert resolve_slot(args, SLOT_CAUSAL_CONV1D) is None
    assert len(calls) == 1


def test_a_build_missing_the_function_falls_back(monkeypatch, caplog):
    _stub_kernels(monkeypatch, module=_kernel_module(causal_conv1d_fn=lambda: None))  # no causal_conv1d_update

    with caplog.at_level("WARNING"):
        assert resolve_slot(_make_args(), SLOT_CAUSAL_CONV1D) is None
    assert "does not expose a callable" in caplog.text

    with pytest.raises(RuntimeError, match="does not expose a callable"):
        resolve_slot(_make_args(kernel_strict=True), SLOT_CAUSAL_CONV1D)


def test_prefetch_is_inert_under_native(monkeypatch):
    calls = []
    _stub_kernels(monkeypatch, module=_kernel_module(), calls=calls)

    hub.prefetch_hub_module_kernels(_make_args(kernel_backend="native"))

    assert calls == []


def test_prefetch_warms_every_mapped_repo_once(monkeypatch):
    calls = []
    _stub_kernels(monkeypatch, modules=_all_hub_modules(), calls=calls)
    args = _make_args()

    hub.prefetch_hub_module_kernels(args)
    for slot in (SLOT_GATED_DELTA_RULE, SLOT_CAUSAL_CONV1D, SLOT_FLASH_ATTN_VARLEN):
        resolve_slot(args, slot)

    # Downloaded once each by the prefetch; the per-slot resolves are all cache hits.
    assert sorted(repo for repo, _, _ in calls) == [
        "kernels-community/causal-conv1d",
        "kernels-community/fla",
        "kernels-community/flash-attn2",
    ]


# ------------------------------------------------------------------------------ GatedDeltaNet


class FakeGatedDeltaNet(nn.Module):
    """Mirrors the parts of Qwen3NextGatedDeltaNet the binder and the packing wrapper touch.

    `native_chunk` defaults to a stand-in for transformers' `torch_chunk_gated_delta_rule`: it
    accepts `cu_seqlens` into `**kwargs` and ignores it, which is exactly the silent degradation
    the `fla` slot exists to remove.
    """

    def __init__(self, native_conv=None, native_chunk=None):
        super().__init__()
        self.causal_conv1d_fn = native_conv
        self.causal_conv1d_update = None
        self.chunk_gated_delta_rule = native_chunk or _torch_chunk_stand_in
        self.recurrent_gated_delta_rule = native_chunk or _torch_chunk_stand_in

    def forward(self, x):
        self.chunk_gated_delta_rule(x, g=None, beta=None, cu_seqlens=None)
        if self.causal_conv1d_fn is None:
            # transformers' fallback: no seq_idx, so packed documents bleed into each other.
            return x
        return self.causal_conv1d_fn(x=x, weight=None, bias=None, activation="silu", seq_idx=None)


def _torch_chunk_stand_in(query, g=None, beta=None, **kwargs):
    """transformers' pure-torch GDN fallback: swallows cu_seqlens and never resets per document."""
    return query


def _build_gdn_model(native_conv=None, native_chunk=None, n_layers=2):
    model = nn.Module()
    model.layers = nn.ModuleList([FakeGatedDeltaNet(native_conv, native_chunk) for _ in range(n_layers)])
    return model


def test_binder_rebinds_every_gated_deltanet(monkeypatch):
    _stub_kernels(monkeypatch, modules=_all_hub_modules())
    model = _build_gdn_model()

    assert apply_hub_module_kernels(model, _make_args()) == {"gated_deltanet": 2}
    fla = sys.modules["kernels"].get_kernel("kernels-community/fla")
    conv = sys.modules["kernels"].get_kernel("kernels-community/causal-conv1d")
    for layer in model.layers:
        assert layer.chunk_gated_delta_rule is fla.chunk_gated_delta_rule
        # transformers stores the recurrent kernel without the `fused_` prefix; the slot maps across.
        assert layer.recurrent_gated_delta_rule is fla.fused_recurrent_gated_delta_rule
        assert layer.causal_conv1d_fn is conv.causal_conv1d_fn
        assert layer.causal_conv1d_update is conv.causal_conv1d_update


def test_gated_deltanet_slots_resolve_independently(monkeypatch):
    """One Hub repo being unavailable must not cost the run the other slot."""
    modules = _all_hub_modules()
    del modules["kernels-community/causal-conv1d"]
    _stub_kernels(monkeypatch, modules=modules)
    model = _build_gdn_model(native_conv=None)

    assert apply_hub_module_kernels(model, _make_args()) == {"gated_deltanet": 2}
    fla = sys.modules["kernels"].get_kernel("kernels-community/fla")
    assert model.layers[0].chunk_gated_delta_rule is fla.chunk_gated_delta_rule
    assert model.layers[0].causal_conv1d_fn is None  # untouched: no build to bind


def test_binder_is_inert_under_native(monkeypatch):
    _stub_kernels(monkeypatch, modules=_all_hub_modules())
    model = _build_gdn_model(native_conv=None)

    assert apply_hub_module_kernels(model, _make_args(kernel_backend="native")) == {}
    assert model.layers[0].causal_conv1d_fn is None
    assert model.layers[0].chunk_gated_delta_rule is _torch_chunk_stand_in


def test_binder_leaves_the_native_kernels_when_every_hub_repo_fails(monkeypatch):
    def native(**kwargs):
        return None

    _stub_kernels(monkeypatch, raises=FileNotFoundError("nope"))
    model = _build_gdn_model(native_conv=native)

    assert apply_hub_module_kernels(model, _make_args()) == {}
    assert model.layers[0].causal_conv1d_fn is native
    assert model.layers[0].chunk_gated_delta_rule is _torch_chunk_stand_in


def test_hub_kernels_still_receive_the_packed_document_boundaries(monkeypatch):
    """The reason this feature exists.

    `_patch_gdn_forward` injects `cu_seqlens` into the recurrence and `seq_idx` into the conv. The
    native fallbacks defeat that -- the torch GDN kernel swallows `cu_seqlens` into `**kwargs`, and
    a missing conv wheel leaves nothing to wrap. Bind both from the Hub and the boundaries must
    arrive at the kernels unchanged.
    """
    from miles.backends.fsdp_utils.models.qwen3_5 import _patch_gdn_forward

    seen = {}

    def hub_chunk(query, g=None, beta=None, **kwargs):
        seen["cu_seqlens"] = kwargs.get("cu_seqlens")
        return query

    def hub_conv(**kwargs):
        seen["seq_idx"] = kwargs.get("seq_idx")
        return kwargs["x"]

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

    # Patch a throwaway subclass: _patch_gdn_forward rewrites the class forward permanently.
    gdn_cls = type("PackedGatedDeltaNet", (FakeGatedDeltaNet,), {})
    _patch_gdn_forward(gdn_cls)

    model = nn.Module()
    model.layers = nn.ModuleList([gdn_cls()])
    assert apply_hub_module_kernels(model, _make_args()) == {"gated_deltanet": 1}

    layer = model.layers[0]
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
    seq_idx = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)
    layer._gdn_cu_seqlens = cu_seqlens
    layer._gdn_seq_idx = seq_idx
    layer(torch.zeros(1, 4))

    assert torch.equal(seen["cu_seqlens"], cu_seqlens)
    assert torch.equal(seen["seq_idx"], seq_idx)
    # The wrapper restores the handles it swapped, so the next forward starts from the Hub kernels
    # again rather than from a stack of boundary-injecting wrappers.
    assert layer.chunk_gated_delta_rule is hub_chunk
    assert layer.causal_conv1d_fn is hub_conv


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

    _stub_kernels(monkeypatch, module=_kernel_module(flash_attn_varlen_func=varlen))

    model = nn.Module()
    model.blocks = nn.ModuleList([FakeAttnBlock() for _ in range(3)])

    assert apply_hub_module_kernels(model, _make_args()) == {"nemotron_h": 3}
    for block in model.blocks:
        assert getattr(block.mixer, HUB_VARLEN_ATTR) is varlen


def test_binders_skip_architectures_that_are_not_present(monkeypatch):
    _stub_kernels(
        monkeypatch,
        module=_kernel_module(
            causal_conv1d_fn=lambda: None, causal_conv1d_update=lambda: None, flash_attn_varlen_func=lambda: None
        ),
    )
    plain = nn.Sequential(nn.Linear(4, 4))

    assert apply_hub_module_kernels(plain, _make_args()) == {}


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
