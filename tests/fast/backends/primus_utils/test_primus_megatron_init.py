import argparse

import pytest


class _FakeRunner:
    def __init__(self, calls):
        self.calls = calls

    def run(self, phase):
        self.calls.append(phase)
        return 0

    def bind_megatron_global_args(self):
        self.calls.append("bind")


def test_phases_run_in_order_after_the_chained_init(monkeypatch):
    from miles.backends.primus_utils import megatron_init

    calls = []
    monkeypatch.setattr(megatron_init, "install_megatron_hooks", lambda: calls.append("hooks"))
    monkeypatch.setattr(
        megatron_init.PrimusPatchRunner, "create_if_enabled", classmethod(lambda cls, args: _FakeRunner(calls))
    )
    monkeypatch.setattr(
        "miles.utils.function_registry.load_function", lambda path: (lambda args: calls.append(f"user:{path}"))
    )

    args = argparse.Namespace(primus_chained_megatron_init_path="my_pkg.my_init")
    monkeypatch.setattr("megatron.training.get_args", lambda: args)
    megatron_init.primus_megatron_init(args)

    assert calls == ["hooks", "user:my_pkg.my_init", "setup", "build_args", "bind", "before_train"]
    assert megatron_init.get_primus_patch_runner() is not None


def test_nothing_runs_when_patches_are_disabled(monkeypatch):
    from miles.backends.primus_utils import megatron_init

    monkeypatch.setattr(megatron_init, "install_megatron_hooks", lambda: None)
    monkeypatch.setattr(megatron_init.PrimusPatchRunner, "create_if_enabled", classmethod(lambda cls, args: None))

    args = argparse.Namespace(primus_chained_megatron_init_path=None)
    monkeypatch.setattr("megatron.training.get_args", lambda: args)
    megatron_init.primus_megatron_init(args)
    assert megatron_init.get_primus_patch_runner() is None


def _packed_grouped_linear(torch, num_experts, out_features, in_features):
    """Mimics PrimusTurboGroupedLinear: packed `weights` plus lazily registered `weight{i}` views."""

    class PackedGroupedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.Parameter(torch.randn(num_experts, out_features, in_features))
            self._registered = False

        def _ensure_weight_views(self):
            if self._registered:
                return
            for i in range(num_experts):
                self.register_parameter(f"weight{i}", torch.nn.Parameter(self.weights[i], requires_grad=False))
            self._registered = True

    return PackedGroupedLinear()


@pytest.fixture
def megatron_hooks(monkeypatch):
    """Install the hooks so that monkeypatch restores every wrapped function afterwards."""
    pytest.importorskip("megatron.core")
    from miles.backends.megatron_utils import checkpoint, megatron_to_hf, named_weights
    from miles.backends.primus_utils import megatron_hooks

    targets = [
        (checkpoint, "_load_checkpoint_hf"),
        (megatron_to_hf, "_convert_to_hf_core"),
        (named_weights, "_named_params_and_buffers_vanilla"),
        (named_weights, "_named_params_and_buffers_global"),
    ]
    for module, name in targets:
        monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setattr(megatron_hooks, "_installed", False)

    def install(**fakes):
        for name, fake in fakes.items():
            module = next(module for module, target in targets if target == name)
            monkeypatch.setattr(module, name, fake)
        megatron_hooks.install_megatron_hooks()

    return install


def test_named_weights_expose_expert_views_not_packed_storage(megatron_hooks):
    torch = pytest.importorskip("torch")
    from miles.backends.megatron_utils import named_weights

    megatron_hooks()
    root = torch.nn.Module()
    root.decoder = torch.nn.Module()
    root.decoder.mlp = torch.nn.Module()
    root.decoder.mlp.experts = torch.nn.Module()
    root.decoder.mlp.experts.linear_fc1 = _packed_grouped_linear(torch, 2, 4, 3)

    names = [name for name, _ in named_weights._named_params_and_buffers_vanilla([root])]

    assert names == [
        "vp_stages.0.decoder.mlp.experts.linear_fc1.weight0",
        "vp_stages.0.decoder.mlp.experts.linear_fc1.weight1",
    ]


def test_primus_converter_is_dispatched_by_model_name(megatron_hooks, monkeypatch):
    from miles.backends.megatron_utils import megatron_to_hf
    from miles.backends.primus_utils import glm5_next_to_hf

    megatron_hooks(_convert_to_hf_core=lambda args, model_name, name, param: [("upstream", param)])
    monkeypatch.setattr(glm5_next_to_hf, "convert_primus_glm5_next_to_hf", lambda args, name, param: [("primus", param)])

    assert megatron_to_hf._convert_to_hf_core(None, "Primus_GLM5_Next", "w", 1) == [("primus", 1)]
    assert megatron_to_hf._convert_to_hf_core(None, "glm5_next", "w", 1) == [("upstream", 1)]


def test_primus_families_load_hf_with_the_primus_loader(megatron_hooks, monkeypatch):
    from miles.backends.megatron_utils import checkpoint
    from miles.backends.primus_utils import models

    calls = []
    megatron_hooks(_load_checkpoint_hf=lambda ddp_model, optimizer, args, load_path: calls.append("upstream") or (7, 7))

    class _Optimizer:
        def reload_model_params(self):
            calls.append("reload")

    args = argparse.Namespace(backend=argparse.Namespace(fp16=False, bf16=True, load_main_params_from_ckpt=False))
    monkeypatch.setattr(models, "load_primus_hf_checkpoint", lambda ddp_model, args, load_path: True)
    assert checkpoint._load_checkpoint_hf(ddp_model=[], optimizer=_Optimizer(), args=args, load_path="/hf") == (0, 0)
    assert calls == ["reload"]

    monkeypatch.setattr(models, "load_primus_hf_checkpoint", lambda ddp_model, args, load_path: False)
    assert checkpoint._load_checkpoint_hf(ddp_model=[], optimizer=None, args=args, load_path="/hf") == (7, 7)
    assert calls == ["reload", "upstream"]
