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
    monkeypatch.setattr(
        megatron_init.PrimusPatchRunner, "create_if_enabled", classmethod(lambda cls, args: _FakeRunner(calls))
    )
    monkeypatch.setattr(
        "miles.utils.function_registry.load_function", lambda path: (lambda args: calls.append(f"user:{path}"))
    )

    args = argparse.Namespace(primus_chained_megatron_init_path="my_pkg.my_init")
    monkeypatch.setattr("megatron.training.get_args", lambda: args)
    megatron_init.primus_megatron_init(args)

    assert calls == ["user:my_pkg.my_init", "setup", "build_args", "bind", "before_train"]
    assert megatron_init.get_primus_patch_runner() is not None


def test_nothing_runs_when_patches_are_disabled(monkeypatch):
    from miles.backends.primus_utils import megatron_init

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


def test_named_weights_expose_expert_views_not_packed_storage():
    torch = pytest.importorskip("torch")
    pytest.importorskip("megatron.core")
    from miles.backends.megatron_utils.named_weights import _named_params_and_buffers_vanilla

    root = torch.nn.Module()
    root.decoder = torch.nn.Module()
    root.decoder.mlp = torch.nn.Module()
    root.decoder.mlp.experts = torch.nn.Module()
    root.decoder.mlp.experts.linear_fc1 = _packed_grouped_linear(torch, 2, 4, 3)

    names = [name for name, _ in _named_params_and_buffers_vanilla([root])]

    assert names == [
        "vp_stages.0.decoder.mlp.experts.linear_fc1.weight0",
        "vp_stages.0.decoder.mlp.experts.linear_fc1.weight1",
    ]
