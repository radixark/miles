"""CPU-only regression tests against the installed, unmodified MCore HDO step.

Only device placement/streams are faked; parameter groups, real AdamW updates,
HDO copy-back hooks, and checkpoint load/save are exercised end to end.
"""

import argparse
import copy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

from miles.backends.megatron_utils.optimizer_cpu_streaming_gradients import (
    add_optimizer_cpu_streaming_gradients_argument,
    install_optimizer_cpu_streaming_gradients,
)


class Command:
    def __init__(self, fn):
        self.fn = fn
        self.done = False

    def run(self):
        if not self.done:
            self.done = True
            self.fn()


class FakeEvent:
    def __init__(self, commands):
        self.commands = tuple(commands)

    def synchronize(self):
        for command in self.commands:
            command.run()

    def wait(self, stream):
        stream.commands.append(Command(self.synchronize))


class FakeStream:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.commands = []
        env.streams.append(self)

    def record_event(self):
        return FakeEvent(self.commands)

    def wait_stream(self, other):
        self.record_wait(other.record_event())

    def record_wait(self, event):
        self.commands.append(Command(event.synchronize))

    def synchronize(self):
        self.env.log.append(("synchronize", self.name))
        self.record_event().synchronize()
        self.commands.clear()


@pytest.fixture
def streams(monkeypatch):
    env = SimpleNamespace(streams=[], log=[], transfers=[])
    env.current = FakeStream(env, "default")

    @contextmanager
    def use_stream(stream):
        previous = env.current
        env.current = stream
        try:
            yield stream
        finally:
            env.current = previous

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: env.current)
    monkeypatch.setattr(torch.cuda, "stream", use_stream)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: FakeStream(env, f"stream-{len(env.streams)}"))
    real_copy, real_to = torch.Tensor.copy_, torch.Tensor.to

    def deferred_copy(target, source, non_blocking=False):
        if non_blocking:
            # Keep the source live, not a snapshot: detects master overwrites
            # before async H2D has actually finished reading it.
            env.current.commands.append(Command(lambda: real_copy(target, source)))
            return target
        env.current.synchronize()
        return real_copy(target, source)

    def blocking_to(tensor, *args, **kwargs):
        if args and args[0] == "cpu":
            env.transfers.append(kwargs.copy())
            if not kwargs.get("non_blocking", False):
                env.current.synchronize()
        return real_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", deferred_copy)
    monkeypatch.setattr(torch.Tensor, "to", blocking_to)
    return env


class CPUHDO(HybridDeviceOptimizer):
    """Fake only CUDA placement; retain real HDO init, step and all hooks."""

    def _get_sub_optimizer_param_groups(self, offload_fraction):
        cpu_groups, gpu_groups, gpu_to_cpu, cpu_to_gpu, fp32 = [], [], {}, {}, {}
        for group in self.param_groups:
            cpu_group = {**group, "params": []}
            gpu_group = {**group, "params": []}
            for source in group["params"]:
                offloaded = source.fake_offloaded
                master = source.detach().clone().float() if offloaded or source.dtype != torch.float32 else source
                if source.dtype != torch.float32:
                    fp32[source] = master
                if offloaded:
                    gpu_to_cpu[source] = master
                    cpu_to_gpu[master] = source
                    cpu_group["params"].append(master)
                else:
                    gpu_group["params"].append(master)
            if cpu_group["params"]:
                cpu_groups.append(cpu_group)
            if gpu_group["params"]:
                gpu_groups.append(gpu_group)
        return cpu_groups, gpu_groups, gpu_to_cpu, cpu_to_gpu, fp32

    def _move_new_state_to_right_device(self):
        # The mock GPU is CPU-resident, so no actual CUDA state transfer.
        pass


def make_hdo(*, cpu_count=3, gpu_count=2, fused=False, foreach=False):
    params = []
    for index in range(cpu_count + gpu_count):
        # Exercise both the separate BF16->FP32 master path and native FP32.
        dtype = torch.bfloat16 if index % 2 == 0 else torch.float32
        param = torch.nn.Parameter(torch.tensor([0.25, -0.5, 1.5, -2.0], dtype=dtype) + index / 8)
        param.fake_offloaded = index < cpu_count
        params.append(param)
    groups = [
        {"params": params[::2], "lr": 0.007, "weight_decay": 0.13},
        {"params": params[1::2], "lr": 0.013, "weight_decay": 0.07},
    ]
    groups = [group for group in groups if group["params"]]
    return CPUHDO(
        groups,
        offload_fraction=1.0,
        cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=torch.optim.AdamW,
        param_update_in_fp32=True,
        overlap_cpu_optimizer_d2h_h2d=True,
        pin_cpu_grads=False,
        pin_cpu_params=False,
        betas=(0.83, 0.97),
        eps=1e-7,
        fused=fused,
        foreach=foreach,
    )


def args(**changes):
    return SimpleNamespace(
        **dict(
            optimizer_cpu_streaming_gradients=True,
            optimizer_cpu_offload=True,
            overlap_cpu_optimizer_d2h_h2d=True,
            train_backend="megatron",
        ) | changes
    )


def install(hdo):
    install_optimizer_cpu_streaming_gradients(args(), SimpleNamespace(optimizer=hdo))
    return hdo


def sources(hdo):
    return [param for group in hdo.param_groups for param in group["params"]]


def set_grads(hdo, step, *, decoupled=True, missing=()):
    hdo.zero_grad()
    for index, source in enumerate(sources(hdo)):
        grad = None if index in missing else torch.tensor([0.3, -0.2, 0.7, -0.9]) * (step + index + 1) / 8
        if decoupled:
            source.decoupled_grad = grad
            if not source.fake_offloaded and source.dtype == torch.float32:
                # torch AdamW stands in for the GPU optimizer; native FP32
                # parameters have no separate master, so give it the direct grad.
                source.grad = grad
        else:
            source.grad = None if grad is None else grad.to(source.dtype)
    for index, group in enumerate(hdo.param_groups):
        group["lr"] = (0.007 + index * 0.002) / (step + 1)
        group["weight_decay"] = 0.05 + step * 0.01


def assert_tree_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_tree_equal(a, b)
    else:
        assert actual == expected


def assert_hdo_equal(actual, expected):
    for a, b in zip(sources(actual), sources(expected), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(actual.param_to_inner_param[a], expected.param_to_inner_param[b], rtol=0, atol=0)
    assert_tree_equal(actual.state_dict(), expected.state_dict())
    for state in actual.state.values():
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32


def assert_released(hdo):
    assert not hdo.cpu_copy_map_grad
    assert not hdo._cpu_optimizer_map_data_event
    assert all(param.grad is None for optimizer in hdo.cpu_optimizers for param in sources(optimizer))


@pytest.mark.parametrize("decoupled", [True, False])
@pytest.mark.parametrize("fused,foreach", [(False, False), (True, False), (False, True)])
def test_multistep_exact_original_hdo_parity(streams, decoupled, fused, foreach):
    reference = make_hdo(fused=fused, foreach=foreach)
    actual = install(make_hdo(fused=fused, foreach=foreach))
    max_live = []
    for child in actual.cpu_optimizers:
        def check_live(optimizer, unused_args, unused_kwargs):
            live = [p for opt in actual.cpu_optimizers for p in sources(opt) if p.grad is not None]
            assert live == sources(optimizer)
            assert set(actual.cpu_copy_map_grad) == set(live)
            assert not live[0].grad.is_pinned()
            assert live[0].grad.dtype == torch.float32
            max_live.append(len(live))

        child.register_step_pre_hook(check_live)
    for step in range(5):
        set_grads(reference, step, decoupled=decoupled)
        set_grads(actual, step, decoupled=decoupled)
        reference.step()
        reference._h2d_stream.synchronize()
        actual.step()
        assert_hdo_equal(actual, reference)
        assert_released(actual)
    assert max(max_live) == 1
    assert streams.transfers
    assert all(transfer["non_blocking"] is False and transfer["copy"] for transfer in streams.transfers)


def test_none_grad_transitions_skip_decay_moments_and_step(streams):
    actual = install(make_hdo())
    reference = make_hdo()
    for step, missing in enumerate([(0, 3), (), (1, 2, 3), (0, 1, 2, 3, 4), ()]):
        set_grads(actual, step, missing=missing)
        set_grads(reference, step, missing=missing)
        # Original HDO caches stale .grad even after zero_grad(set_to_none=True).
        # Clear only that staging cache to obtain true AdamW None-grad semantics;
        # its step, optimizer mathematics, and copy-back hooks stay unmodified.
        reference.cpu_copy_map_grad.clear()
        for child in reference.sub_optimizers:
            for param in sources(child):
                if reference.inner_param_to_orig_param[param] is not param:
                    param.grad = None
        before = {
            index: (actual.param_to_inner_param[source].clone(), copy.deepcopy(actual.state.get(source, {})))
            for index, source in enumerate(sources(actual)) if index in missing
        }
        reference.step()
        reference._h2d_stream.synchronize()
        actual.step()
        assert_hdo_equal(actual, reference)
        assert_released(actual)
        for index, (weight, state) in before.items():
            source = sources(actual)[index]
            assert_tree_equal(actual.param_to_inner_param[source], weight)
            assert_tree_equal(actual.state.get(source, {}), state)


@pytest.mark.parametrize("cpu_count,gpu_count", [(0, 3), (3, 0), (2, 2)])
def test_cpu_only_gpu_only_and_mixed_hdo(streams, cpu_count, gpu_count):
    actual = install(make_hdo(cpu_count=cpu_count, gpu_count=gpu_count))
    reference = make_hdo(cpu_count=cpu_count, gpu_count=gpu_count)
    for step in range(3):
        set_grads(actual, step)
        set_grads(reference, step)
        reference.step()
        reference._h2d_stream.synchronize()
        actual.step()
        assert_hdo_equal(actual, reference)
        assert_released(actual)


def test_hook_order_and_pending_copies_are_completed(streams):
    hdo = install(make_hdo())
    log = []
    hdo.register_step_pre_hook(lambda *_: log.append("hdo-pre"))
    hdo.register_step_post_hook(lambda *_: log.append("hdo-post"))
    hdo.gpu_optimizer.register_step_post_hook(lambda *_: log.append("gpu-post"))
    for index, child in enumerate(hdo.cpu_optimizers):
        child.register_step_pre_hook(lambda *_, index=index: log.append(f"cpu-{index}-pre"))
        child.register_step_post_hook(lambda *_, index=index: log.append(f"cpu-{index}-post"))
    original_sync = hdo._sync_hdo_param_groups_to_sub_optimizers
    original_state = hdo._sync_sub_optimizers_state_to_hdo

    def sync_groups():
        log.append("groups")
        original_sync()

    def sync_state():
        log.append("state")
        original_state()

    hdo._sync_hdo_param_groups_to_sub_optimizers = sync_groups
    hdo._sync_sub_optimizers_state_to_hdo = sync_state
    set_grads(hdo, 0)
    source = hdo.cpu_copys_map_gpu_param[sources(hdo.cpu_optimizers[0])[0]]
    expected_grad = source.decoupled_grad.clone()
    source.decoupled_grad.zero_()
    streams.current.commands.append(Command(lambda: source.decoupled_grad.copy_(expected_grad)))
    hdo.step()
    assert log == [
        "hdo-pre", "groups", "gpu-post",
        "cpu-0-pre", "cpu-0-post", "cpu-1-pre", "cpu-1-post", "cpu-2-pre", "cpu-2-post",
        "state", "hdo-post",
    ]
    state = hdo.cpu_optimizers[0].state[sources(hdo.cpu_optimizers[0])[0]]
    torch.testing.assert_close(state["exp_avg"], expected_grad * (1 - 0.83), rtol=1e-7, atol=0)
    for cpu, gpu in hdo.cpu_copys_map_gpu_param.items():
        assert_tree_equal(gpu, cpu.to(gpu.dtype))
    assert not hdo._h2d_stream.commands
    assert_released(hdo)


@pytest.mark.parametrize("fail_at", ["gpu", "cpu", "copyback", "d2h"])
def test_exception_stops_later_updates_and_prevents_retry(streams, monkeypatch, fail_at):
    hdo = install(make_hdo())
    set_grads(hdo, 0)
    later = hdo.cpu_optimizers[-1]
    snapshots = [p.clone() for p in sources(later)]

    def fail(*_args, **_kwargs):
        raise ValueError("injected failure")

    if fail_at == "gpu":
        hdo.gpu_optimizer.register_step_pre_hook(fail)
    elif fail_at == "cpu":
        hdo.cpu_optimizers[1].register_step_pre_hook(fail)
    elif fail_at == "copyback":
        hdo.cpu_optimizers[1].register_step_post_hook(fail)
    else:
        real_to = torch.Tensor.to

        def fail_to(tensor, *to_args, **kwargs):
            if to_args and to_args[0] == "cpu":
                fail()
            return real_to(tensor, *to_args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", fail_to)
    with pytest.raises(ValueError, match="injected failure"):
        hdo.step()
    assert not later.state
    assert_tree_equal(sources(later), snapshots)
    assert_released(hdo)
    assert not hdo._h2d_stream.commands
    with pytest.raises(RuntimeError, match="previous step failed"):
        hdo.step()


def test_checkpoint_schema_and_resume_exactly_match_reference(streams):
    actual, reference = install(make_hdo()), make_hdo()
    for step in range(2):
        set_grads(actual, step)
        set_grads(reference, step)
        actual.step()
        reference.step()
        reference._h2d_stream.synchronize()
    checkpoint = copy.deepcopy(actual.state_dict())
    assert checkpoint.keys() == {"state", "param_groups"}
    assert_tree_equal(checkpoint, reference.state_dict())
    resumed = make_hdo()
    with torch.no_grad():
        for target, source in zip(sources(resumed), sources(actual), strict=True):
            target.copy_(source)
    install(resumed)
    resumed.load_state_dict(copy.deepcopy(checkpoint))
    assert resumed._miles_streaming_gradients_installed
    for step in range(2, 5):
        set_grads(actual, step)
        set_grads(reference, step)
        set_grads(resumed, step)
        actual.step()
        reference.step()
        reference._h2d_stream.synchronize()
        resumed.step()
        assert_hdo_equal(actual, reference)
        assert_hdo_equal(resumed, reference)
        assert_released(resumed)


def test_opt_in_parser_and_disabled_noop():
    parser = add_optimizer_cpu_streaming_gradients_argument(argparse.ArgumentParser())
    assert parser.parse_args([]).optimizer_cpu_streaming_gradients is False
    assert parser.parse_args(["--optimizer-cpu-streaming-gradients"]).optimizer_cpu_streaming_gradients is True
    install_optimizer_cpu_streaming_gradients(SimpleNamespace(), object())


@pytest.mark.parametrize("change,match", [
    ({"optimizer_cpu_offload": False}, "requires --optimizer-cpu-offload"),
    ({"overlap_cpu_optimizer_d2h_h2d": False}, "requires --overlap-cpu"),
    ({"train_backend": "fsdp"}, "requires the Megatron"),
])
def test_incompatible_args_rejected(streams, change, match):
    hdo = make_hdo()
    with pytest.raises(RuntimeError, match=match):
        install_optimizer_cpu_streaming_gradients(args(**change), hdo)
    assert "step" not in hdo.__dict__


@pytest.mark.parametrize("defect,match", [
    ("batched", "exactly one"),
    ("mapping", "mapping"),
    ("missing_hook", "copy-back hook"),
    ("stream", "stream"),
    ("cached", "cached CPU gradients"),
    ("overlap", "per-parameter overlap"),
])
def test_capability_checks_are_atomic_across_wrappers(streams, defect, match):
    good, bad = make_hdo(), make_hdo()
    child = bad.cpu_optimizers[0]
    param = sources(child)[0]
    if defect == "batched":
        child.param_groups[0]["params"].append(sources(bad.cpu_optimizers[1])[0])
    elif defect == "mapping":
        del bad.cpu_copys_map_gpu_param[param]
    elif defect == "missing_hook":
        child._optimizer_step_post_hooks.clear()
    elif defect == "stream":
        bad._h2d_stream = object()
    elif defect == "cached":
        bad.cpu_copy_map_grad[param] = torch.ones_like(param)
    else:
        bad.overlap_cpu_optimizer_d2h_h2d = False
    wrapped = SimpleNamespace(chained_optimizers=[SimpleNamespace(optimizer=good), SimpleNamespace(optimizer=bad)])
    with pytest.raises(RuntimeError, match=match):
        install_optimizer_cpu_streaming_gradients(args(), wrapped)
    assert "step" not in good.__dict__
    assert "step" not in bad.__dict__


def test_install_is_instance_local_idempotent_and_closures_fail_before_mutation(streams):
    original_class_step = HybridDeviceOptimizer.step
    actual, other = install(make_hdo()), make_hdo()
    step = actual.step
    install_optimizer_cpu_streaming_gradients(
        args(), SimpleNamespace(chained_optimizers=[SimpleNamespace(optimizer=actual), actual])
    )
    assert actual.step is step
    assert HybridDeviceOptimizer.step is original_class_step
    assert "step" not in other.__dict__
    set_grads(actual, 0)
    before = copy.deepcopy(actual.state_dict())
    with pytest.raises(RuntimeError, match="closures are unsupported"):
        actual.step(lambda: None)
    assert_tree_equal(actual.state_dict(), before)
    actual.step()
    assert_released(actual)
