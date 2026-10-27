import sys
import types

import pytest
import torch

from miles.backends.megatron_utils.te_fused_adam import (
    MAX_HANDLES_PER_CALL,
    ChunkedMultiTensorApplier,
    install_fused_adam_chunking,
    split_tensor_lists,
)

# TE 2.17's NVTETensor pool: 20 MiB / sizeof(Tensor)
TE_HANDLE_POOL = 20321


class _TeApplier:
    """TE's multi_tensor_applier: one NVTETensor handle per tensor plus the noop flag, from a capped pool."""

    def __init__(self, pool: int | None = TE_HANDLE_POOL) -> None:
        self.pool = pool
        self.handles_per_call: list[int] = []

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        handles = 1 + sum(len(tensors) for tensors in tensor_lists)
        if self.pool is not None and handles > self.pool:
            raise RuntimeError(
                f"Cannot allocate {handles - self.pool} new NVTETensors. Maximum number of tensors reached: {self.pool}"
            )
        self.handles_per_call.append(handles)
        return op(2048 * 32, noop_flag_buffer, tensor_lists, *args)


def _adam_op(chunk_size, noop_flag_buffer, tensor_lists, lr, beta1, beta2, eps, step, mode, bias_correction, wd):
    """Elementwise AdamW over [grads, params, exp_avgs, exp_avg_sqs], the layout TE's multi_tensor_adam takes."""
    for grad, param, exp_avg, exp_avg_sq in zip(*tensor_lists, strict=True):
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = (exp_avg_sq / (1 - beta2**step)).sqrt_().add_(eps)
        param.mul_(1 - lr * wd).addcdiv_(exp_avg / (1 - beta1**step), denom, value=-lr)


def _expert_heavy_group(num_tensors: int, seed: int = 0) -> list[list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    grads = [torch.randn(3, generator=generator) for _ in range(num_tensors)]
    params = [torch.randn(3, generator=generator) for _ in range(num_tensors)]
    return [grads, params, [torch.zeros(3) for _ in grads], [torch.zeros(3) for _ in grads]]


def _adam_steps(applier, tensor_lists, steps: int = 2) -> None:
    for step in range(1, steps + 1):
        applier(_adam_op, torch.zeros(1, dtype=torch.int), tensor_lists, 1e-3, 0.9, 0.999, 1e-8, step, 1, 1, 0.01)


def test_an_expert_heavy_group_overflows_te_unchunked_and_updates_identically_chunked():
    # Qwen3.6-35B-A3B at EP4: 64 local experts x 40 layers x 2 weights, plus the dense params
    num_tensors = 64 * 40 * 2 + 80
    with pytest.raises(RuntimeError, match="Maximum number of tensors reached: 20321"):
        _adam_steps(_TeApplier(), _expert_heavy_group(num_tensors))

    reference = _expert_heavy_group(num_tensors)
    _adam_steps(_TeApplier(pool=None), reference)
    te_applier = _TeApplier()
    chunked = _expert_heavy_group(num_tensors)
    _adam_steps(ChunkedMultiTensorApplier(te_applier), chunked)

    assert max(te_applier.handles_per_call) <= MAX_HANDLES_PER_CALL + 1
    for chunked_tensors, reference_tensors in zip(chunked, reference, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(chunked_tensors, reference_tensors, strict=True))


@pytest.mark.parametrize("num_lists", [4, 5, 8])
@pytest.mark.parametrize("num_tensors", [1, 7, 100])
def test_chunks_are_aligned_bounded_and_cover_every_tensor_once(num_lists, num_tensors):
    tensor_lists = [[(list_index, i) for i in range(num_tensors)] for list_index in range(num_lists)]

    chunks = split_tensor_lists(tensor_lists, max_handles=20)

    for chunk in chunks:
        assert len(chunk) == num_lists
        assert len({len(tensors) for tensors in chunk}) == 1
        assert sum(len(tensors) for tensors in chunk) <= max(20, num_lists)
    for list_index in range(num_lists):
        assert [item for chunk in chunks for item in chunk[list_index]] == tensor_lists[list_index]


def test_a_call_within_the_budget_reaches_te_unchanged():
    calls = []
    tensor_lists = [[1, 2], [3, 4]]

    ChunkedMultiTensorApplier(lambda *call: calls.append(call), max_handles=4)("op", "noop", tensor_lists, 0.1, 7)

    assert calls == [("op", "noop", tensor_lists, 0.1, 7)]
    assert calls[0][2] is tensor_lists


def test_misaligned_lists_are_refused():
    with pytest.raises(AssertionError, match="must be aligned"):
        split_tensor_lists([[1, 2], [3]], max_handles=1)


def _fake_te(monkeypatch: pytest.MonkeyPatch, applier) -> types.ModuleType:
    fused_adam = types.ModuleType("transformer_engine.pytorch.optimizers.fused_adam")
    fused_adam.multi_tensor_applier = applier
    optimizers = types.ModuleType("transformer_engine.pytorch.optimizers")
    optimizers.fused_adam = fused_adam
    for name, module in {
        "transformer_engine": types.ModuleType("transformer_engine"),
        "transformer_engine.pytorch": types.ModuleType("transformer_engine.pytorch"),
        "transformer_engine.pytorch.optimizers": optimizers,
        "transformer_engine.pytorch.optimizers.fused_adam": fused_adam,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return fused_adam


def test_install_wraps_fused_adam_once(monkeypatch):
    te_applier = _TeApplier()
    fused_adam = _fake_te(monkeypatch, te_applier)

    install_fused_adam_chunking()
    install_fused_adam_chunking()

    assert isinstance(fused_adam.multi_tensor_applier, ChunkedMultiTensorApplier)
    assert fused_adam.multi_tensor_applier.applier is te_applier


def test_install_is_a_no_op_without_transformer_engine(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformer_engine", None)

    install_fused_adam_chunking()  # training without TE (e.g. apex FusedAdam) must not fail here
