"""CPU coverage of request-scoped replay using the real queue, packing and top-k code."""

from argparse import Namespace
from copy import deepcopy

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci
from torch.utils.checkpoint import checkpoint

from miles.backends.megatron_utils.lora.replay import rollout_routing_replay
from miles.backends.training_utils import parallel
from miles.backends.training_utils.replay_data import register_replay_list_sequential
from miles.utils.replay_base import Replay, routing_replay_manager

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])


class _Iterator:
    """Exercise replay's iterator contract with deliberately reordered microbatches."""

    def __init__(self, data):
        self.data = data
        self.order = [1, 0]
        self.offset = 0

    def reset(self):
        self.offset = 0

    def get_next(self, keys):
        index = self.order[self.offset]
        self.offset += 1
        return {key: [self.data[key][index]] if key in self.data else None for key in keys}


class _Router(torch.nn.Module):
    def __init__(self):
        super().__init__()
        routing_replay_manager.register_to_module(self, "replay")
        self.topk = routing_replay_manager.get_topk_fn(torch.topk, return_probs=True)

    def forward(self, scores):
        return self.topk(scores, 2)


@pytest.fixture
def replay_env(monkeypatch):
    manager = routing_replay_manager
    for key, value in dict(
        enabled=True,
        stage="fallthrough",
        replays=[],
        current=None,
        register_replay_list_func=register_replay_list_sequential,
        enable_check_replay_result=False,
    ).items():
        monkeypatch.setattr(manager, key, value)
    # Only the pinned-host allocation and CUDA destination are replaced. The real
    # Replay cursors, manager hook, gather, and fill_replay_data run unchanged.
    monkeypatch.setattr(Replay, "record", lambda self, picks: self.top_indices_list.append(picks.clone()))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    group = Namespace(rank=0, size=1)
    monkeypatch.setattr(parallel, "_parallel_state", Namespace(tp=group, cp=group))
    args = Namespace(
        use_rollout_routing_replay=True,
        moe_router_fusion=False,
        qkv_format="thd",
        allgather_cp=False,
        sequence_parallel=False,
        data_pad_size_multiplier=1,
    )
    routes = [
        torch.tensor([[[0, 2], [1, 0]], [[2, 1], [0, 2]]], dtype=torch.int32),
        torch.tensor([[[1, 2], [2, 0]]], dtype=torch.int32),
    ]
    data = {"tokens": [torch.tensor([1, 2, 3]), torch.tensor([4, 5])], "rollout_routed_experts": routes}
    return args, data


@pytest.mark.parametrize("mode", ["forward_only", "train", "recompute"])
def test_forward_and_recomputed_backward_use_exact_captured_indices(replay_env, mode):
    args, data = replay_env
    manager = routing_replay_manager
    routers = [_Router(), _Router()]
    iterator = _Iterator(data)
    captured = deepcopy(data[manager.data_key])
    scores = torch.tensor([[1.0, 2.0, 3.0]]).expand(3, -1).clone().requires_grad_()

    with rollout_routing_replay(args, routers, data, [iterator], [2]):
        assert manager.stage == "replay_backward"
        assert iterator.offset == 0
        assert manager.data_key not in data
        expected_grad = torch.zeros_like(scores)
        for index in iterator.order:
            total = 0
            length = len(data["tokens"][index])
            # This matches the shared Megatron forward step's stage scope.
            manager.stage = "replay_forward"
            for layer, router in enumerate(routers):
                expected = torch.cat([captured[index][:, layer], torch.tensor([[0, 1]], dtype=torch.int32)])
                if mode == "recompute":
                    values, picks = checkpoint(router, scores[:length], use_reentrant=True)
                else:
                    values, picks = router(scores[:length])
                torch.testing.assert_close(picks, expected)
                total = total + values.sum()
                expected_grad[:length].scatter_add_(1, expected.long(), torch.ones_like(expected, dtype=scores.dtype))
            manager.stage = "replay_backward"
            if mode != "forward_only":
                total.backward()
        if mode != "forward_only":
            torch.testing.assert_close(scores.grad, expected_grad)
        else:
            assert scores.grad is None
        for stream in manager.replays:
            assert stream.forward_index == 2
            assert stream.backward_index == (2 if mode == "recompute" else 0)
    assert manager.enabled and manager.stage == "fallthrough"
    for stream in manager.replays:
        assert stream.top_indices_list == []
        assert stream.forward_index == stream.backward_index == 0


def test_replay_then_ordinary_request_uses_normal_topk(replay_env):
    args, data = replay_env
    manager = routing_replay_manager
    router = _Router()
    scores = torch.tensor([[1.0, 2.0, 3.0]]).expand(2, -1)
    with rollout_routing_replay(args, [router], data, [_Iterator(data)], [2]):
        manager.stage = "replay_forward"
        _, picks = router(scores)
        assert picks[0].tolist() == [1, 2]
    with rollout_routing_replay(args, [router], {}, [], [1]):
        manager.stage = "replay_forward"
        values, picks = router(scores)
        torch.testing.assert_close(picks, scores.topk(2).indices)
        torch.testing.assert_close(values, scores.topk(2).values)
    assert manager.enabled and manager.stage == "fallthrough"


@pytest.mark.parametrize("failure", ["fill", "forward"])
def test_errors_clear_queues_and_restore_capability(replay_env, failure):
    args, data = replay_env
    manager = routing_replay_manager
    _Router()
    iterator = _Iterator(data)
    if failure == "fill":
        # One microbatch has already loaded when this second one fails.
        data[manager.data_key][0] = torch.zeros((1, 2, 2), dtype=torch.int64)
    with pytest.raises((AssertionError, RuntimeError)):
        with rollout_routing_replay(args, [], data, [iterator], [2]):
            manager.stage = "replay_forward"
            raise RuntimeError("forward failed")
    assert manager.enabled and manager.stage == "fallthrough"
    assert all(stream.top_indices_list == [] for stream in manager.replays)


@pytest.mark.parametrize("disabled", ["flag", "hooks", "fusion"])
def test_unsupported_configuration_fails_before_running(replay_env, disabled):
    args, data = replay_env
    manager = routing_replay_manager
    if disabled == "flag":
        args.use_rollout_routing_replay = False
    elif disabled == "hooks":
        manager.enabled = False
    else:
        args.moe_router_fusion = True
    with pytest.raises(ValueError, match="routing.replay"):
        with rollout_routing_replay(args, [], data, [_Iterator(data)], [2]):
            pytest.fail("invalid replay configuration reached the model")
