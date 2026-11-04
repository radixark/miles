from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["replay"])

import torch
from torch._dynamo.testing import CompileCounter, CompileCounterWithBackend
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from miles.backends.torchtitan_utils import routing_replay
from miles.utils import replay_base
from miles.utils.replay_base import BaseReplayManager, RoutingReplayManager


class _FakeReplay:
    def __init__(self, *top_indices):
        self.top_indices = list(top_indices)
        self.staged_forward_ids = None

    def pop_forward(self):
        return self.top_indices.pop(0)

    def pop_backward(self):
        return self.pop_forward()


def _topk(scores, topk):
    return torch.topk(scores, topk, dim=1).indices.to(torch.int32)


def _make_replay_manager(top_indices):
    manager = BaseReplayManager()
    manager.enable_check_replay_result = False
    manager.enabled = True
    manager.stage = "replay_forward"
    manager.set_current(_FakeReplay(top_indices))
    return manager


def test_get_topk_fn_fills_all_invalid_rows_with_arange():
    # an all-(-1) row is a masked/pad token; fill it with arange to avoid reading
    # invalid (-1) positions downstream
    scores = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    manager = _make_replay_manager(torch.tensor([[-1, -1, -1]], dtype=torch.int32))

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    torch.testing.assert_close(topk_fn(scores, 3), torch.tensor([[0, 1, 2]], dtype=torch.int32))


def test_get_topk_fn_preserves_partial_padding():
    # a row with some valid picks keeps its -1 padding (only all-(-1) rows are filled)
    scores = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    replayed_top_indices = torch.tensor([[2, -1, -1]], dtype=torch.int32)
    manager = _make_replay_manager(replayed_top_indices)

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    torch.testing.assert_close(topk_fn(scores, 3), replayed_top_indices)


def _make_routing_replay_manager(top_indices):
    manager = RoutingReplayManager()
    manager.enabled = True
    manager.stage = "replay_forward"
    manager.set_current(_FakeReplay(top_indices))
    return manager


def test_the_padding_fill_traces_without_a_graph_break():
    """The torchtitan trainer can compile each block with fullgraph=True, where a data-dependent branch fails."""
    scores = torch.tensor([[0.0, 4.0, 1.0, 3.0, 2.0]] * 2)
    manager = _make_routing_replay_manager(torch.tensor([[-1, -1], [2, 0]], dtype=torch.int32))
    topk_fn = torch.compile(manager.get_topk_fn(_topk, return_probs=False), backend="eager", fullgraph=True)

    torch.testing.assert_close(topk_fn(scores, 2), torch.tensor([[0, 1], [2, 0]], dtype=torch.int32))


def test_a_staged_forward_replays_without_popping(monkeypatch):
    """torchtitan pops ids before the compiled blocks run; the block reads the staged ids instead of popping."""
    monkeypatch.setattr(replay_base.current_platform, "dynamic_shapes", False)
    monkeypatch.setattr(replay_base.current_platform, "current_device", lambda: torch.device("cpu"))
    manager = RoutingReplayManager()
    manager.enabled = True
    manager.stage = "replay_forward"
    replay = manager.create_replay()
    replay.record(torch.tensor([[3, 1]], dtype=torch.int32))
    replay.record(torch.tensor([[4, 2]], dtype=torch.int32))
    manager.set_current(replay)

    replay.staged_forward_ids = replay.pop_forward()
    topk_fn = manager.get_topk_fn(_topk, return_probs=False)
    scores = torch.zeros(1, 5)

    torch.testing.assert_close(topk_fn(scores, 2), torch.tensor([[3, 1]], dtype=torch.int32))
    torch.testing.assert_close(topk_fn(scores, 2), torch.tensor([[3, 1]], dtype=torch.int32))
    assert replay.forward_index == 1


def test_without_dynamic_shapes_rollout_ids_stay_on_the_host_until_popped(monkeypatch):
    """Without dynamic shapes, record keeps a plain host copy and pop_forward moves only the popped entry."""
    monkeypatch.setattr(replay_base.current_platform, "dynamic_shapes", False)
    monkeypatch.setattr(replay_base.current_platform, "current_device", lambda: torch.device("meta"))
    replay = RoutingReplayManager().create_replay()
    replay.record(torch.tensor([[3, 1]], dtype=torch.int32))

    assert replay.top_indices_list[0].device.type == "cpu"
    assert replay.pop_forward().device.type == "meta"


def test_staged_ids_compile_once_across_microbatches(monkeypatch):
    """A compiled block that pops its own ids guards on forward_index and recompiles every microbatch."""
    monkeypatch.setattr(replay_base.current_platform, "dynamic_shapes", False)
    monkeypatch.setattr(replay_base.current_platform, "current_device", lambda: torch.device("cpu"))
    manager = RoutingReplayManager()
    manager.enabled = True
    manager.stage = "replay_forward"
    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    class _Router(torch.nn.Module):
        def forward(self, scores):
            return topk_fn(scores, 2)

    routers = [_Router() for _ in range(3)]
    replays = [manager.register_to_module(router, "routing_replay", stream_idx=i) for i, router in enumerate(routers)]
    ids = torch.randint(0, 5, (4, 3, 8, 2), dtype=torch.int32)  # [microbatch, layer, token, topk]
    for microbatch_ids in ids:
        for replay, layer_ids in zip(replays, microbatch_ids, strict=True):
            replay.record(layer_ids)

    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = [torch.compile(router, backend=counter, fullgraph=True, dynamic=False) for router in routers]
    scores = torch.rand(8, 5)
    for microbatch_ids in ids:
        with routing_replay._stage_forward_ids(replays):
            for router, layer_ids in zip(compiled, microbatch_ids, strict=True):
                torch.testing.assert_close(router(scores), layer_ids)

    assert counter.frame_count == 1, f"{counter.frame_count} compiles for {len(ids)} microbatches"


def test_staged_ids_compile_once_under_activation_checkpointing(monkeypatch):
    """torchtitan compiles each checkpointed block fullgraph: the router pre-hook is a forward side effect
    under the checkpoint, and the recompute in backward runs inside the compiled graph without popping."""
    monkeypatch.setattr(replay_base.current_platform, "dynamic_shapes", False)
    monkeypatch.setattr(replay_base.current_platform, "current_device", lambda: torch.device("cpu"))
    manager = RoutingReplayManager()
    manager.enabled = True
    topk_fn = manager.get_topk_fn(_topk, return_probs=True)

    class _Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Linear(5, 5)

        def forward(self, hidden):
            probs, _ = topk_fn(self.gate(hidden).softmax(dim=-1), 2)
            return probs.sum(dim=-1, keepdim=True) * hidden

    routers = [_Router() for _ in range(3)]
    replays = [manager.register_to_module(router, "routing_replay", stream_idx=i) for i, router in enumerate(routers)]
    ids = torch.randint(0, 5, (4, 3, 8, 2), dtype=torch.int32)  # [microbatch, layer, token, topk]
    for microbatch_ids in ids:
        for replay, layer_ids in zip(replays, microbatch_ids, strict=True):
            replay.record(layer_ids)

    torch._dynamo.reset()
    counter = CompileCounterWithBackend("aot_eager")
    blocks = [checkpoint_wrapper(router) for router in routers]
    for block in blocks:
        block.compile(backend=counter, fullgraph=True, dynamic=False)
    with torch._dynamo.config.patch(skip_fwd_side_effects_in_bwd_under_checkpoint=False):
        routing_replay._allow_forward_side_effects_in_compiled_checkpoints()  # what install() does
        for _ in ids:
            hidden = torch.rand(8, 5, requires_grad=True)
            manager.stage = "replay_forward"
            with routing_replay._stage_forward_ids(replays):
                for block in blocks:
                    hidden = block(hidden)
            manager.stage = "replay_backward"
            hidden.sum().backward()

    assert counter.frame_count == 1, f"{counter.frame_count} compiles for {len(ids)} microbatches"
    assert [replay.backward_index for replay in replays] == [0, 0, 0]
