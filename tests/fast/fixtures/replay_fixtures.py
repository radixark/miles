"""Shared helpers for routing-replay tests."""

from miles.utils.replay_base import routing_replay_manager


class CpuReplay:
    """Device-neutral stand-in for ``Replay``.

    The real one pins its host buffer and returns indices on ``torch.cuda.current_device()``;
    routing-replay suites are registered stage-a-cpu. The two independent cursors are the
    behaviour under test and are reproduced exactly.
    """

    def __init__(self, stream_idx=0):
        self.stream_idx = stream_idx
        self.recorded = []
        self.forward_index = 0
        self.backward_index = 0

    def record(self, top_indices):
        self.recorded.append(top_indices.detach().clone())

    def pop_forward(self):
        top_indices = self.recorded[self.forward_index]
        self.forward_index += 1
        return top_indices

    def pop_backward(self):
        top_indices = self.recorded[self.backward_index]
        self.backward_index += 1
        return top_indices

    def clear_forward(self):
        self.forward_index = 0


def wire_replay(module, install) -> CpuReplay:
    """Install an arch hook on ``module`` and make a fresh CPU replay the current stream."""
    install(module)
    replay = CpuReplay()
    routing_replay_manager.replays.append(replay)
    routing_replay_manager.set_current(replay)
    return replay


def reset_routing_replay_manager(enabled: bool) -> None:
    routing_replay_manager.enabled = enabled
    routing_replay_manager.enable_check_replay_result = False
    routing_replay_manager.replays = []
    routing_replay_manager.current = None
    routing_replay_manager.stage = "fallthrough"
