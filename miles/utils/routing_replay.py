import atexit
import os
from pathlib import Path

import torch
import torch.distributed as dist


ROUTING_REPLAY = None
_file_loaded = False
_save_registered = False


def set_routing_replay(replay):
    global ROUTING_REPLAY
    ROUTING_REPLAY = replay


def _get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


class RoutingReplay:
    all_routing_replays = []

    def __init__(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        RoutingReplay.all_routing_replays.append(self)

    def record(self, top_indices):
        # offload top_indices to CPU pinned memory
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    def pop_forward(self):
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def pop_backward(self):
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []

    def clear_forward(self):
        self.forward_index = 0

    @staticmethod
    def clear_all():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear()

    @staticmethod
    def clear_all_forward():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear_forward()

    @staticmethod
    def save_all_to_files(base_path: str):
        if _get_rank() != 0:
            return
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        data = [r.top_indices_list for r in RoutingReplay.all_routing_replays]
        torch.save(data, base_path / "routing_replay.pt")

    @staticmethod
    def load_all_from_files(base_path: str):
        data = torch.load(Path(base_path) / "routing_replay.pt", weights_only=False)
        for replay, indices_list in zip(RoutingReplay.all_routing_replays, data):
            replay.top_indices_list = indices_list
            replay.forward_index = 0
            replay.backward_index = 0


def _maybe_load_from_file():
    global _file_loaded
    if _file_loaded:
        return
    load_path = os.environ.get("ROUTING_REPLAY_LOAD_PATH")
    if load_path:
        RoutingReplay.load_all_from_files(load_path)
        _file_loaded = True


def _register_save_on_exit():
    global _save_registered
    if _save_registered:
        return
    dump_path = os.environ.get("ROUTING_REPLAY_DUMP_PATH")
    if dump_path:
        atexit.register(lambda: RoutingReplay.save_all_to_files(dump_path))
        _save_registered = True


def get_routing_replay_compute_topk(old_compute_topk):
    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            routing_replay_stage = os.environ["ROUTING_REPLAY_STAGE"]
            if routing_replay_stage == "fallthrough":
                return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            if routing_replay_stage == "record":
                _register_save_on_exit()
                probs, top_indices = old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
                ROUTING_REPLAY.record(top_indices)
            elif routing_replay_stage == "replay_forward":
                top_indices = ROUTING_REPLAY.pop_forward()
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"[{torch.distributed.get_rank()}] top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                probs = scores.gather(1, top_indices)
            elif routing_replay_stage == "replay_backward":
                top_indices = ROUTING_REPLAY.pop_backward()
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                probs = scores.gather(1, top_indices)
            elif routing_replay_stage == "replay_from_file":
                _maybe_load_from_file()
                top_indices = ROUTING_REPLAY.pop_forward()
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                probs = scores.gather(1, top_indices)
            return probs, top_indices
        else:
            return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

    return compute_topk


def register_routing_replay(module):
    if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
        module.routing_replay = RoutingReplay()

        def pre_forward_hook(*args, **kwargs):
            set_routing_replay(module.routing_replay)

        module.register_forward_pre_hook(pre_forward_hook)
