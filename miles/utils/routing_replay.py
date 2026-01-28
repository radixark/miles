import atexit
import os
from pathlib import Path
from typing import Optional

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
        p = base_path / "routing_replay.pt"
        print(f"[RoutingReplay] save_all_to_files {p}")
        torch.save(data, p)

    @staticmethod
    def load_all_from_files(base_path: str):
        p = Path(base_path) / "routing_replay.pt"
        print(f"[RoutingReplay] load_all_from_files {p}")
        data = torch.load(p, weights_only=False)
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


def get_routing_replay_compute_topk(old_compute_topk, layer_id: Optional[int] = None):
    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        def get_probs_and_top_indices(top_indices):
            if -1 in top_indices: # skip at padding
                return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk) 
            else:
                return scores.gather(1, top_indices), top_indices
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
                probs, top_indices = get_probs_and_top_indices(top_indices)
            elif routing_replay_stage == "replay_backward":
                top_indices = ROUTING_REPLAY.pop_backward()
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                probs, top_indices = get_probs_and_top_indices(top_indices)
            elif routing_replay_stage == "replay_from_file":
                _maybe_load_from_file()
                top_indices = ROUTING_REPLAY.pop_forward()
                num_tokens = scores.shape[0]
                # vanilla handle TP1 vs TP4 in our hacky comparison test
                if top_indices.shape[0] != num_tokens:
                    tp_size = top_indices.shape[0] // num_tokens
                    tp_rank = _get_rank() % tp_size
                    top_indices = top_indices[tp_rank * num_tokens : (tp_rank + 1) * num_tokens]
                assert top_indices.shape[0] == num_tokens and top_indices.shape[1] == topk
                probs, top_indices = get_probs_and_top_indices(top_indices)
            if os.environ.get("CHECK_ROUTING_REPLAY_RESULT", "0") == "1":
                orig_probs, orig_top_indices = old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
                try:
                    assert orig_top_indices.shape == top_indices.shape, \
                        f"Shape mismatch: orig_top_indices {orig_top_indices.shape} vs replay_top_indices {top_indices.shape}"
                    
                    # check at least half of the indices match for each token    
                    num_tokens, k = top_indices.shape
                    min_match = k // 2
                    for i in range(num_tokens):
                        orig_set = set(orig_top_indices[i].tolist()) 
                        replay_set = set(top_indices[i].tolist())
                        num_match = len(orig_set & replay_set)
                        assert num_match >= min_match, f"rank {_get_rank()}: Token {i}: only {num_match}/{k} indices match (need at least {min_match})"

                except Exception as e:
                    print(f"Routing replay stage: {routing_replay_stage}, layer: {layer_id}, rank: {_get_rank()}", flush=True)
                    torch.set_printoptions(threshold=float('inf'))
                    print(f"original top_indices: {orig_top_indices}", flush=True)
                    print(f"replay top_indices: {top_indices}", flush=True)
                    raise e

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
