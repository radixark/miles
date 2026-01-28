import atexit
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist


def _get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


class Replay:
    def __init__(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list: List[torch.Tensor] = []

    def record(self, top_indices: torch.Tensor):
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    def pop_forward(self) -> torch.Tensor:
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def pop_backward(self) -> torch.Tensor:
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []

    def clear_forward(self):
        self.forward_index = 0


class BaseReplayManager:
    name: str = ""
    filename: str = ""

    def __init__(self):
        self.replays: List[Replay] = []
        self.current: Optional[Replay] = None
        self.enabled = False
        self.stage = "fallthrough"
        self._save_path: Optional[str] = None
        self._save_registered = False

    def create_replay(self) -> Replay:
        replay = Replay()
        self.replays.append(replay)
        return replay

    def set_current(self, replay: Replay):
        self.current = replay

    def get_current(self) -> Optional[Replay]:
        return self.current

    def clear_all(self):
        for replay in self.replays:
            replay.clear()

    def clear_all_forward(self):
        for replay in self.replays:
            replay.clear_forward()

    def save_all_to_files(self, base_path: str):
        if _get_rank() != 0:
            return
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        data = [r.top_indices_list for r in self.replays]
        p = base_path / self.filename
        print(f"[{self.__class__.__name__}] save_all_to_files {p}")
        torch.save(data, p)

    def load_all_from_files(self, base_path: str):
        p = Path(base_path) / self.filename
        print(f"[{self.__class__.__name__}] load_all_from_files {p}")
        data = torch.load(p, weights_only=False)
        for replay, indices_list in zip(self.replays, data):
            replay.top_indices_list = indices_list
            replay.forward_index = 0
            replay.backward_index = 0

    def set_save_path(self, path: str):
        self._save_path = path
        if not self._save_registered and path:
            atexit.register(lambda: self.save_all_to_files(path))
            self._save_registered = True

    def get_topk_fn(self, old_topk_fn, layer_id: Optional[int] = None):
        manager = self

        def new_topk_fn(scores, topk, **kwargs):
            def get_probs_and_top_indices(top_indices):
                if -1 in top_indices:
                    return old_topk_fn(scores, topk, **kwargs)
                else:
                    return scores.gather(1, top_indices), top_indices

            if not manager.enabled:
                return old_topk_fn(scores, topk, **kwargs)

            stage = manager.stage
            replay = manager.get_current()

            if stage == "fallthrough":
                return old_topk_fn(scores, topk, **kwargs)

            elif stage == "record":
                probs, top_indices = old_topk_fn(scores, topk, **kwargs)
                replay.record(top_indices)
                return probs, top_indices

            elif stage == "replay_forward":
                top_indices = replay.pop_forward()
                assert (
                    top_indices.shape[0] == scores.shape[0]
                    and top_indices.shape[1] == topk
                ), f"[{_get_rank()}] top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                return get_probs_and_top_indices(top_indices)

            elif stage == "replay_backward":
                top_indices = replay.pop_backward()
                assert (
                    top_indices.shape[0] == scores.shape[0]
                    and top_indices.shape[1] == topk
                ), f"top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                return get_probs_and_top_indices(top_indices)

            else:
                return old_topk_fn(scores, topk, **kwargs)

            if os.environ.get("MILES_CHECK_REPLAY_RESULT", "0") == "1":
                orig_probs, orig_top_indices = old_topk_fn(scores, topk, **kwargs)
                try:
                    assert orig_top_indices.shape == top_indices.shape, \
                        f"Shape mismatch: orig {orig_top_indices.shape} vs replay {top_indices.shape}"
                    num_tokens, k = top_indices.shape
                    min_match = k // 2
                    for i in range(num_tokens):
                        orig_set = set(orig_top_indices[i].tolist())
                        replay_set = set(top_indices[i].tolist())
                        num_match = len(orig_set & replay_set)
                        assert num_match >= min_match, \
                            f"rank {_get_rank()}: Token {i}: only {num_match}/{k} indices match (need at least {min_match})"
                except Exception as e:
                    print(f"Routing replay stage: {stage}, layer: {layer_id}, rank: {_get_rank()}", flush=True)
                    torch.set_printoptions(threshold=float('inf'))
                    print(f"original top_indices: {orig_top_indices}", flush=True)
                    print(f"replay top_indices: {top_indices}", flush=True)
                    raise e

            return probs, top_indices

        return new_topk_fn

    def register_to_module(self, module, attr_name: str):
        if not self.enabled:
            return
        replay = self.create_replay()
        setattr(module, attr_name, replay)
        manager = self

        def pre_forward_hook(*args, **kwargs):
            manager.set_current(replay)

        module.register_forward_pre_hook(pre_forward_hook)


class RoutingReplayManager(BaseReplayManager):
    name = "routing"
    filename = "routing_replay.pt"
    data_key = "rollout_routed_experts"
    needs_moe_layer_indices = True


class IndexerReplayManager(BaseReplayManager):
    name = "indexer"
    filename = "indexer_replay.pt"
    data_key = "rollout_indexer_topk"
    needs_moe_layer_indices = False


routing_replay_manager = RoutingReplayManager()
indexer_replay_manager = IndexerReplayManager()
all_replay_managers = [routing_replay_manager, indexer_replay_manager]