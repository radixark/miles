import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


class Replay:
    def __init__(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list: list[torch.Tensor] = []

    def record(self, top_indices: torch.Tensor):
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    def pop_forward(self) -> torch.Tensor:
        if self.forward_index >= len(self.top_indices_list):
            shapes = [
                t.shape if isinstance(t, torch.Tensor) else f"non-tensor({type(t)})" for t in self.top_indices_list
            ]
            raise IndexError(
                f"pop_forward out of range: forward_index={self.forward_index}, "
                f"len(top_indices_list)={len(self.top_indices_list)}, shapes={shapes}"
            )
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
        self.replays: list[Replay] = []
        self.current: Replay | None = None
        self.enabled = False
        self.stage = "fallthrough"

    def create_replay(self) -> Replay:
        replay = Replay()
        self.replays.append(replay)
        return replay

    def set_current(self, replay: Replay):
        self.current = replay

    def get_current(self) -> Replay | None:
        return self.current

    def clear_all(self):
        for replay in self.replays:
            replay.clear()

    def clear_all_forward(self):
        for replay in self.replays:
            replay.clear_forward()

    def get_topk_fn(self, old_topk_fn, return_probs):
        manager = self

        def shape_sanity_check(replay_top_indices, scores, topk):
            n_replay_tokens, n_actual_tokens = replay_top_indices[..., 0].numel(), scores[..., 0].numel()
            assert (
                n_replay_tokens == n_actual_tokens
            ), f"rank {_get_rank()}: replay n_tokens {n_replay_tokens} does not match scores n_tokens {n_actual_tokens}"

            assert (
                replay_top_indices.shape[1] >= topk
            ), f"not enough topk indices in replay, got {replay_top_indices.shape[1]}, expected at least {topk}"

        def new_topk_fn(scores, topk, *args, **kwargs):
            def get_probs_and_top_indices(top_indices, return_probs):
                if return_probs:
                    if -1 in top_indices:
                        return old_topk_fn(scores, topk, *args, **kwargs)
                    else:
                        return scores.gather(1, top_indices), top_indices
                else:
                    return top_indices

            if not manager.enabled:
                return old_topk_fn(scores, topk, *args, **kwargs)

            stage = manager.stage
            replay = manager.get_current()

            if stage == "fallthrough":
                return old_topk_fn(scores, topk, *args, **kwargs)

            elif stage == "record":
                result = old_topk_fn(scores, topk, *args, **kwargs)
                if return_probs:
                    probs, top_indices = result
                else:
                    top_indices = result
                replay.record(top_indices)
                return result

            elif stage == "replay_forward":
                replay_top_indices = replay.pop_forward()

                shape_sanity_check(replay_top_indices, scores, topk)
                top_indices = replay_top_indices[..., :topk].view(scores.shape[:-1] + (topk,))

                self.check_replay_result(old_topk_fn, scores, topk, top_indices, **kwargs)

                return get_probs_and_top_indices(top_indices, return_probs)

            elif stage == "replay_backward":
                replay_top_indices = replay.pop_backward()

                shape_sanity_check(replay_top_indices, scores, topk)
                top_indices = replay_top_indices[..., :topk].view(scores.shape[:-1] + (topk,))

                self.check_replay_result(old_topk_fn, scores, topk, top_indices, **kwargs)

                return get_probs_and_top_indices(top_indices, return_probs)
            else:
                return old_topk_fn(scores, topk, *args, **kwargs)

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

    def check_replay_result(self, old_topk_fn, scores, topk, top_indices, **kwargs):
        if os.environ.get("MILES_CHECK_REPLAY_RESULT", "0") == "0":
            return

        orig_top_indices = old_topk_fn(scores, topk, **kwargs)
        if isinstance(orig_top_indices, tuple):
            _, orig_top_indices = orig_top_indices

        try:
            orig_flat = orig_top_indices.view(-1, orig_top_indices.shape[-1])
            replay_flat = top_indices.view(-1, top_indices.shape[-1])
            for i, (orig_idx, replay_idx) in enumerate(zip(orig_flat, replay_flat, strict=True)):
                orig_set = set(orig_idx.tolist()) - {-1}
                replay_set = set(replay_idx.tolist()) - {-1}
                if len(replay_set) == 0:
                    continue
                if len(orig_set & replay_set) < len(replay_set) * self.thresh_check_replay_result:
                    raise AssertionError(
                        f"token {i} failed replay check, {len(orig_set & replay_set)=} {len(replay_set)=}"
                    )
        except Exception as e:
            logger.error(f"Rollout Replay Check Failed - Stage: {self.stage}, rank: {_get_rank()}")
            logger.error(f"original top_indices: {orig_top_indices}")
            logger.error(f"replay top_indices (padding removed): {top_indices}")
            raise e


class RoutingReplayManager(BaseReplayManager):
    name = "routing"
    filename = "routing_replay.pt"
    data_key = "rollout_routed_experts"
    if_sp_region = True
    thresh_check_replay_result = 0.5


routing_replay_manager = RoutingReplayManager()
all_replay_managers = [routing_replay_manager]
