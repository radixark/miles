import os
import atexit
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core import mpu
from megatron.core.transformer.deepseek_v4_cp_utils import natural_to_zigzag_slice
from sglang.srt.debug_utils.dumper import dumper

import logging
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
            shapes = [t.shape if isinstance(t, torch.Tensor) else f"non-tensor({type(t)})" for t in self.top_indices_list]
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
        self._save_path: str | None = None
        self._save_registered = False

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

    def save_all_to_files(self, base_path: str):
        if _get_rank() != 0:
            return
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        data = [r.top_indices_list for r in self.replays]
        p = base_path / self.filename
        print(f"[{self.__class__.__name__}] save_all_to_files {p}")
        torch.save(data, p)

    # Hacky, only support raw file obtained from 1rank
    def load_all_from_files(self, base_path: str, sequence_parallel: bool = False):
        p = Path(base_path) / self.filename
        print(f"[{self.__class__.__name__}] load_all_from_files {p}")
        data = torch.load(p, weights_only=False)

        cp_size = mpu.get_context_parallel_world_size() if mpu.is_initialized() else 1
        cp_rank = mpu.get_context_parallel_rank() if mpu.is_initialized() else 0
        tp_size = mpu.get_tensor_model_parallel_world_size() if mpu.is_initialized() else 1
        tp_rank = mpu.get_tensor_model_parallel_rank() if mpu.is_initialized() else 0

        do_sp_slice = sequence_parallel and self.if_sp_region and tp_size > 1

        for replay_idx, (replay, indices_list) in enumerate(zip(self.replays, data)):
            shapes_before = [t.shape if isinstance(t, torch.Tensor) else torch.tensor(t).shape for t in indices_list]
            if self.squeeze_batch_for_load_from_file:
                indices_list = [t.squeeze(0) for t in indices_list]
            if cp_size > 1:
                indices_list = [natural_to_zigzag_slice(t, dim=0, cp_size=cp_size, cp_rank=cp_rank) for t in indices_list]
            if do_sp_slice:
                def sp_slice(t):
                    seqlen = t.size(0)
                    assert seqlen % tp_size == 0, f"seqlen {seqlen} not divisible by tp_size {tp_size}"
                    start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
                    return t[start:end]
                indices_list = [sp_slice(t) for t in indices_list]
            shapes_after = [t.shape if isinstance(t, torch.Tensor) else torch.tensor(t).shape for t in indices_list]
            print(f"[{self.__class__.__name__}] replay[{replay_idx}]: cp_size={cp_size}, cp_rank={cp_rank}, "
                  f"tp_size={tp_size}, tp_rank={tp_rank}, sp={sequence_parallel}, if_sp_region={self.if_sp_region}, "
                  f"n_tensors={len(indices_list)}, shapes_before={shapes_before}, shapes_after={shapes_after}")
            replay.top_indices_list = indices_list
            replay.forward_index = 0
            replay.backward_index = 0

    def set_save_path(self, path: str):
        self._save_path = path
        if not self._save_registered and path:
            atexit.register(lambda: self.save_all_to_files(path))
            self._save_registered = True

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
            for i, (orig_idx, replay_idx) in enumerate(zip(orig_flat, replay_flat)):
                orig_set = set(orig_idx.tolist()) - {-1}
                replay_set = set(replay_idx.tolist()) - {-1}
                if len(replay_set) == 0:
                    continue
                if len(orig_set & replay_set) < len(replay_set) * 0.7:
                    raise AssertionError(f"token {i} failed replay check, {len(orig_set & replay_set)=} {len(replay_set)=}")
        except Exception as e:
            logger.error(f"Rollout Replay Check Failed - Stage: {self.stage}, rank: {_get_rank()}")
            logger.error(f"original top_indices: {orig_top_indices}")
            logger.error(f"replay top_indices (padding removed): {top_indices}")
            dumper.dump("orig_top_indices", orig_top_indices)
            dumper.dump("replay_top_indices", top_indices)
            raise e


class RoutingReplayManager(BaseReplayManager):
    name = "routing"
    filename = "routing_replay.pt"
    data_key = "rollout_routed_experts"
    needs_moe_layer_indices = True
    if_sp_region = True
    squeeze_batch_for_load_from_file = False


class IndexerReplayManager(BaseReplayManager):
    name = "indexer"
    filename = "indexer_replay.pt"
    data_key = "rollout_indexer_topk"
    needs_moe_layer_indices = False
    if_sp_region = False
    squeeze_batch_for_load_from_file = True  # indexer has (batch, seq, topk) format, squeeze batch dim


routing_replay_manager = RoutingReplayManager()
indexer_replay_manager = IndexerReplayManager()
all_replay_managers = [routing_replay_manager, indexer_replay_manager]
