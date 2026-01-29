from curses import tigetflag
import os
import atexit
from pathlib import Path

import torch
import torch.distributed as dist

import logging
logger = logging.getLogger(__name__)

def _get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


class Replay:
    def __init__(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list: list[torch.Tensor] = []
        self.total_lengths_list: list[torch.Tensor] = []
        self.response_lengths_list: list[torch.Tensor] = []

    def record(self, top_indices: torch.Tensor, total_lengths: torch.Tensor = None, response_lengths: torch.Tensor = None):
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)
        
        assert total_lengths is not None and response_lengths is not None
        if total_lengths is not None:
            self.total_lengths_list.append(total_lengths)
        if response_lengths is not None:
            self.response_lengths_list.append(response_lengths)

    def pop_forward(self) -> torch.Tensor:
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def pop_backward(self) -> torch.Tensor:
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def get_forward_lengths(self, is_forward: bool) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        idx = self.forward_index - 1 if is_forward else self.backward_index - 1
        return self.total_lengths_list[idx], self.response_lengths_list[idx]

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        self.total_lengths_list = []
        self.response_lengths_list = []

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
                probs, top_indices = old_topk_fn(scores, topk, *args, **kwargs)
                replay.record(top_indices)
                return probs, top_indices

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

    def _check_padding_with_valid_counts(self, top_indices, valid_kv_counts, topk, total_length):
        flat_indices = top_indices.view(-1, top_indices.shape[-1])
        flat_counts = valid_kv_counts.view(-1)
        num_positions = flat_indices.shape[0]
        
        assert flat_counts.shape[0] == num_positions, \
            f"valid_kv_counts shape {flat_counts.shape} doesn't match indices shape {flat_indices.shape}"
        
        invalid_positions = []
        
        for i in range(num_positions):
            indices = flat_indices[i].tolist()
            expected_valid = min(int(flat_counts[i].item()), topk)

            if all(x == -1 for x in indices):
                if i < total_length:
                    print(f"[WARNING] rank {_get_rank()}: [{self.name}] Maybe invalid replay: {i} is before total length {total_length}")
                continue
            
            # (non-masked positions must have valid values)
            for j in range(expected_valid):
                if indices[j] == -1:
                    invalid_positions.append({'pos': i, 'index': j, 'expected_valid': expected_valid, 'indices': indices})
                    break
            
        return len(invalid_positions) == 0, invalid_positions

    def check_replay_result(self, old_topk_fn, scores, topk, top_indices, **kwargs):
        if os.environ.get("MILES_CHECK_REPLAY_RESULT", "0") == "0":
            return
        print(f"rank {_get_rank()}: [{self.name}] check_replay_result called, scores shape: {scores.shape}, top_indices shape: {top_indices.shape}")
        valid_kv_counts = kwargs.pop('valid_kv_counts', None)
        replay = self.get_current()
        
        total_lengths, response_lengths = replay.get_forward_lengths(True if self.stage == "replay_forward" else False)
                
        orig_top_indices = old_topk_fn(scores, topk, **kwargs)
        if isinstance(orig_top_indices, tuple):
            _, orig_top_indices = orig_top_indices
        
        if valid_kv_counts is not None:
            padding_valid, invalid_positions = self._check_padding_with_valid_counts(top_indices, valid_kv_counts, topk, total_lengths[0])
            if not padding_valid:
                for i, (t, r) in enumerate(zip(total_lengths, response_lengths)):
                    prompt_len = t - r
                    print(f"  Sample {i}: total_len={t}, prompt_len={prompt_len}, response_len={r}", flush=True)

                logger.error(f"[{self.name}] Missing valid indices in non-masked positions!")
                for info in invalid_positions[:5]:
                    logger.error(f"  Position {info['pos']}: index {info['index']} should be valid but is -1 "
                                 f"(Megatron expects at least {info['expected_valid']} valid indices)")
                    logger.error(f"    indices: {info['indices']}")
                raise AssertionError(f"[{self.name}] Invalid replay: {len(invalid_positions)} positions missing valid indices")
        
        mismatched_tokens = []
        try:
            orig_flat = orig_top_indices.view(-1, orig_top_indices.shape[-1])
            replay_flat = top_indices.view(-1, top_indices.shape[-1])
            num_positions = replay_flat.shape[0]
            
            for i in range(num_positions):
                orig_set = set(orig_flat[i].tolist()) - {-1}
                replay_set = set(replay_flat[i].tolist()) - {-1}
                # Skip check if replay has no valid indices (all -1, e.g., early positions with no KV)
                if len(replay_set) == 0:
                    continue
                min_match = len(replay_set) // 2
                num_match = len(orig_set & replay_set)
                if num_match < min_match:
                    mismatched_tokens.append({
                        'token_idx': i,
                        'num_match': num_match,
                        'num_expected': len(replay_set),
                        'min_match': min_match,
                        'orig_indices': orig_flat[i].tolist(),
                        'replay_indices': replay_flat[i].tolist(),
                    })
            
            if mismatched_tokens:
                raise AssertionError(f"rank {_get_rank()}: [{self.name}] {len(mismatched_tokens)} tokens failed replay check")

            print(f"rank {_get_rank()}: [{self.name}] replay test passed, shape: {top_indices.shape}")

        except Exception as e:
            print(f"[{self.name}] Replay check exception - Stage: {self.stage}, rank: {_get_rank()}", flush=True)
            torch.set_printoptions(threshold=float("inf"))
            
            if mismatched_tokens:
                print(f"\n[{self.name}] [Rank {_get_rank()}] {len(mismatched_tokens)} tokens mismatched", flush=True)
                print(f"Mismatched token indices: {[t['token_idx'] for t in mismatched_tokens]}", flush=True)
                print(f"\nFirst {min(5, len(mismatched_tokens))} mismatched tokens:", flush=True)
                
                for idx, mismatch in enumerate(mismatched_tokens[:5]):
                    print(f"\n--- Token {mismatch['token_idx']} (mismatch #{idx+1}) ---", flush=True)
                    print(f"  Match rate: {mismatch['num_match']}/{mismatch['num_expected']} (need >= {mismatch['min_match']})", flush=True)
                    print(f"  Expected (original): {mismatch['orig_indices']}", flush=True)
                    print(f"  Actual (replay):     {mismatch['replay_indices']}", flush=True)
            
            print(f"{'='*80}\n", flush=True)
            raise e


class RoutingReplayManager(BaseReplayManager):
    name = "routing"
    filename = "routing_replay.pt"
    data_key = "rollout_routed_experts"
    needs_moe_layer_indices = True
    if_sp_region = True


class IndexerReplayManager(BaseReplayManager):
    name = "indexer"
    filename = "indexer_replay.pt"
    data_key = "rollout_indexer_topk"
    needs_moe_layer_indices = False
    if_sp_region = False


routing_replay_manager = RoutingReplayManager()
indexer_replay_manager = IndexerReplayManager()
all_replay_managers = [routing_replay_manager, indexer_replay_manager]
