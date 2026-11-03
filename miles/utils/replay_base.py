import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait

import torch
import torch.distributed as dist

from miles.platforms import current_platform
from miles.utils.timer import Timer

logger = logging.getLogger(__name__)


def _get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _worker_device() -> dict:
    """`ThreadPoolExecutor` kwargs binding its thread to this process's accelerator.

    `Replay.record` allocates pinned memory on the worker, and a thread's first accelerator call
    would otherwise target device 0, which need not be this rank's.
    """
    if not torch.accelerator.is_available():
        return {}
    return {"initializer": torch.accelerator.set_device_index, "initargs": (torch.accelerator.current_device_index(),)}


class Replay:
    def __init__(self, stream_idx: int | None = None):
        self.stream_idx = stream_idx
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list: list[torch.Tensor] = []
        # popped before the forward: a compiled block that pops specializes on forward_index
        self.staged_forward_ids: torch.Tensor | None = None

    def record(self, top_indices: torch.Tensor):
        if not current_platform.dynamic_shapes:
            # keep the ids on the host until popped: on the device each new length compiles a new program
            self.top_indices_list.append(top_indices.to("cpu", copy=True))
            return
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
        if top_indices is None:
            raise IndexError(f"pop_forward: entry {self.forward_index} was already released by its backward pop")
        self.forward_index += 1
        return top_indices.to(current_platform.current_device())

    def pop_backward(self) -> torch.Tensor:
        # The train pass is the step's last pass that pops, and its backward (recompute) pop is the
        # entry's last use: release the pinned entry here, so the step's host memory shrinks as the
        # pass advances. Without recompute nothing pops backward and `clear` releases them.
        top_indices = self.top_indices_list[self.backward_index]
        if top_indices is None:
            raise IndexError(f"pop_backward: entry {self.backward_index} was already released by its backward pop")
        self.top_indices_list[self.backward_index] = None
        self.backward_index += 1
        return top_indices.to(current_platform.current_device())

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        self.staged_forward_ids = None

    def clear_forward(self):
        self.forward_index = 0


class BaseReplayManager:
    name: str = ""
    filename: str = ""
    # Defined by each subclass: the rollout column to replay, and whether sequence parallelism
    # splits its rows across tensor-parallel ranks.
    data_key: str
    if_sp_region: bool
    replay_check_min_overlap_ratio = 0.0  # 0.0 = mismatch only on zero overlap
    # True when replayed indices are token/KV positions (indexer): rebase the
    # per-sample 0-based rollout indices onto the packed training sequence.
    replay_indices_are_token_positions = False
    # True when a train-data dump keeps the rollout column replay consumed. Only R3 is
    # kept: indexer top-k takes about 0.5 MB per token on DeepSeek-V3.2.
    kept_in_train_dump = False

    def __init__(self):
        self.replays: list[Replay] = []
        self.current: Replay | None = None
        self.enabled = False
        self.stage = "fallthrough"
        self.register_replay_list_func = None
        # This step's fills, one per micro batch: `fill(i)` records micro batch i's entry in every
        # replay. They run in order on `_worker` from `attach` on, ahead of the forwards that pop
        # their entries; the first `_ready` are known to be done. `_finish` runs on the main thread
        # once the last fill has been waited for.
        self._worker: ThreadPoolExecutor | None = None
        self._fills: list[Future] = []
        self._ready = 0
        self._finish: Callable[[], None] | None = None
        # Set once a fill failed or the step ended: the worker skips the fills still queued.
        self._stopped = False
        self._fill_seconds = 0.0
        # With output store columns: the ReadSlots their rows are read into, kept across steps.
        self.read_slots = None

    def create_replay(self, stream_idx: int | None = None) -> Replay:
        replay = Replay(stream_idx=stream_idx)
        self.replays.append(replay)
        return replay

    def set_current(self, replay: Replay):
        self.current = replay

    def get_current(self) -> Replay | None:
        return self.current

    def attach(self, fill: Callable[[int], None], total: int, finish: Callable[[], None] | None = None) -> None:
        """Take this step's source, dropping the previous step's entries; returns without waiting.

        `fill(i)` for i < `total` records micro batch i's entry in every replay; they run in order
        on this manager's worker thread, starting now, as far ahead of the pops as the reads allow.
        `finish` runs on the main thread after the last of them has been waited for.
        """
        self.clear_all()
        if self._worker is None:
            self._worker = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"replay-{self.name}", **_worker_device()
            )
        self._stopped = False
        self._fill_seconds = 0.0
        self._finish = finish
        self._fills = [self._worker.submit(self._run, fill, i) for i in range(total)]

    def _run(self, fill: Callable[[int], None], index: int) -> None:
        if self._stopped:
            return
        start = time.perf_counter()
        try:
            fill(index)
        except BaseException:
            self._stopped = True
            raise
        self._fill_seconds += time.perf_counter() - start

    def wait_filled(self, cursor: int) -> None:
        """Block until micro batches up to and including `cursor` are filled; the only wait a pop does.

        A fill's failure is raised here, at the first pop that needs it.
        """
        if not self._fills or self._ready > cursor:
            return
        start = time.perf_counter()
        for future in self._fills[self._ready : cursor + 1]:
            future.result()
            self._ready += 1
        Timer().add("replay_wait", time.perf_counter() - start)
        if self._ready == len(self._fills) and self._finish is not None:
            Timer().add("replay_fill", self._fill_seconds)
            finish, self._finish = self._finish, None
            finish()

    def clear_all(self):
        """Drop this step's entries; blocks until the worker has stopped filling.

        The fill still running appends to the queues, so it is joined before they are cleared.
        """
        self._stopped = True
        for future in self._fills:
            future.cancel()
        wait(self._fills)
        self._fills, self._ready, self._finish = [], 0, None
        for replay in self.replays:
            replay.clear()

    def clear_all_forward(self):
        for replay in self.replays:
            replay.clear_forward()

    def get_replay_indices(self, n_tokens, topk):
        """Consume one forward's indices, independently of how its scores are chunked."""
        if not self.enabled:
            return None
        replay = self.get_current()
        if self.stage == "replay_forward":
            top_indices = replay.staged_forward_ids
            if top_indices is None:
                self.wait_filled(replay.forward_index)
                top_indices = replay.pop_forward()
        elif self.stage == "replay_backward":
            self.wait_filled(replay.backward_index)
            top_indices = replay.pop_backward()
        else:
            return None
        assert (
            top_indices.shape[0] == n_tokens
        ), f"rank {_get_rank()}: replay n_tokens {top_indices.shape[0]} does not match scores n_tokens {n_tokens}"
        assert (
            top_indices.shape[1] == topk
        ), f"replay topk does not match expected topk, replay topk {top_indices.shape[1]}, topk {topk}"
        return top_indices

    def record_indices(self, top_indices):
        if self.enabled and self.stage == "record":
            self.get_current().record(top_indices)

    @staticmethod
    def fill_replay_padding(top_indices, num_scores):
        # padding rows (all -1) take arange; branchless so fullgraph traces it
        all_invalid = (top_indices == -1).all(dim=-1)
        ar = torch.arange(top_indices.shape[1], device=top_indices.device, dtype=top_indices.dtype) % num_scores
        return torch.where(all_invalid.unsqueeze(-1), ar, top_indices)

    def get_topk_fn(self, old_topk_fn, return_probs):
        manager = self

        def new_topk_fn(scores, topk, *args, **kwargs):
            top_indices = manager.get_replay_indices(scores.shape[0], topk)
            if top_indices is None:
                result = old_topk_fn(scores, topk, *args, **kwargs)
                manager.record_indices(result[1] if return_probs else result)
                return result
            if manager.enable_check_replay_result:
                manager.check_replay_result(old_topk_fn, scores, topk, top_indices, *args, **kwargs)
            top_indices = manager.fill_replay_padding(top_indices, scores.shape[1])
            return (scores.gather(1, top_indices), top_indices) if return_probs else top_indices

        return new_topk_fn

    def register_to_module(self, module, attr_name: str, stream_idx: int | None = None) -> Replay | None:
        if not self.enabled:
            return None
        replay = self.create_replay(stream_idx=stream_idx)
        setattr(module, attr_name, replay)
        manager = self

        def pre_forward_hook(*args, **kwargs):
            manager.set_current(replay)

        module.register_forward_pre_hook(pre_forward_hook)
        return replay

    def check_replay_result(self, old_topk_fn, scores, topk, top_indices, *args, **kwargs):
        """
        CI checker for R3. Only enable when enable_check_replay_result=True.
        Per token, measure the overlap between the training engine's recomputed
        topk and the replayed topk (ignoring -1 padding). A token is mismatched
        when overlap < replay_check_min_overlap_ratio of its valid picks (ratio 0.0 =
        only zero overlap). Raise when the mismatched fraction exceeds
        replay_check_max_mismatch_fraction.
        """
        orig_top_indices = old_topk_fn(scores, topk, *args, **kwargs)
        self.check_replay_indices(orig_top_indices, top_indices)

    def check_replay_indices(self, orig_top_indices, top_indices):
        """Check the whole forward, including when selection was computed in chunks."""
        if isinstance(orig_top_indices, tuple):
            _, orig_top_indices = orig_top_indices

        orig_flat = orig_top_indices.view(-1, orig_top_indices.shape[-1])  # [n_tokens, topk]
        replay_flat = top_indices.view(-1, top_indices.shape[-1])
        valid_orig = orig_flat != -1
        valid_replay = replay_flat != -1

        # token-wise set overlap via a membership mask, avoiding the O(n*topk^2)
        # all-pairs tensor; -1 padding is routed to a sentinel column
        n_kv = int(torch.maximum(orig_flat.max(), replay_flat.max()).clamp_min(0)) + 1
        membership = orig_flat.new_zeros((orig_flat.shape[0], n_kv + 1), dtype=torch.bool)
        membership.scatter_(1, torch.where(valid_orig, orig_flat, n_kv).long(), True)
        replay_idx = torch.where(valid_replay, replay_flat, n_kv).long()
        hit = membership.gather(1, replay_idx) & valid_replay  # [n_tokens, topk]

        overlap = hit.sum(dim=1)
        valid_count = valid_replay.sum(dim=1)
        is_padding = valid_count == 0
        if self.replay_check_min_overlap_ratio == 0.0:
            required = torch.ones_like(overlap)  # legacy: mismatch only on zero overlap
        else:
            required = (valid_count * self.replay_check_min_overlap_ratio).ceil().to(overlap.dtype)
        is_mismatch = (overlap < required) & ~is_padding

        mismatch_count = is_mismatch.sum().item()
        threshold = float(os.environ.get("MILES_TEST_R3_THRESHOLD", self.replay_check_max_mismatch_fraction))
        mismatch_threshold = threshold * orig_flat.shape[0]
        logger.info(
            f"Replay check (rank {_get_rank()}, stage {self.stage}): "
            f"mismatch {mismatch_count}/{orig_flat.shape[0]} tokens, threshold {mismatch_threshold:.0f}"
        )
        if mismatch_count == 0:
            return

        mismatch_indices = is_mismatch.nonzero(as_tuple=False).squeeze(1)
        for idx in mismatch_indices[:10]:
            i = idx.item()
            lines = []
            for j in range(max(0, i - 3), min(len(orig_flat), i + 4)):
                marker = " <<<" if j == i else ""
                lines.append(f"  token {j}: orig={orig_flat[j].tolist()}, replay={replay_flat[j].tolist()}{marker}")
            logger.warning(
                f"Replay check (rank {_get_rank()}, stage {self.stage}): "
                f"token {i} overlap {overlap[i].item()}/{valid_count[i].item()}, topk={orig_flat.shape[-1]}\n"
                + "\n".join(lines)
            )

        if mismatch_count > mismatch_threshold:
            raise AssertionError(f"R3 mismatch tokens ({mismatch_count}) > threshold ({mismatch_threshold:.0f})")


class RoutingReplayManager(BaseReplayManager):
    name = "routing"
    filename = "routing_replay.pt"
    data_key = "rollout_routed_experts"
    if_sp_region = True
    enable_check_replay_result = False
    replay_check_max_mismatch_fraction = 1e-2
    kept_in_train_dump = True


class IndexerReplayManager(BaseReplayManager):
    name = "indexer"
    filename = "indexer_replay.pt"
    data_key = "rollout_indexer_topk"
    if_sp_region = False
    enable_check_replay_result = False
    replay_check_max_mismatch_fraction = 1e-2
    replay_check_min_overlap_ratio = 0.8
    replay_indices_are_token_positions = True


routing_replay_manager = RoutingReplayManager()
indexer_replay_manager = IndexerReplayManager()
all_replay_managers = [routing_replay_manager, indexer_replay_manager]
