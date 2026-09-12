import hashlib
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

_SourceGetter = Callable[[], Iterable[tuple[str, torch.Tensor]]]


@dataclass(frozen=True)
class MainCastContext:
    # Writes this rank's owned shard from the master weights, as the train-step end does.
    cast_main_to_params: Callable[[], None]
    model_chunks: list
    extras_getter: _SourceGetter
    rematerializable_ids: set
    check: bool


class TensorBackuper(ABC):
    @staticmethod
    def create(source_getter, main_cast_ctx: "MainCastContext | None" = None):
        if main_cast_ctx is not None:
            return _TensorBackuperMainCast(source_getter=source_getter, ctx=main_cast_ctx)
        return _TensorBackuperNormal(source_getter=source_getter)

    def __init__(self, source_getter: _SourceGetter):
        self._source_getter = source_getter

    @property
    @abstractmethod
    def backup_tags(self):
        raise NotImplementedError

    @abstractmethod
    def get(self, tag: str):
        raise NotImplementedError

    @abstractmethod
    def backup(self, tag: str):
        raise NotImplementedError

    def copy(self, *, src_tag: str, dst_tag: str):
        raise NotImplementedError

    @abstractmethod
    def restore(self, tag: str):
        raise NotImplementedError


def _file_backed_like(param: torch.Tensor, path: str) -> torch.Tensor:
    """A host tensor shaped like `param` whose storage is a shared mapping of `path`, so the
    backup lives in the page cache and on disk instead of pinned RAM."""
    nbytes = param.numel() * param.element_size()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(max(nbytes, 1))
    flat = torch.from_file(path, shared=True, size=max(param.numel(), 1), dtype=param.dtype)
    return flat[: param.numel()].view(param.shape)


def _drop_from_page_cache(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


class _TensorBackuperNormal(TensorBackuper):
    def __init__(self, source_getter):
        super().__init__(source_getter=source_getter)
        self._backups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        self._disk_dir = os.environ.get("MILES_WEIGHT_BACKUP_DIR")
        if self._disk_dir:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            self._disk_dir = os.path.join(self._disk_dir, f"rank{rank:05d}")
        self._paths: dict[str, dict[str, str]] = defaultdict(dict)

    @property
    def backup_tags(self):
        return list(self._backups)

    def get(self, tag: str):
        assert tag in self._backups, f"tag {tag!r} was never backed up"
        return self._backups[tag]

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        if self._disk_dir:
            for name, param in self._source_getter():
                if name not in backup_dict:
                    path = os.path.join(self._disk_dir, tag, f"{hashlib.md5(name.encode()).hexdigest()}.bin")
                    backup_dict[name] = _file_backed_like(param, path)
                    self._paths[tag][name] = path
                backup_dict[name].copy_(param.detach())
                _drop_from_page_cache(self._paths[tag][name])
            return
        for name, param in self._source_getter():
            if name not in backup_dict:
                backup_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
            backup_dict[name].copy_(param.detach(), non_blocking=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def copy(self, *, src_tag: str, dst_tag: str):
        for name in self._backups[dst_tag]:
            self._backups[dst_tag][name].copy_(self._backups[src_tag][name])

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            assert name in backup_dict
            param.copy_(backup_dict[name], non_blocking=True)
        torch.cuda.synchronize()


class _TensorBackuperMainCast(TensorBackuper):
    """Rebuilds the actor weights instead of keeping a pinned CPU copy of them.

    Restore replays the step end's cast + all-gather, so it is bit-identical. Only
    `extras_getter` tensors keep a pinned backup. Non-actor tags (ref/teacher) have no
    master weights to rebuild from, so they keep full pinned copies via a delegated
    _TensorBackuperNormal.
    """

    _check_num_cycles = 2

    def __init__(self, source_getter, ctx: MainCastContext):
        super().__init__(source_getter=source_getter)
        self._ctx = ctx
        self._others = _TensorBackuperNormal(source_getter=source_getter)
        self._extras_backup: dict[str, torch.Tensor] = {}
        self._extras_backup_by_id: dict[int, torch.Tensor] = {}
        self._backup_count = 0
        self._expected_hashes: dict[str, str] | None = None

    @property
    def backup_tags(self):
        return ["actor", *self._others.backup_tags]

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        if tag != "actor":
            return self._others.backup(tag)
        for name, tensor in self._ctx.extras_getter():
            if name not in self._extras_backup:
                self._extras_backup[name] = torch.empty_like(tensor, device=torch.device("cpu"), pin_memory=True)
            self._extras_backup[name].copy_(tensor.detach(), non_blocking=True)
            self._extras_backup_by_id[id(tensor)] = self._extras_backup[name]
        torch.cuda.synchronize()
        self._backup_count += 1
        if self._ctx.check and self._backup_count <= self._check_num_cycles:
            self._expected_hashes = self._compute_hashes()
        else:
            self._expected_hashes = None

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        if tag != "actor":
            return self._others.restore(tag)
        self._ctx.cast_main_to_params()
        for model_chunk in self._ctx.model_chunks:
            model_chunk.start_param_sync(force_sync=True)
        for name, tensor in self._ctx.extras_getter():
            tensor.copy_(self._extras_backup[name], non_blocking=True)
        torch.cuda.synchronize()
        if self._expected_hashes is not None:
            self._verify_hashes()

    def get(self, tag: str):
        if tag != "actor":
            return self._others.get(tag)
        # Extras are paused during update_weights. Read them from the pinned backup.
        out = {}
        for name, tensor in self._source_getter():
            backup = self._extras_backup_by_id.get(id(tensor))
            if backup is None:
                assert (
                    id(tensor) in self._ctx.rematerializable_ids
                ), f"{name} is neither in the DDP param buffers nor in the extras backup"
                backup = tensor.detach()
            out[name] = backup
        return out

    def _compute_hashes(self) -> dict[str, str]:
        return {name: _hash_tensor_sha256(tensor) for name, tensor in self._source_getter()}

    def _verify_hashes(self) -> None:
        actual = self._compute_hashes()
        expected = self._expected_hashes
        assert expected is not None
        assert actual.keys() == expected.keys(), (
            f"main-cast restore changed the tensor set: "
            f"missing={sorted(expected.keys() - actual.keys())[:5]} "
            f"extra={sorted(actual.keys() - expected.keys())[:5]}"
        )
        mismatches = [name for name in expected if actual[name] != expected[name]]
        if mismatches:
            raise RuntimeError(
                f"main-cast weight restore is not bit-identical to the weights at "
                f"backup time for {len(mismatches)}/{len(expected)} tensors "
                f"(cycle {self._backup_count}): {mismatches[:20]}"
            )


def _hash_tensor_sha256(x: torch.Tensor) -> str:
    """Real (cryptographic) hash: a mismatch here has to mean a bug."""
    data = x.detach().cpu().contiguous()
    return hashlib.sha256(data.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
