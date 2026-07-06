from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

_SourceGetter = Callable[[], Iterable[tuple[str, torch.Tensor]]]


@dataclass
class MainCastContext:
    optimizer: object
    model_chunks: list
    extras_getter: _SourceGetter
    check: bool


class TensorBackuper(ABC):
    @staticmethod
    def create(source_getter, single_tag, main_cast_ctx: "MainCastContext | None" = None):
        if main_cast_ctx is not None:
            return _TensorBackuperMainCast(source_getter=source_getter, ctx=main_cast_ctx)
        if single_tag is None:
            return _TensorBackuperNormal(source_getter=source_getter)
        else:
            return _TensorBackuperNoop(source_getter=source_getter, single_tag=single_tag)

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


class _TensorBackuperNormal(TensorBackuper):
    def __init__(self, source_getter):
        super().__init__(source_getter=source_getter)
        self._backups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

    @property
    def backup_tags(self):
        return list(self._backups)

    def get(self, tag: str):
        return self._backups[tag]

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        backup_dict = self._backups[tag]
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


_CHECK_NUM_CYCLES = 2


class _TensorBackuperMainCast(TensorBackuper):
    """Rematerialize weights from the optimizer's fp32 master weights (same cast +
    param all-gather as the step end, so bit-identical) instead of a pinned CPU copy;
    only `extras_getter` tensors keep a small pinned backup. With `check`, the first
    `_CHECK_NUM_CYCLES` restores are SHA256-verified against backup time."""

    def __init__(self, source_getter, ctx: MainCastContext):
        super().__init__(source_getter=source_getter)
        self._ctx = ctx
        self._extras_backup: dict[str, torch.Tensor] = {}
        self._extras_backup_by_id: dict[int, torch.Tensor] = {}
        self._backup_count = 0
        self._expected_hashes: dict[str, str] | None = None

    @property
    def backup_tags(self):
        return ["actor"]

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        assert tag == "actor", f"main-cast restore supports only the 'actor' tag, got {tag}"
        for name, tensor in self._ctx.extras_getter():
            if name not in self._extras_backup:
                self._extras_backup[name] = torch.empty_like(tensor, device=torch.device("cpu"), pin_memory=True)
            self._extras_backup[name].copy_(tensor.detach(), non_blocking=True)
            self._extras_backup_by_id[id(tensor)] = self._extras_backup[name]
        torch.cuda.synchronize()
        self._backup_count += 1
        if self._ctx.check and self._backup_count <= _CHECK_NUM_CYCLES:
            self._expected_hashes = self._compute_hashes()
        else:
            self._expected_hashes = None

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        assert tag == "actor", f"main-cast restore supports only the 'actor' tag, got {tag}"
        optimizers = getattr(self._ctx.optimizer, "chained_optimizers", [self._ctx.optimizer])
        for optimizer in optimizers:
            optimizer._copy_main_params_to_model_params()
        for model_chunk in self._ctx.model_chunks:
            model_chunk.start_param_sync(force_sync=True)
        for name, tensor in self._ctx.extras_getter():
            tensor.copy_(self._extras_backup[name], non_blocking=True)
        torch.cuda.synchronize()
        if self._expected_hashes is not None:
            self._verify_hashes()

    def get(self, tag: str):
        assert tag == "actor", f"main-cast restore supports only the 'actor' tag, got {tag}"
        # update_weights runs while non-param TMS regions are paused: extras must
        # be read from their pinned backup, params are live GPU tensors.
        return {
            name: self._extras_backup_by_id.get(id(tensor), tensor.detach()) for name, tensor in self._source_getter()
        }

    def _compute_hashes(self) -> dict[str, str]:
        from miles.backends.megatron_utils.ci_utils import _hash_tensor_sha256

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


class _TensorBackuperNoop(TensorBackuper):
    def __init__(self, source_getter, single_tag):
        super().__init__(source_getter=source_getter)
        self._single_tag = single_tag
        # Sanity check for safety
        self._backup_hash_dict = None

    @property
    def backup_tags(self):
        return [self._single_tag]

    def get(self, tag: str):
        ans = dict(self._source_getter())
        ans = {k: v.detach() for k, v in ans.items()}
        assert _compute_hash_dict(ans) == self._backup_hash_dict
        return ans

    def backup(self, tag: str) -> None:
        assert tag == self._single_tag
        self._backup_hash_dict = _compute_hash_dict(dict(self._source_getter()))
        torch.cuda.synchronize()

    def restore(self, tag: str) -> None:
        assert tag == self._single_tag
        assert _compute_hash_dict(dict(self._source_getter())) == self._backup_hash_dict
        torch.cuda.synchronize()


def _compute_hash_dict(tensors: dict[str, torch.Tensor]):
    return {k: _compute_hash_tensor(v) for k, v in tensors.items()}


def _compute_hash_tensor(x: torch.Tensor):
    # Not a real/good hash, but pretty fast
    x = x.contiguous().view(-1).view(torch.uint8)

    alignment = 4
    if (remainder := (x.numel() % alignment)) != 0:
        x = torch.nn.functional.pad(x, (0, alignment - remainder))

    x = x.view(torch.uint32).sum()
    return x.item()
