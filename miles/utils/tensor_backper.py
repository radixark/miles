from collections import defaultdict
from typing import Dict, Iterable, Callable, Tuple, Any

import torch

NamedParametersGetter = Callable[[], Iterable[Tuple[str, torch.Tensor]]]


class TensorBackuper:
    def __init__(self, source_getter: NamedParametersGetter):
        self._source_getter = source_getter
        self._backups: Dict[str, Dict[str, torch.Tensor]] = defaultdict()

    @property
    def backup_tags(self):
        return list(self._backups)

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
